import time
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class LoadingOverlayRenderer:
    def __init__(self) -> None:
        # Session tracking to invalidate caches when reload sessions change
        self._session_wallclock: float = 0.0

        # Cached size and base images
        self._cached_size: Tuple[int, int] = (0, 0)
        self._base_image_color: Optional[Image.Image] = None
        self._base_image_gray: Optional[Image.Image] = None

        # Text caching
        self._font: Optional[ImageFont.FreeTypeFont] = None
        self._font_size: int = 0
        self._text_image: Optional[Image.Image] = None  # RGBA
        self._text_pos: Tuple[int, int] = (0, 0)

        # Spinner caching
        self._spinner_frames: List[Image.Image] = []  # RGBA frames
        self._spinner_num_frames: int = 32
        self._spinner_radius: int = 0
        self._spinner_thickness: int = 0
        self._spinner_supersample_scale: int = 3

        # Blended base frames cache: key is t_index in [0, _blend_steps]
        self._blend_steps: int = 24
        self._blended_rgba_cache: Dict[int, Image.Image] = {}

        # Dimming overlay cache keyed by alpha
        self._dim_overlay_cache: Dict[int, Image.Image] = {}

    def reset_session(self, session_wallclock: float) -> None:
        self._session_wallclock = session_wallclock
        self._cached_size = (0, 0)
        self._base_image_color = None
        self._base_image_gray = None
        self._font = None
        self._font_size = 0
        self._text_image = None
        self._text_pos = (0, 0)
        self._spinner_frames = []
        self._spinner_radius = 0
        self._spinner_thickness = 0
        self._blended_rgba_cache.clear()
        self._dim_overlay_cache.clear()

    def _ensure_base_images(self, w: int, h: int, overlay_base_tensor: Optional[torch.Tensor]) -> None:
        # If session size or base not initialized or size changed, (re)create base images
        if self._cached_size != (w, h) or self._base_image_color is None:
            base_np = None
            if overlay_base_tensor is not None:
                try:
                    base_np = (overlay_base_tensor.clamp(0, 1) * 255).byte().cpu().numpy()[0]
                except Exception:
                    base_np = None
            if base_np is None or base_np.shape[0] != h or base_np.shape[1] != w:
                color = Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8), mode="RGB")
                gray = color
            else:
                color = Image.fromarray(base_np[..., :3], mode="RGB")
                gray = color.convert("L").convert("RGB")
            self._base_image_color = color
            self._base_image_gray = gray
            self._cached_size = (w, h)
            # Invalidate dependent caches
            self._blended_rgba_cache.clear()
            self._dim_overlay_cache.clear()
            self._spinner_frames = []
            self._text_image = None

    def _get_blended_base_rgba(self, t: float) -> Image.Image:
        # Quantize t to limit blends needed
        idx = max(0, min(self._blend_steps, int(round(t * self._blend_steps))))
        cached = self._blended_rgba_cache.get(idx)
        if cached is not None:
            return cached
        tq = idx / float(self._blend_steps)
        blended = Image.blend(self._base_image_color, self._base_image_gray, tq)
        blended_rgba = blended.convert("RGBA")
        self._blended_rgba_cache[idx] = blended_rgba
        return blended_rgba

    def _get_dim_overlay(self, w: int, h: int, t: float) -> Optional[Image.Image]:
        # Quantize alpha to reduce allocations
        dim_alpha = int(90 * t)
        # Round to nearest multiple of 8 for fewer unique overlays
        dim_alpha = max(0, min(96, int(round(dim_alpha / 8.0) * 8)))
        if dim_alpha <= 0:
            return None
        key = dim_alpha
        overlay = self._dim_overlay_cache.get(key)
        if overlay is None:
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, dim_alpha))
            self._dim_overlay_cache[key] = overlay
        return overlay

    def _ensure_spinner_frames(self, w: int, h: int) -> None:
        if self._spinner_frames and self._spinner_radius > 0 and self._spinner_thickness > 0:
            return
        radius = max(12, int(min(w, h) * 0.085))
        thickness = max(6, int(min(w, h) * 0.015))
        canvas_size = 2 * radius + thickness

        # Supersampled canvas for smoother edges, later downsampled with Lanczos
        s = max(2, int(self._spinner_supersample_scale))
        hr_radius = radius * s
        hr_thickness = max(1, thickness * s)
        hr_canvas = 2 * hr_radius + hr_thickness

        # PIL resampling enums compatibility
        Resampling = getattr(Image, "Resampling", None)
        lanczos = Resampling.LANCZOS if Resampling else getattr(Image, "LANCZOS", Image.BICUBIC)
        bicubic = Resampling.BICUBIC if Resampling else getattr(Image, "BICUBIC", Image.BILINEAR)

        # Draw a single high-res base spinner (270-degree arc) once
        hr_base = Image.new("RGBA", (hr_canvas, hr_canvas), (0, 0, 0, 0))
        d = ImageDraw.Draw(hr_base)
        hr_bbox = (
            hr_thickness // 2,
            hr_thickness // 2,
            hr_thickness // 2 + 2 * hr_radius,
            hr_thickness // 2 + 2 * hr_radius,
        )
        spinner_color = (255, 255, 255, 230)
        try:
            d.arc(hr_bbox, start=0.0, end=270.0, fill=spinner_color, width=hr_thickness)
        except Exception:
            d.ellipse(hr_bbox, outline=spinner_color, width=hr_thickness)

        # Generate rotated frames from the high-res base, then downsample once
        frames: List[Image.Image] = []
        for k in range(self._spinner_num_frames):
            angle = (k * (360.0 / self._spinner_num_frames)) % 360.0
            rotated = hr_base.rotate(angle, resample=bicubic, expand=False)
            down = rotated.resize((canvas_size, canvas_size), resample=lanczos)
            frames.append(down)

        self._spinner_frames = frames
        self._spinner_radius = radius
        self._spinner_thickness = thickness

    def _ensure_text(self, w: int, h: int) -> None:
        text = "Reloading pipeline..."
        cx, cy = w // 2, h // 2
        radius = max(12, int(min(w, h) * 0.085))
        thickness = max(6, int(min(w, h) * 0.015))
        desired_font_size = max(16, int(min(w, h) * 0.05))
        if self._text_image is not None and self._font_size == desired_font_size:
            return
        font = None
        for candidate in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]:
            try:
                font = ImageFont.truetype(candidate, desired_font_size)
                break
            except Exception:
                continue
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        td = ImageDraw.Draw(tmp)
        try:
            tb = td.textbbox((0, 0), text, font=font, stroke_width=2)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = (len(text) * 10, 20)
        text_img = Image.new("RGBA", (tw + 8, th + 8), (0, 0, 0, 0))
        tdraw = ImageDraw.Draw(text_img)
        try:
            tdraw.text((4, 4), text, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 160))
        except Exception:
            tdraw.text((4, 4), text, font=font, fill=(255, 255, 255, 255))
        text_x = int(cx - text_img.width / 2)
        text_y = int(cy + radius + max(10, thickness))

        self._text_image = text_img
        self._font = font
        self._font_size = desired_font_size
        self._text_pos = (text_x, text_y)

    def render(self, frame, overlay_start_wallclock: float, overlay_base_tensor: Optional[torch.Tensor]) -> torch.Tensor:
        # Import type locally to avoid circular import at module level
        from trickle import VideoFrame  # type: ignore

        if not isinstance(frame, VideoFrame):
            raise TypeError("render expects a VideoFrame")

        _, h, w, _ = frame.tensor.shape

        # Reset caches when session changes or size changes
        if self._session_wallclock != overlay_start_wallclock:
            self.reset_session(overlay_start_wallclock)

        # Ensure base images are ready
        self._ensure_base_images(w, h, overlay_base_tensor)

        # Time-based easing for grey-in and dimming
        fade_duration = 0.6
        t = 1.0
        if overlay_start_wallclock:
            t = max(0.0, min(1.0, (time.time() - overlay_start_wallclock) / fade_duration))

        # Get blended base RGBA and optional dim overlay
        img = self._get_blended_base_rgba(t)
        dim_overlay = self._get_dim_overlay(w, h, t)
        if dim_overlay is not None:
            img = Image.alpha_composite(img, dim_overlay)

        # Spinner
        self._ensure_spinner_frames(w, h)
        cx, cy = w // 2, h // 2
        if self._spinner_frames:
            angle = (time.time() * 180.0) % 360.0
            k = int((angle / 360.0) * self._spinner_num_frames) % self._spinner_num_frames
            spinner_img = self._spinner_frames[k]
            sx = int(cx - spinner_img.width / 2)
            sy = int(cy - spinner_img.height / 2)
            img.paste(spinner_img, (sx, sy), spinner_img)

        # Text
        self._ensure_text(w, h)
        if self._text_image is not None:
            img.paste(self._text_image, self._text_pos, self._text_image)

        # Convert back to torch tensor in (1, H, W, C), float32 in [0,1]
        img_rgb = img.convert("RGB")
        out_np = np.asarray(img_rgb).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_np).unsqueeze(0)
        return out_tensor

