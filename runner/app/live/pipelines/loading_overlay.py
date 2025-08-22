import asyncio
import asyncio
import concurrent.futures
import logging
import time
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F


class LoadingOverlayRenderer:
    def __init__(self) -> None:
        # Session tracking to invalidate caches when reload sessions change
        self._session_wallclock: float = 0.0
        self._active: bool = False
        self._show_overlay: bool = True
        # Fade configuration (seconds)
        self._fade_duration: float = 2.0

        # Cached size and base images
        self._cached_size: Tuple[int, int] = (0, 0)
        self._base_image_color: Optional[Image.Image] = None
        self._base_image_gray: Optional[Image.Image] = None

        # Base tensor for current session and last-output cache
        self._base_tensor: Optional[torch.Tensor] = None
        self._base_tensor_wallclock: float = 0.0
        self._last_output_tensor: Optional[torch.Tensor] = None
        self._last_output_wallclock: float = 0.0
        self._base_max_age_seconds: float = 5.0

        # Text caching
        self._font: Optional[Any] = None
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

        # Dedicated executor so overlay rendering isn't starved by heavy default executor tasks
        self._executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="overlay-renderer"
        )

    def reset_session(self, session_wallclock: float) -> None:
        self._session_wallclock = session_wallclock
        # Force base images to rebuild next render while preserving cached size
        # and keeping spinner/text caches when the size is unchanged.
        self._base_image_color = None
        self._base_image_gray = None
        self._blended_rgba_cache.clear()
        self._dim_overlay_cache.clear()
        # Do not clear spinner/text or _cached_size to avoid heavy first-frame work.
        # Do not clear last-output cache here; that is cross-session state.

    def set_fade_duration(self, seconds: float) -> None:
        try:
            value = float(seconds)
        except Exception:
            return
        if value <= 0.0:
            return
        self._fade_duration = value

    def _ensure_base_images(self, w: int, h: int) -> None:
        # If session size or base not initialized or size changed, (re)create base images
        size_changed = (self._cached_size != (w, h))
        if size_changed or self._base_image_color is None:
            base_np = None
            if self._base_tensor is not None:
                try:
                    base_np = (self._base_tensor.clamp(0, 1) * 255).byte().cpu().numpy()[0]
                except Exception:
                    base_np = None
            if base_np is None or base_np.shape[0] != h or base_np.shape[1] != w:
                color = Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8), mode="RGB")
                gray = color.convert("L").convert("RGB")
            else:
                color = Image.fromarray(base_np[..., :3], mode="RGB")
                gray = color.convert("L").convert("RGB")
            self._base_image_color = color
            self._base_image_gray = gray
            self._cached_size = (w, h)
            # Invalidate dependent caches that depend on base or size
            self._blended_rgba_cache.clear()
            self._dim_overlay_cache.clear()
            # Only clear spinner/text if size actually changed
            if size_changed:
                self._spinner_frames = []
                self._spinner_radius = 0
                self._spinner_thickness = 0
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

    def _render_base_gpu_rgba(self, w: int, h: int, t: float) -> Image.Image:
        """
        Render the gray/dimmed base frame using GPU tensors if available, then return a PIL RGBA image.
        Falls back to CPU torch if CUDA is not available, but still avoids PIL for the heavy ops.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare base tensor in [0,1], shape (1, H, W, 3)
        if self._base_tensor is not None:
            base = self._base_tensor
            if base.shape[1] != h or base.shape[2] != w:
                bchw = base.permute(0, 3, 1, 2)
                bchw = F.interpolate(bchw, size=(h, w), mode="bilinear", align_corners=False)
                base = bchw.permute(0, 2, 3, 1)
        else:
            base = torch.full((1, h, w, 3), 0.5, dtype=torch.float32)

        base = base.to(device, non_blocking=True)

        # Grayscale conversion and blend
        r = base[..., 0:1]
        g = base[..., 1:2]
        b = base[..., 2:3]
        gray = (0.299 * r + 0.587 * g + 0.114 * b)
        gray3 = gray.expand_as(base)

        tq = max(0.0, min(1.0, t))
        blended = (1.0 - tq) * base + tq * gray3

        # Dim overlay factor, equivalent to compositing black with alpha
        dim_alpha = int(90 * t)
        dim_alpha = max(0, min(96, int(round(dim_alpha / 8.0) * 8)))
        dim = 1.0 - (dim_alpha / 255.0)
        blended = blended * dim

        # To CPU uint8
        out_np = (blended.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()[0]
        img = Image.fromarray(out_np, mode="RGB").convert("RGBA")
        return img

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
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, dim_alpha))  # type: ignore[arg-type]
            self._dim_overlay_cache[key] = overlay
        return overlay

    def _ensure_spinner_frames(self, w: int, h: int) -> None:
        if self._spinner_frames and self._spinner_radius > 0 and self._spinner_thickness > 0:
            return
        radius = max(8, int(min(w, h) * 0.035))
        thickness = max(3, int(min(w, h) * 0.008))
        canvas_size = 2 * radius + thickness

        # Supersampled canvas for smoother edges, later downsampled with Lanczos
        s = max(2, int(self._spinner_supersample_scale))
        hr_radius = radius * s
        hr_thickness = max(1, thickness * s)
        hr_canvas = 2 * hr_radius + hr_thickness

        # PIL resampling enums compatibility
        Resampling = getattr(Image, "Resampling", None)
        if Resampling is not None:
            lanczos = getattr(Resampling, "LANCZOS", 1)
            bicubic = getattr(Resampling, "BICUBIC", 3)
        else:
            lanczos = 1
            bicubic = 3

        # Draw a single high-res base spinner (270-degree arc) once
        hr_base = Image.new("RGBA", (hr_canvas, hr_canvas), (0, 0, 0, 0))  # type: ignore[arg-type]
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
            rotated = hr_base.rotate(angle, resample=bicubic, expand=False)  # type: ignore[arg-type]
            down = rotated.resize((canvas_size, canvas_size), resample=lanczos)  # type: ignore[arg-type]
            frames.append(down)

        self._spinner_frames = frames
        self._spinner_radius = radius
        self._spinner_thickness = thickness

    def _ensure_text(self, w: int, h: int) -> None:
        text = "Pipeline is reloading…"
        cx, cy = w // 2, h // 2
        radius = max(8, int(min(w, h) * 0.035))
        thickness = max(3, int(min(w, h) * 0.008))
        desired_font_size = max(14, int(min(w, h) * 0.04))
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

        tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # type: ignore[arg-type]
        td = ImageDraw.Draw(tmp)
        try:
            tb = td.textbbox((0, 0), text, font=font, stroke_width=2)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = (len(text) * 10, 20)
        text_img = Image.new("RGBA", (tw + 8, th + 8), (0, 0, 0, 0))  # type: ignore[arg-type]
        tdraw = ImageDraw.Draw(text_img)
        try:
            tdraw.text((4, 4), text, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 160))
        except Exception:
            tdraw.text((4, 4), text, font=font, fill=(255, 255, 255, 255))
        text_x = int(cx - text_img.width / 2)
        text_y = int(cy - radius - max(10, thickness) - text_img.height)

        self._text_image = text_img
        self._font = font
        self._font_size = desired_font_size
        self._text_pos = (text_x, text_y)

    def render(self, width: int, height: int) -> torch.Tensor:
        w = int(width)
        h = int(height)

        # Reset caches when session changes or size changes
        # No-op: session changes are controlled by begin_reload/end_reload

        # Ensure base images are ready
        self._ensure_base_images(w, h)

        # Time-based easing for grey-in and dimming
        fade_duration = self._fade_duration
        t = 1.0
        if self._session_wallclock:
            t = max(0.0, min(1.0, (time.time() - self._session_wallclock) / fade_duration))

        # Get blended base via GPU tensors when available, otherwise PIL path
        img: Image.Image
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            try:
                img = self._render_base_gpu_rgba(w, h, t)
            except Exception:
                # Fallback to CPU PIL path on error
                img = self._get_blended_base_rgba(t)
                dim_overlay = self._get_dim_overlay(w, h, t)
                if dim_overlay is not None:
                    img = Image.alpha_composite(img, dim_overlay)
        else:
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

    def update_last_frame(self, out_bhwc: torch.Tensor) -> None:
        try:
            with torch.no_grad():
                self._last_output_tensor = out_bhwc.detach().cpu().contiguous()
                self._last_output_wallclock = time.time()
        except Exception:
            # Best-effort cache, just log
            logging.error("Failed to update last frame", exc_info=True)

    def begin_reload(self) -> None:
        self._active = True
        now = time.time()
        # Choose base tensor if recent enough
        base: Optional[torch.Tensor] = None
        if self._last_output_tensor is not None and (now - self._last_output_wallclock) <= self._base_max_age_seconds:
            try:
                with torch.no_grad():
                    base = self._last_output_tensor.detach().cpu().contiguous()
            except Exception:
                base = None
        self._base_tensor = base
        self._base_tensor_wallclock = now
        # Reset per-session caches
        self.reset_session(now)

    def end_reload(self) -> None:
        self._active = False
        # Keep show preference for next time
        self._base_tensor = None
        self._base_tensor_wallclock = 0.0
        self.reset_session(0.0)

    def is_active(self) -> bool:
        return self._active and self._show_overlay

    def set_show_overlay(self, show_overlay: bool) -> None:
        self._show_overlay = bool(show_overlay)

    async def render_if_active(self, width: int, height: int):
        if not self.is_active():
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.render, width, height)

    async def prewarm(self, width: int, height: int) -> None:
        """
        Precompute spinner/text resources off the main thread to avoid first-frame stutters.
        """
        w = int(width)
        h = int(height)
        def _prewarm_sync() -> None:
            try:
                self._ensure_base_images(w, h)
                self._ensure_spinner_frames(w, h)
                self._ensure_text(w, h)
            except Exception:
                # Best-effort; ignore failures
                pass
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _prewarm_sync)
