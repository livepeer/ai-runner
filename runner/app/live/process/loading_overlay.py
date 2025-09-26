# NGL this was vibe coded and I can't attest for the quality, but it works.

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

        # Grayscale base cache (no blending needed)
        self._gray_rgba_cache: Optional[Image.Image] = None

        # Dimming overlay cache keyed by alpha
        self._dim_overlay_cache: Dict[int, Image.Image] = {}

        # Cached background (gray + dim + text) to avoid per-frame compositing
        self._background_rgba: Optional[Image.Image] = None

        # Scratch canvas reused every frame to avoid reallocations
        self._scratch_rgba: Optional[Image.Image] = None

        # Cache the last composed output tensor by spinner frame index
        self._last_spinner_idx: int = -1
        self._last_output_tensor_cached: Optional[torch.Tensor] = None

        # Torch-based caches to minimize per-frame work
        self._background_tensor: Optional[torch.Tensor] = (
            None  # (1,H,W,3) float32 [0,1]
        )
        self._scratch_tensor: Optional[torch.Tensor] = None  # (1,H,W,3) float32 [0,1]
        self._spinner_tensors_rgba: List[Optional[torch.Tensor]] = (
            []
        )  # each (H,W,4) float32 [0,1]
        self._spinner_roi: Tuple[int, int, int, int] = (0, 0, 0, 0)  # sx, sy, sw, sh

        # Dedicated executor so overlay rendering isn't starved by heavy default executor tasks
        self._executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="overlay-renderer"
            )
        )
        # Background executor for non-blocking precompute
        self._bg_executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="overlay-prewarm"
            )
        )
        self._spinner_building: bool = False
        # Background executor to offload precomputation without blocking render
        self._bg_executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="overlay-prewarm"
            )
        )
        self._spinner_building: bool = False

    def reset_session(self, session_wallclock: float) -> None:
        self._session_wallclock = session_wallclock
        # Force base images to rebuild next render while preserving cached size
        # and keeping spinner/text caches when the size is unchanged.
        self._base_image_color = None
        self._base_image_gray = None
        self._gray_rgba_cache = None
        self._dim_overlay_cache.clear()
        self._background_rgba = None
        self._scratch_rgba = None
        self._last_spinner_idx = -1
        self._last_output_tensor_cached = None
        self._background_tensor = None
        self._scratch_tensor = None
        self._spinner_tensors_rgba = []
        self._spinner_roi = (0, 0, 0, 0)
        self._spinner_building = False
        # Do not clear spinner/text or _cached_size to avoid heavy first-frame work.
        # Do not clear last-output cache here; that is cross-session state.

    def _ensure_base_images(self, w: int, h: int) -> None:
        # If session size or base not initialized or size changed, (re)create base images
        size_changed = self._cached_size != (w, h)
        if size_changed or self._base_image_color is None:
            base_np = None
            if self._base_tensor is not None:
                try:
                    base_np = (
                        (self._base_tensor.clamp(0, 1) * 255).byte().cpu().numpy()[0]
                    )
                except Exception:
                    base_np = None
            if base_np is None or base_np.shape[0] != h or base_np.shape[1] != w:
                color = Image.fromarray(
                    np.full((h, w, 3), 128, dtype=np.uint8), mode="RGB"
                )
                gray = color.convert("L").convert("RGB")
            else:
                color = Image.fromarray(base_np[..., :3], mode="RGB")
                gray = color.convert("L").convert("RGB")
            self._base_image_color = color
            self._base_image_gray = gray
            self._cached_size = (w, h)
            # Invalidate dependent caches that depend on base or size
            self._gray_rgba_cache = None
            self._dim_overlay_cache.clear()
            self._background_rgba = None
            self._scratch_rgba = None
            self._last_spinner_idx = -1
            self._last_output_tensor_cached = None
            self._background_tensor = None
            self._scratch_tensor = None
            self._spinner_tensors_rgba = []
            self._spinner_roi = (0, 0, 0, 0)
            self._spinner_building = False
            # Only clear spinner/text if size actually changed
            if size_changed:
                self._spinner_frames = []
                self._spinner_radius = 0
                self._spinner_thickness = 0
                self._text_image = None

    def _get_gray_base_rgba(self) -> Image.Image:
        # Return cached grayscale RGBA if available
        if self._gray_rgba_cache is not None:
            return self._gray_rgba_cache
        # Convert grayscale to RGBA and cache it
        if self._base_image_gray is not None:
            gray_rgba = self._base_image_gray.convert("RGBA")
            self._gray_rgba_cache = gray_rgba
            return gray_rgba
        else:
            # Fallback to a default gray image if base_image_gray is None
            gray = Image.new("RGB", self._cached_size, (128, 128, 128))
            gray_rgba = gray.convert("RGBA")
            self._gray_rgba_cache = gray_rgba
            return gray_rgba

    def _ensure_background_rgba(self, w: int, h: int) -> Image.Image:
        if self._background_rgba is not None:
            return self._background_rgba
        base = self._get_gray_base_rgba()
        dim_overlay = self._get_dim_overlay(w, h)
        bg = Image.alpha_composite(base, dim_overlay)
        if self._text_image is not None:
            bg.paste(self._text_image, self._text_pos, self._text_image)
        self._background_rgba = bg
        # Also cache as torch tensor for fast per-frame updates
        img_rgb = bg.convert("RGB")
        out_np = np.asarray(img_rgb).astype(np.float32) / 255.0
        self._background_tensor = torch.from_numpy(out_np).unsqueeze(0)
        if self._background_tensor is not None:
            self._scratch_tensor = self._background_tensor.clone()
        return bg

    def _render_base_gpu_rgba(self, w: int, h: int) -> Image.Image:
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
                bchw = F.interpolate(
                    bchw, size=(h, w), mode="bilinear", align_corners=False
                )
                base = bchw.permute(0, 2, 3, 1)
        else:
            base = torch.full((1, h, w, 3), 0.5, dtype=torch.float32)

        base = base.to(device, non_blocking=True)

        # Grayscale conversion (full grayscale, no blending)
        r = base[..., 0:1]
        g = base[..., 1:2]
        b = base[..., 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray3 = gray.expand_as(base)

        # Apply full dimming (equivalent to t=1.0)
        dim_alpha = 90
        dim = 1.0 - (dim_alpha / 255.0)
        dimmed = gray3 * dim

        # To CPU uint8
        out_np = (dimmed.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()[0]
        img = Image.fromarray(out_np, mode="RGB").convert("RGBA")
        return img

    def _get_dim_overlay(self, w: int, h: int) -> Image.Image:
        # Always use full dimming (equivalent to t=1.0)
        dim_alpha = 90
        key = dim_alpha
        overlay = self._dim_overlay_cache.get(key)
        if overlay is None:
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, dim_alpha))  # type: ignore[arg-type]
            self._dim_overlay_cache[key] = overlay
        return overlay

    def _ensure_spinner_frames(self, w: int, h: int) -> None:
        if (
            self._spinner_frames
            and self._spinner_radius > 0
            and self._spinner_thickness > 0
        ):
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
        # Initialize lazy tensor cache; tensors are built per-index on demand
        self._spinner_tensors_rgba = [None] * len(self._spinner_frames)

    def _ensure_spinner_frames_async(self, w: int, h: int) -> None:
        if self._spinner_building:
            return
        self._spinner_building = True

        def _build() -> None:
            try:
                self._ensure_spinner_frames(w, h)
            finally:
                self._spinner_building = False

        try:
            self._bg_executor.submit(_build)
        except Exception:
            self._spinner_building = False

    def _get_spinner_tensor_rgba(self, k: int) -> Optional[torch.Tensor]:
        if not self._spinner_frames:
            return None
        if not self._spinner_tensors_rgba or k >= len(self._spinner_tensors_rgba):
            self._spinner_tensors_rgba = [None] * len(self._spinner_frames)
        t = self._spinner_tensors_rgba[k]
        if t is None:
            f = self._spinner_frames[k]
            arr = np.asarray(f.convert("RGBA")).astype(np.float32) / 255.0
            t = torch.from_numpy(arr)
            self._spinner_tensors_rgba[k] = t
        return t

    def _ensure_text(self, w: int, h: int) -> None:
        text = "Loading pipeline…"
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
            tdraw.text(
                (4, 4),
                text,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 160),
            )
        except Exception:
            tdraw.text((4, 4), text, font=font, fill=(255, 255, 255, 255))
        text_x = int(cx - text_img.width / 2)
        text_y = int(cy - radius - max(10, thickness) - text_img.height)

        self._text_image = text_img
        self._font = font
        self._font_size = desired_font_size
        self._text_pos = (text_x, text_y)
        # Invalidate background since text was (re)built
        self._background_rgba = None
        self._last_spinner_idx = -1
        self._last_output_tensor_cached = None
        self._background_tensor = None
        self._scratch_tensor = None
        self._spinner_tensors_rgba = []
        self._spinner_roi = (0, 0, 0, 0)
        self._spinner_building = False

    # Removed composite-per-index helpers; using fast torch ROI blend instead

    def render_sync(self, width: int, height: int) -> torch.Tensor:
        w = int(width)
        h = int(height)

        # Ensure base images and text are ready; then reuse cached background
        self._ensure_base_images(w, h)
        self._ensure_text(w, h)
        self._ensure_background_rgba(w, h)
        # Determine spinner index for this frame (build spinner frames synchronously if missing)
        if not self._spinner_frames:
            self._ensure_spinner_frames(w, h)
        angle = (-time.time() * 180.0) % 360.0
        k = int((angle / 360.0) * self._spinner_num_frames) % self._spinner_num_frames
        # Compute spinner ROI if unknown
        if self._spinner_frames:
            sp0 = self._spinner_frames[0]
            sw, sh = sp0.width, sp0.height
            cx, cy = w // 2, h // 2
            sx = int(cx - sw / 2)
            sy = int(cy - sh / 2)
            self._spinner_roi = (sx, sy, sw, sh)
        # Ensure scratch/background tensors exist
        if self._scratch_tensor is None or self._background_tensor is None:
            # Guard for None
            if self._background_rgba is None:
                self._ensure_background_rgba(w, h)
            img_rgb = self._background_rgba.convert("RGB")
            out_np = np.asarray(img_rgb).astype(np.float32) / 255.0
            self._background_tensor = torch.from_numpy(out_np).unsqueeze(0)
            if self._background_tensor is not None:
                self._scratch_tensor = self._background_tensor.clone()
        # If spinner unchanged, reuse previous tensor
        if self._last_output_tensor_cached is not None and k == self._last_spinner_idx:
            return self._last_output_tensor_cached
        # Restore ROI from background
        sx, sy, sw, sh = self._spinner_roi
        if sw > 0 and sh > 0:
            self._scratch_tensor[:, sy : sy + sh, sx : sx + sw, :] = (
                self._background_tensor[:, sy : sy + sh, sx : sx + sw, :]
            )
            sp = self._get_spinner_tensor_rgba(k)
            if sp is not None:
                # Alpha blend into ROI
                sp_rgb = sp[..., :3]
                sp_a = sp[..., 3:4]
                roi = self._scratch_tensor[:, sy : sy + sh, sx : sx + sw, :]
                roi *= 1.0 - sp_a
                roi += sp_rgb * sp_a
        # Cache and return
        self._last_spinner_idx = k
        self._last_output_tensor_cached = self._scratch_tensor
        return self._scratch_tensor

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
        if (
            self._last_output_tensor is not None
            and (now - self._last_output_wallclock) <= self._base_max_age_seconds
        ):
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
        self._base_tensor = None
        self._base_tensor_wallclock = 0.0
        self.reset_session(0.0)

    def is_active(self) -> bool:
        return self._active

    async def render(self, width: int, height: int):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.render_sync, width, height)

    async def prewarm(self, width: int, height: int) -> None:
        """
        Precompute spinner/text resources off the main thread to avoid first-frame stutters.
        Only prewarms if the resolution has changed.
        """
        w = int(width)
        h = int(height)

        # Only prewarm if resolution changed
        if self._cached_size == (w, h):
            return

        def _prewarm_sync() -> None:
            try:
                self._ensure_base_images(w, h)
                self._ensure_spinner_frames(w, h)
                self._ensure_text(w, h)
                self._ensure_background_rgba(w, h)
            except Exception:
                # Best-effort; ignore failures
                pass

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._bg_executor, _prewarm_sync)
