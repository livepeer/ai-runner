import io
import logging
import os
import tempfile
import torch

import soundfile as sf
from pathlib import Path

from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from app.utils.errors import (
    InferenceError,
    ModelOOMError,
    GenerationError,
)

logger = logging.getLogger(__name__)


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        """Instantiate a Chatterbox-based text-to-speech pipeline.

        The constructor attempts to load a local checkpoint from the Hugging Face
        cache first (to avoid repeated downloads). If none is found, it falls
        back to fetching the weights from the Hub. All heavy objects live on
        the device returned by ``get_torch_device`` (GPU if available).
        """
        self.device = get_torch_device()
        self.model_id = model_id

        try:
            from chatterbox.src.chatterbox.tts import ChatterboxTTS  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover – optional
            ChatterboxTTS = None  # type: ignore     

        if ChatterboxTTS is None:
            raise ImportError(
                "ChatterboxTTS requested but chatterbox package not installed."
            )

        # Try to locate a local checkpoint first. The huggingface cache layout is:
        #   <hf_cache>/models--<model_id with slashes replaced by -->/snapshots/<hash>/
        safe_model_id = self.model_id.replace("/", "--")
        snapshots_root = Path(get_model_dir()) / f"models--{safe_model_id}" / "snapshots"
        ckpt_dir: Path | None = None
        if snapshots_root.exists():
            # Choose the most recently modified snapshot directory if multiple exist.
            snapshot_dirs = [d for d in snapshots_root.iterdir() if d.is_dir()]
            if snapshot_dirs:
                # Sort by modification time, newest first.
                snapshot_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                if len(snapshot_dirs) > 1:
                    logger.warning(
                        "Multiple snapshots found for %s; using the most recent: %s",
                        self.model_id,
                        snapshot_dirs[0],
                    )
                ckpt_dir = snapshot_dirs[0]

        if ckpt_dir and ckpt_dir.exists():
            logger.info("Loading ChatterboxTTS from local checkpoint: %s", ckpt_dir)
            self.model = ChatterboxTTS.from_local(ckpt_dir, self.device)
        else:
            logger.info("No local checkpoint found for %s — downloading from HuggingFace Hub.", self.model_id)
            self.model = ChatterboxTTS.from_pretrained(self.device)

        # Sample rate attribute is exposed for audio writing.
        self.sample_rate = getattr(self.model, "sr", 44100)
        if not hasattr(self.model, "sr"):
            logger.warning("Chatterbox model does not expose 'sr'; defaulting to 44100 Hz.")

    def _generate_speech(
        self,
        text: str,
        audio_prompt_bytes: bytes | None = None,
    ) -> io.BytesIO:
        """Generate speech audio from text.

        Args:
            text: Input text to synthesise.
            audio_prompt_path: Optional path to a reference audio clip used by the
                model for voice cloning / style transfer.

        Returns:
            BytesIO: In-memory WAV file containing the generated speech.
        """
        with torch.no_grad():
            # The Chatterbox generate method returns either a Tensor of shape
            # (1, n) or a NumPy-like 1-D array representing audio samples.
            if audio_prompt_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    tmp_audio.write(audio_prompt_bytes)
                    temp_path = tmp_audio.name
                try:
                    wav = self.model.generate(text, audio_prompt_path=temp_path)
                except Exception as e:
                    raise GenerationError(original_exception=e) from e
                finally:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
            else:
                try:
                    wav = self.model.generate(text)
                except Exception as e:
                    raise GenerationError(original_exception=e) from e
            audio = wav.squeeze(0).cpu().numpy() if torch.is_tensor(wav) else wav

            buffer = io.BytesIO()
            sf.write(buffer, audio, samplerate=self.sample_rate, format="WAV")
            buffer.seek(0)

        return buffer

    def __call__(self, params) -> io.BytesIO:
        try:
            audio_prompt_bytes: bytes | None = getattr(params, "audio_prompt_base64", None)
            output = self._generate_speech(
                params.text,
                audio_prompt_bytes=audio_prompt_bytes,
            )
        except torch.cuda.OutOfMemoryError as e:
            # Translate low-level OOM into domain-specific error.
            raise ModelOOMError(original_exception=e) from e
        except Exception as e:
            # If it's already a GenerationError, bubble up unchanged; otherwise wrap.
            if isinstance(e, GenerationError):
                raise
            raise GenerationError(original_exception=e) from e

        return output

    def __str__(self) -> str:
        return f"TextToSpeechPipeline(Chatterbox) model_id={self.model_id}"
