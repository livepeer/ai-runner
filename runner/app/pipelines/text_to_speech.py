import io
import logging
import torch

import soundfile as sf
from transformers import AutoTokenizer
from pathlib import Path

from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from app.utils.errors import InferenceError

logger = logging.getLogger(__name__)


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        """Instantiate a TTS pipeline.

        Depending on *model_id*, either the Parler-TTS or Chatterbox model will be
        loaded. We treat any model_id that contains the substring "chatterbox" (case
        insensitive) as a request for Chatterbox and default to Parler otherwise.
        """
        self.device = get_torch_device()
        self.model_id = model_id
        self._is_chatterbox = "chatterbox" in model_id.lower()

        if self._is_chatterbox:
            try:
                from chatterbox.src.chatterbox.tts import ChatterboxTTS  # type: ignore
            except ModuleNotFoundError:  # pragma: no cover – optional
                ChatterboxTTS = None  # type: ignore     

            if ChatterboxTTS is None:
                raise ImportError(
                    "ChatterboxTTS requested but chatterbox package not installed."
                )

            # Try using locally downloaded checkpoints first (downloaded via
            # `dl_checkpoints.sh`).
            base_cache = Path(get_model_dir()) / "models--ResembleAI--chatterbox" / "snapshots"
            ckpt_dir = None
            if base_cache.exists():
                # Grab first snapshot (should only be one) as the checkpoint dir.
                for d in base_cache.iterdir():
                    if d.is_dir():
                        ckpt_dir = d
                        break

            if ckpt_dir is not None and ckpt_dir.exists():
                self.model = ChatterboxTTS.from_local(ckpt_dir, self.device)
            else:
                # Fallback to HF hub (will also cache under above path).
                self.model = ChatterboxTTS.from_pretrained(self.device)

            # Sample rate attribute is exposed for audio writing.
            self.sample_rate = getattr(self.model, "sr", 44100)

        else:
            # Now import Parler-TTS which will use our patched register method
            from parler_tts import ParlerTTSForConditionalGeneration  # type: ignore

            # Default to Parler-TTS.
            kwargs = {"cache_dir": get_model_dir()}

            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                model_id,
                **kwargs,
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                **kwargs,
            )
            self.sample_rate = 44100

    def _generate_speech(
        self,
        text: str,
        tts_steering: str,
        audio_prompt_path: str | None = None,
    ) -> io.BytesIO:
        """Generate speech from text input using the text-to-speech model.

        Args:
            text: Text input for speech generation.
            tts_steering: Description of speaker to steer text to speech generation.
            audio_prompt_path: Optional path to an audio prompt.

        Returns:
            buffer: BytesIO buffer containing the generated audio.
        """
        # Branch based on backend.
        if self._is_chatterbox:
            with torch.no_grad():
                # Chatterbox generate returns a tensor with shape (1, n) or (n,)
                wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)
                audio = wav.squeeze(0).cpu().numpy() if torch.is_tensor(wav) else wav

                buffer = io.BytesIO()
                sf.write(buffer, audio, samplerate=self.sample_rate, format="WAV")
                buffer.seek(0)
        else:
            with torch.no_grad():
                input_ids = self.tokenizer(tts_steering, return_tensors="pt").input_ids.to(
                    self.device
                )
                prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(
                    self.device
                )

                generation = self.model.generate(
                    input_ids=input_ids, prompt_input_ids=prompt_input_ids
                )
                generated_audio = generation.cpu().numpy().squeeze()

                buffer = io.BytesIO()
                sf.write(buffer, generated_audio, samplerate=self.sample_rate, format="WAV")
                buffer.seek(0)

        return buffer

    def __call__(self, params) -> io.BytesIO:
        try:
            output = self._generate_speech(
                params.text,
                params.description,
                getattr(params, "audio_prompt_path", None),
            )
        except torch.cuda.OutOfMemoryError as e:
            raise e
        except Exception as e:
            raise InferenceError(original_exception=e)

        return output

    def __str__(self) -> str:
        backend = "Chatterbox" if self._is_chatterbox else "ParlerTTS"
        return f"TextToSpeechPipeline({backend}) model_id={self.model_id}"
