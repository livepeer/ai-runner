import io
import logging

import soundfile as sf
import torch
from app.utils.errors import InferenceError
from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_torch_device
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.device = get_torch_device()
        self.model_id = model_id
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

    def _generate_speech(self, text: str, tts_steering: str) -> io.BytesIO:
        """Generate speech from text input using the text-to-speech model.

        Args:
            text: Text input for speech generation.
            tts_steering: Description of speaker to steer text to speech generation.

        Returns:
            buffer: BytesIO buffer containing the generated audio.
        """
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
            sf.write(buffer, generated_audio, samplerate=44100, format="WAV")
            buffer.seek(0)

            del input_ids, prompt_input_ids, generation, generated_audio

        return buffer

    def __call__(self, params) -> io.BytesIO:
        try:
            output = self._generate_speech(params.text, params.description)
        except torch.cuda.OutOfMemoryError as e:
            raise e
        except Exception as e:
            raise InferenceError(original_exception=e)

        return output

    def __str__(self) -> str:
        return f"TextToSpeechPipeline model_id={self.model_id}"
