import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Any, List, AsyncGenerator, Union, Optional

from app.pipelines.base import Pipeline
from app.pipelines.utils import get_model_dir, get_max_memory
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from huggingface_hub import file_download
from transformers import AutoConfig
from app.routes.utils import LLMResponse, LLMChoice, LLMMessage, LLMTokenUsage

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def validate(self):
        """Validate generation parameters"""
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if not 0 <= self.top_p <= 1:
            raise ValueError("Top_p must be between 0 and 1")
        if self.max_tokens < 1:
            raise ValueError("Max_tokens must be positive")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("Presence penalty must be between -2.0 and 2.0")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("Frequency penalty must be between -2.0 and 2.0")


class LLMPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        """
        Initialize the LLM Pipeline.

        Args:
            model_id: The identifier for the model to load
            use_8bit: Whether to use 8-bit quantization
            max_batch_size: Maximum batch size for inference
            max_num_seqs: Maximum number of sequences
            mem_utilization: GPU memory utilization target
            max_num_batched_tokens: Maximum number of batched tokens
        """
        logger.info("Initializing LLM pipeline")

        self.model_id = model_id
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model")
        base_path = os.path.join(get_model_dir(), folder_name)

        # Find the actual model path
        self.local_model_path = self._find_model_path(base_path)

        if not self.local_model_path:
            raise ValueError(f"Could not find model files for {model_id}")

        use_8bit = os.getenv("USE_8BIT", "").strip().lower() == "true"
        max_num_batched_tokens = int(os.getenv("MAX_NUM_BATCHED_TOKENS", "8192"))
        max_num_seqs = int(os.getenv("MAX_NUM_SEQS", "128"))
        max_model_len = int(os.getenv("MAX_MODEL_LEN", "8192"))
        mem_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
        tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
        pipeline_parallel_size = int(os.getenv("PIPELINE_PARALLEL_SIZE", "1"))

        if max_num_batched_tokens < max_model_len:
            max_num_batched_tokens = max_model_len
            logger.info(
                f"max_num_batched_tokens ({max_num_batched_tokens}) is smaller than max_model_len ({max_model_len}). This effectively limits the maximum sequence length to max_num_batched_tokens and makes vLLM reject longer sequences.")
            logger.info(f"setting 'max_model_len' to equal 'max_num_batched_tokens'")

    # Load config to check model compatibility
        try:
            config = AutoConfig.from_pretrained(self.local_model_path)
            num_heads = config.num_attention_heads
            num_layers = config.num_hidden_layers
            logger.info(
                f"Model has {num_heads} attention heads and {num_layers} layers")

            # Validate tensor parallelism
            if num_heads % tensor_parallel_size != 0:
                raise ValueError(
                    f"Total number of attention heads ({num_heads}) must be divisible "
                    f"by tensor parallel size ({tensor_parallel_size})."
                )

            # Validate pipeline parallelism
            if num_layers < pipeline_parallel_size:
                raise ValueError(
                    f"Pipeline parallel size ({pipeline_parallel_size}) cannot be larger "
                    f"than number of layers ({num_layers})."
                )

            # Validate total GPU requirement
            total_gpus_needed = tensor_parallel_size * pipeline_parallel_size
            max_memory = get_max_memory()
            if total_gpus_needed > max_memory.num_gpus:
                raise ValueError(
                    f"Total GPUs needed ({total_gpus_needed}) exceeds available GPUs "
                    f"({max_memory.num_gpus}). Reduce tensor_parallel_size ({tensor_parallel_size}) "
                    f"or pipeline_parallel_size ({pipeline_parallel_size})."
                )

            logger.info(f"Using tensor parallel size: {tensor_parallel_size}")
            logger.info(f"Using pipeline parallel size: {pipeline_parallel_size}")
            logger.info(f"Total GPUs used: {total_gpus_needed}")

        except Exception as e:
            logger.error(f"Error in parallelism configuration: {e}")
            raise

        engine_args = AsyncEngineArgs(
            model=self.local_model_path,
            tokenizer=self.local_model_path,
            trust_remote_code=True,
            dtype="auto",  # This specifies BFloat16 precision, TODO: Check GPU capabilities to set best type
            kv_cache_dtype="auto",  # or "fp16" if you want to force it
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=mem_utilization,
            max_num_seqs=max_num_seqs,
            enforce_eager=False,
            enable_prefix_caching=True,
            max_model_len=max_model_len
        )

        if use_8bit:
            engine_args.quantization = "bitsandbytes"
            engine_args.load_format = "bitsandbytes"
            logger.info("Using 8-bit quantization")
        else:
            logger.info("Using BFloat16 precision")

        self.engine_args = engine_args
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        logger.info(f"Model loaded: {self.model_id}")
        logger.info(f"Using GPU memory utilization: {mem_utilization}")

    @staticmethod
    def _get_model_dir() -> str:
        """Get the model directory from environment or default"""
        return os.getenv("MODEL_DIR", "/models")

    def validate_messages(self, messages: List[Dict[str, str]]):
        """Validate message format"""
        if not messages:
            raise ValueError("Messages cannot be empty")

        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Invalid message format: {msg}")
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError(f"Invalid role in message: {msg['role']}")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        generation_config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Internal generation method"""
        start_time = time.time()
        config = generation_config or GenerationConfig()
        tokenizer = await self.engine.get_tokenizer()

        try:
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            raise

        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
        )

        request_id = f"chatcmpl-{uuid.uuid4()}"

        results_generator = self.engine.generate(
            prompt=full_prompt, sampling_params=sampling_params, request_id=request_id)

        input_tokens = len(tokenizer.encode(full_prompt))
        if input_tokens > self.engine_args.max_model_len:
            raise ValueError(
                f"Input sequence length ({input_tokens}) exceeds maximum allowed ({self.engine_args.max_model_len})")

        total_tokens = 0
        current_response = ""
        first_token_time = None

        try:
            async for output in results_generator:
                if output.outputs:
                    if first_token_time is None:
                        first_token_time = time.time()

                    generated_text = output.outputs[0].text
                    delta = generated_text[len(current_response):]
                    current_response = generated_text
                    total_tokens += len(tokenizer.encode(delta))

                    yield LLMResponse(
                        choices=[
                            LLMChoice(
                                delta=LLMMessage(
                                    role="assistant",
                                    content=delta
                                ),
                                index=0
                            )
                        ],
                        usage=LLMTokenUsage(
                            prompt_tokens=input_tokens,
                            completion_tokens=total_tokens,
                            total_tokens=input_tokens + total_tokens
                        ),
                        id=request_id,
                        model=self.model_id,
                        created=int(time.time())
                    )

                    await asyncio.sleep(0)

            # Final message
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Generation completed in {duration:.2f}s")
            logger.info(
                f"  Time to first token: {(first_token_time - start_time):.2f} seconds")
            logger.info(f"  Total tokens: {total_tokens}")
            logger.info(f"  Prompt tokens: {input_tokens}")
            logger.info(f"  Generated tokens: {total_tokens}")
            generation_time = end_time - first_token_time if first_token_time else 0
            logger.info(f"  Tokens per second: {total_tokens / generation_time:.2f}")

            yield LLMResponse(
                choices=[
                    LLMChoice(
                        delta=LLMMessage(
                            role="assistant",
                            content=""
                        ),
                        index=0,
                        finish_reason="stop"
                    )
                ],
                usage=LLMTokenUsage(
                    prompt_tokens=input_tokens,
                    completion_tokens=total_tokens,
                    total_tokens=input_tokens + total_tokens
                ),
                id=request_id,
                model=self.model_id,
                created=int(time.time())
            )

        except Exception as e:
            if "CUDA out of memory" in str(e):
                logger.error(
                    "GPU memory exhausted, consider reducing batch size or model parameters")
            elif "tokenizer" in str(e).lower():
                logger.error("Tokenizer error, check input text format")
            else:
                logger.error(f"Error generating response: {e}")
            raise

    async def __call__(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Generate responses for messages.

        Args:
            messages: List of message dictionaries in OpenAI format
            **kwargs: Generation parameters
        """
        logger.debug(f"Generating response for messages: {messages}")
        start_time = time.time()

        try:
            # Validate inputs
            self.validate_messages(messages)
            config = GenerationConfig(**kwargs)
            config.validate()

            async for response in self.generate(messages, config):
                yield response

        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            raise

    def __str__(self):
        return f"LLMPipeline(model_id={self.model_id})"

    def _find_model_path(self, base_path):
        # Check if the model files are directly in the base path
        if any(file.endswith('.bin') or file.endswith('.safetensors') for file in os.listdir(base_path)):
            return base_path

        # If not, look in subdirectories
        for root, dirs, files in os.walk(base_path):
            if any(file.endswith('.bin') or file.endswith('.safetensors') for file in files):
                return root
