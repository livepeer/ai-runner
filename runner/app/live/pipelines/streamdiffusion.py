import logging
from typing import Dict, List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field
from StreamDiffusionWrapper import StreamDiffusionWrapper

from .interface import Pipeline


class StreamDiffusionParams(BaseModel):
    class Config:
        extra = "forbid"

    prompt: str = "talking head, cyberpunk, tron, matrix, ultra-realistic, dark, futuristic, neon, 8k"
    model_id: str = "KBlueLeaf/kohaku-v2.1"
    lora_dict: Optional[Dict[str, float]] = None
    use_lcm_lora: bool = True
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"
    num_inference_steps: int = 50
    t_index_list: Optional[List[int]] = [37, 45, 48]
    scale: float = 1.0
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt"
    use_denoising_batch: bool = True
    enable_similar_image_filter: bool = False
    seed: int = 2
    guidance_scale: float = 1.2
    do_add_noise: bool = False
    similar_image_filter_threshold: float = 0.98


class StreamDiffusion(Pipeline):
    def __init__(self, **params):
        super().__init__(**params)
        self.pipe: Optional[StreamDiffusionWrapper] = None
        self.first_frame = True
        self.update_params(**params)

    def process_frame(self, image: Image.Image) -> Image.Image:
        img_tensor = self.pipe.preprocess_image(image)
        img_tensor = self.pipe.stream.image_processor.denormalize(img_tensor)

        if self.first_frame:
            self.first_frame = False
            for _ in range(self.pipe.batch_size):
                self.pipe(image=img_tensor)

        return self.pipe(image=img_tensor)

    def update_params(self, **params):
        new_params = StreamDiffusionParams(**params)
        if self.pipe is not None:
            # avoid resetting the pipe if only the prompt changed
            only_prompt = self.params.model_copy(update={"prompt": new_params.prompt})
            if new_params == only_prompt:
                logging.info(f"Updating prompt: {new_params.prompt}")
                self.pipe.stream.update_prompt(new_params.prompt)
                self.params = new_params
                return

        logging.info(f"Resetting diffuser for params change")
        pipe = StreamDiffusionWrapper(
            model_id_or_path=new_params.model_id,
            lora_dict=new_params.lora_dict,
            use_lcm_lora=new_params.use_lcm_lora,
            lcm_lora_id=new_params.lcm_lora_id,
            t_index_list=new_params.t_index_list,
            frame_buffer_size=1,
            width=512,
            height=512,
            warmup=10,
            acceleration=new_params.acceleration,
            do_add_noise=new_params.do_add_noise,
            mode="img2img",
            # output_type="pt",
            enable_similar_image_filter=new_params.enable_similar_image_filter,
            similar_image_filter_threshold=new_params.similar_image_filter_threshold,
            use_denoising_batch=new_params.use_denoising_batch,
            seed=new_params.seed,
        )
        pipe.prepare(
            prompt=new_params.prompt,
            num_inference_steps=new_params.num_inference_steps,
            guidance_scale=new_params.guidance_scale,
        )

        self.params = new_params
        self.pipe = pipe
        self.first_frame = True
