import logging
from typing import List, Optional, Tuple
from PIL import Image
import torch
from sam2.build_sam import build_sam2_camera_predictor
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
import os

MODEL_MAPPING = {
    "facebook/sam2-hiera-tiny": {
        "config": "sam2_hiera_t.yaml",
        "checkpoint": "sam2_hiera_tiny.pt"
    },
    "facebook/sam2-hiera-small": {
        "config": "sam2_hiera_s.yaml",
        "checkpoint": "sam2_hiera_small.pt"
    },
    "facebook/sam2-hiera-large": {
        "config": "sam2_hiera_l.yaml",
        "checkpoint": "sam2_hiera_large.pt"
    }
}

class Sam2Wrapper:
    def __init__(
        self,
        model_id_or_path: str,
        device: str,
        **kwargs
    ):
        self.device = device
        self.model_id = model_id_or_path
        if model_id_or_path in MODEL_MAPPING:
            model_info = MODEL_MAPPING[model_id_or_path]
            config_path = os.path.join("/models/sam2--checkpoints/", model_id_or_path.replace("/", "--"))
            model_cfg = model_info['config']
            sam2_checkpoint = f"{config_path}/{model_info['checkpoint']}"
        else:
            raise ValueError(f"Model ID {model_id_or_path} not supported")
        
        logging.info(f"Initializing segment-anything-2 with model_id {self.model_id}")
        
        torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Initialize Hydra to load the configuration
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        
        # Code from sam2.build_sam.build_sam2_camera_predictor to appease Hydra
        with initialize_config_dir(config_dir=config_path, version_base=None):
            cfg = compose(config_name=model_cfg)

            hydra_overrides = [
                "++model._target_=sam2.sam2_camera_predictor.SAM2CameraPredictor",
            ]
            hydra_overrides_extra = [
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                "++model.binarize_mask_from_pts_for_mem_enc=true",
                "++model.fill_hole_area=8",
            ]
            hydra_overrides.extend(hydra_overrides_extra)

            cfg = compose(config_name=model_cfg, overrides=hydra_overrides)
            OmegaConf.resolve(cfg)

            #Load the model
            model = instantiate(cfg.model, _recursive_=True)
            load_checkpoint(model, sam2_checkpoint, self.device)
            model.to(self.device)
            model.eval()

            # Set the model in memory
            self.predictor = model

def load_checkpoint(model, ckpt_path, device):
    if ckpt_path is not None:

        sd = torch.load(ckpt_path, map_location=device)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(f"Missing keys: {missing_keys}")
            raise RuntimeError("Missing keys while loading checkpoint.")
        if unexpected_keys:
            logging.error(f"Unexpected keys: {unexpected_keys}")
            raise RuntimeError("Unexpected keys while loading checkpoint.")
        logging.info("Loaded checkpoint successfully.")
