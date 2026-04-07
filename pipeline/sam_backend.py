import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
import config as cfg

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_SAM = sam_model_registry[cfg.SAM_MODEL_TYPE](checkpoint=str(cfg.SAM_WEIGHT))
_SAM.to(_DEVICE)
_PREDICTOR = SamPredictor(_SAM)

def predict_sam_mask(image_bgr: np.ndarray, point_coords: np.ndarray, point_labels: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    _PREDICTOR.set_image(image_rgb)
    masks, scores, _ = _PREDICTOR.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    mask = (masks[0].astype(np.uint8) * 255)
    return mask