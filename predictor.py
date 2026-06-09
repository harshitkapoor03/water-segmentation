import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from logger import log_inference_time


logger = logging.getLogger(__name__)


class WaterSegmentationPredictor:
    """
    Loads a trained model and runs inference on new images.

    If prediction were a plain function, it would reload the model
    on every call. As a class, we load once in __init__ and reuse
    the model for every subsequent prediction call.
    
    """

    def __init__(self, config):
        self.config = config
        self.threshold = config["inference"]["threshold"]
        self.patch_size = config["data"]["patch_size"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(config["inference"]["model_path"])

        # Same normalization as training 
        self.transform = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def _load_model(self, model_path):
        from src.model import build_model

        # Load the checkpoint dictionary saved during training
        checkpoint = torch.load(model_path, map_location=self.device)


        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

  
        model.eval()

        logger.info(
            f"Loaded model — epoch {checkpoint['epoch']}, "
            f"IoU {checkpoint['metrics']['iou']:.4f}"
        )
        return model
    
    @log_inference_time
    def predict(self, image_path):
        image = np.array(Image.open(image_path).convert("RGB"))
        original_h, original_w = image.shape[:2]

        resized = np.array(
                Image.fromarray(image).resize(
                    (self.patch_size, self.patch_size), Image.BILINEAR
                )
            )
        with torch.no_grad():
                transformed = self.transform(image=resized)
                tensor = transformed["image"].unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                avg_prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary_mask = (avg_prob > self.threshold).astype(np.uint8) * 255

        mask_resized = np.array(
            Image.fromarray(binary_mask).resize((original_w, original_h), Image.NEAREST)
        )

        return mask_resized


