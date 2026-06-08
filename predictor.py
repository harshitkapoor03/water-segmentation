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
        # self.stride = config["data"]["stride"]

        # Load model once at startup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(config["inference"]["model_path"])

        # Same normalization as training — must match exactly
        # If normalization differs between training and inference,
        # the model receives inputs outside the distribution it learned on
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

        # Build model architecture using the config saved inside the checkpoint
        # This is why we save config in the checkpoint —
        # so we don't need a separate config file at inference time
        model = build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

        # model.eval() is critical:
        # 1. Disables dropout layers (they randomly zero neurons during training
        #    for regularisation — at inference you want deterministic results)
        # 2. BatchNorm uses running statistics from training instead of
        #    batch statistics — important when batch size is 1
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

#from src.tiling import tile_for_inference, stitch_predictions
    # @log_inference_time
    # def predict(self, image_path):
    #     image = np.array(Image.open(image_path).convert("RGB"))
    #     original_h, original_w = image.shape[:2]

    #     use_tiling = self.config["inference"].get("use_tiling", False)

    #     if use_tiling:
    #         # Phase 2 — patch-trained model
    #         patches, coords = tile_for_inference(image, self.patch_size, self.stride)
    #         patch_probs = []
    #         with torch.no_grad():
    #             for patch in patches:
    #                 transformed = self.transform(image=patch)
    #                 tensor = transformed["image"].unsqueeze(0).to(self.device)
    #                 logits = self.model(tensor)
    #                 prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    #                 patch_probs.append(prob)

    #         avg_prob = stitch_predictions(
    #             patch_probs, coords, original_h, original_w, self.patch_size
    #         )

    #     else:
    #         # Phase 1 — resize-trained model
    #         # Resize whole image, predict, resize mask back
    #         resized = np.array(
    #             Image.fromarray(image).resize(
    #                 (self.patch_size, self.patch_size), Image.BILINEAR
    #             )
    #         )
    #         with torch.no_grad():
    #             transformed = self.transform(image=resized)
    #             tensor = transformed["image"].unsqueeze(0).to(self.device)
    #             logits = self.model(tensor)
    #             avg_prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    #     binary_mask = (avg_prob > self.threshold).astype(np.uint8) * 255

    #     mask_resized = np.array(
    #         Image.fromarray(binary_mask).resize((original_w, original_h), Image.NEAREST)
    #     )

    #     return mask_resized

    # def predict(self, image_path):
    #     """
    #     Full inference pipeline on one image.

    #     Steps:
    #     1. Load image
    #     2. Record original dimensions (we resize back at the end)
    #     3. Tile into patches (same strategy as training)
    #     4. Normalize and convert each patch to tensor
    #     5. Run each patch through model
    #     6. Apply sigmoid to get probabilities
    #     7. Stitch probability maps together (average in overlap zones)
    #     8. Threshold once to get binary mask
    #     9. Resize back to original dimensions
    #     10. Return as numpy array (0 = land, 255 = water)
    #     """
    #     # Load as RGB numpy array — shape (H, W, 3)
    #     image = np.array(Image.open(image_path).convert("RGB"))
    #     original_h, original_w = image.shape[:2]

    #     # Tile the image — returns list of patches and their top-left coords
    #     patches, coords = tile_for_inference(
    #         image, self.patch_size, self.stride
    #     )

    #     patch_probs = []

    #     # torch.no_grad() — don't build computation graph, not training
    #     with torch.no_grad():
    #         for patch in patches:
    #             # Apply normalization transform
    #             # Note: no Resize here — patches are already patch_size x patch_size
    #             transformed = self.transform(image=patch)

    #             # unsqueeze(0) adds batch dimension:
    #             # (3, 256, 256) → (1, 3, 256, 256)
    #             # Model expects a batch even when processing one patch
    #             tensor = transformed["image"].unsqueeze(0).to(self.device)

    #             # Forward pass through the model
    #             logits = self.model(tensor)

    #             # sigmoid converts raw logits to probabilities 0-1
    #             # squeeze() removes batch and channel dimensions:
    #             # (1, 1, 256, 256) → (256, 256)
    #             prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    #             patch_probs.append(prob)

    #     # Stitch probability maps together, averaging overlapping regions
    #     avg_prob = stitch_predictions(
    #         patch_probs, coords, original_h, original_w, self.patch_size
    #     )

    #     # Threshold the averaged probability map once
    #     # (prob > threshold) gives True/False, multiply by 255 for display
    #     binary_mask = (avg_prob > self.threshold).astype(np.uint8) * 255

    #     # Resize mask back to original dimensions
    #     # NEAREST interpolation — avoids creating intermediate values
    #     # between 0 and 255 that would corrupt the binary mask
    #     mask_resized = np.array(
    #         Image.fromarray(binary_mask).resize(
    #             (original_w, original_h), Image.NEAREST
    #         )
    #     )

    #     return mask_resized
