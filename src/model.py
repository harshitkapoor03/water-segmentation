import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(config):
    """
    Builds the UNet model using segmentation-models-pytorch.

 
    """
    model_map = {
        "unet":       smp.Unet,
        "unet++":     smp.UnetPlusPlus,
        "deeplabv3+": smp.DeepLabV3Plus,
    }

    name = config["training"]["model_name"].lower()

    if name not in model_map:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(model_map.keys())}")

    model = model_map[name](
        encoder_name=config["training"]["encoder"],
        encoder_weights=config["training"]["encoder_weights"],
        in_channels=3,
        classes=1,
        activation=None
    )

    return model


class CombinedLoss(nn.Module):
    """
    Weighted sum of BCE loss and Dice loss.

    """

    def __init__(self, bce_weight=0.5):
        # super().__init__() calls the parent class (nn.Module) constructor
        # This is required for PyTorch modules — it sets up internal state
        super().__init__()
        self.bce_loss  = nn.BCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(mode="binary")
        self.bce_weight = bce_weight

    def forward(self, predictions, targets):
        # forward() is called automatically when you do loss(pred, target)
        bce  = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice
