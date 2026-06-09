import torch


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Intersection over Union.

    IoU = (predicted water AND actual water) / (predicted water OR actual water)

    """
    # Apply sigmoid to convert raw logits to probabilities 0-1
    # Then threshold to get binary prediction
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    # sum over height and width dimensions (dim 2 and 3)
    # leaving shape (batch_size, 1)
    intersection = (pred_binary * target).sum(dim=(2, 3))
    union        = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)

    # .mean() averages across the batch, .item() converts 1-element tensor to Python float
    return iou.mean().item()


def dice_score(pred, target, smooth=1e-6):
    """
    Dice coefficient.
    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Always >= IoU. Relationship: Dice = 2*IoU / (1 + IoU)
    So IoU=0.80 → Dice≈0.89.

    We track both because different papers use different metrics.
    Having both makes your results comparable to existing literature.
    """


    pred_soft    = torch.sigmoid(pred)
    intersection = (pred_soft * target).sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (
        pred_soft.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
    )
    return dice.mean().item()


def pixel_accuracy(pred, target, threshold=0.5):
    """
    Fraction of pixels correctly classified.
    Least meaningful metric for this task due to class imbalance.
    Included for completeness and comparability.
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    correct     = (pred_binary == target).float().sum()
    total       = torch.tensor(target.numel(), dtype=torch.float32)
    return (correct / total).item()


def precision_recall(pred, target, threshold=0.5, smooth=1e-6):
    """
    Precision: of pixels called water, what fraction were actually water?
    Recall:    of actual water pixels, what fraction did we find?

    These have a direct tradeoff controlled by threshold:
    - Lower threshold (0.3): higher recall, lower precision
      Use case: flood detection — missing actual flooding is dangerous
    - Higher threshold (0.7): higher precision, lower recall
      Use case: mapping — false positives pollute the map

    Default 0.5 is a neutral starting point.
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()

    tp = (pred_binary * target).sum().item()
    fp = (pred_binary * (1 - target)).sum().item()
    fn = ((1 - pred_binary) * target).sum().item()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)

    return precision, recall


def compute_all_metrics(pred, target, threshold=0.5):
    """
    Convenience function — computes all metrics at once.
    Returns a dictionary so trainer.py can log them cleanly.
    """
    iou  = iou_score(pred, target, threshold)
    dice = dice_score(pred, target)
    acc  = pixel_accuracy(pred, target, threshold)
    prec, rec = precision_recall(pred, target, threshold)

    return {
        "iou":       iou,
        "dice":      dice,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec
    }
