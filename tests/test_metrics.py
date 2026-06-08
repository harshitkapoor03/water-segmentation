import torch

from src.metrics import iou_score, dice_score, pixel_accuracy, precision_recall


def test_iou_perfect_prediction():
    # Large positive logits → sigmoid → ~1.0 → above threshold
    pred   = torch.ones(2, 1, 4, 4) * 10.0
    target = torch.ones(2, 1, 4, 4)
    assert iou_score(pred, target) > 0.99


def test_iou_zero_overlap():
    # Large negative logits → sigmoid → ~0.0 → below threshold
    pred   = torch.ones(2, 1, 4, 4) * -10.0
    target = torch.ones(2, 1, 4, 4)
    assert iou_score(pred, target) < 0.01


def test_dice_perfect():
    pred   = torch.ones(2, 1, 4, 4) * 10.0
    target = torch.ones(2, 1, 4, 4)
    assert dice_score(pred, target) > 0.99


def test_pixel_accuracy_perfect():
    pred   = torch.ones(2, 1, 4, 4) * 10.0
    target = torch.ones(2, 1, 4, 4)
    assert pixel_accuracy(pred, target) > 0.99


def test_precision_recall_perfect():
    pred   = torch.ones(2, 1, 4, 4) * 10.0
    target = torch.ones(2, 1, 4, 4)
    p, r   = precision_recall(pred, target)
    assert p > 0.99
    assert r > 0.99


def test_dice_always_gte_iou():
    # Dice >= IoU always — mathematical property
    pred   = torch.randn(4, 1, 8, 8)
    target = (torch.randn(4, 1, 8, 8) > 0).float()
    assert dice_score(pred, target) >= iou_score(pred, target) - 1e-5
