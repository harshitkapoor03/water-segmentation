#import os
import numpy as np
import pytest
from PIL import Image
from src.dataset import WaterBodyDataset
from src.transforms import get_val_transforms


def make_dummy_data(tmp_path, n=4):
    """Creates fake images and masks for testing without needing real data."""
    img_dir  = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    img_paths  = []
    mask_paths = []

    for i in range(n):
        # Random RGB image
        img_arr  = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        # Binary mask — only 0 and 255 values
        mask_arr = np.random.choice([0, 255], (300, 300)).astype(np.uint8)

        ip = str(img_dir  / f"img_{i}.png")
        mp = str(mask_dir / f"img_{i}.png")

        Image.fromarray(img_arr).save(ip)
        Image.fromarray(mask_arr).save(mp)

        img_paths.append(ip)
        mask_paths.append(mp)

    return img_paths, mask_paths


def test_dataset_length(tmp_path):
    imgs, masks = make_dummy_data(tmp_path, n=4)
    ds = WaterBodyDataset(imgs, masks, transform=get_val_transforms(256))
    assert len(ds) == 4


def test_output_shapes(tmp_path):
    imgs, masks = make_dummy_data(tmp_path, n=2)
    ds          = WaterBodyDataset(imgs, masks, transform=get_val_transforms(256))
    img, mask   = ds[0]
    assert img.shape  == (3, 256, 256)
    assert mask.shape == (1, 256, 256)


def test_mask_is_binary(tmp_path):
    import torch
    imgs, masks = make_dummy_data(tmp_path, n=2)
    ds          = WaterBodyDataset(imgs, masks, transform=get_val_transforms(256))
    _, mask     = ds[0]
    unique_vals = torch.unique(mask).tolist()
    for v in unique_vals:
        assert v in [0.0, 1.0], f"Non-binary mask value: {v}"


def test_mismatched_lengths_raises(tmp_path):
    imgs, masks = make_dummy_data(tmp_path, n=4)
    with pytest.raises(ValueError):
        WaterBodyDataset(imgs[:3], masks, transform=None)
