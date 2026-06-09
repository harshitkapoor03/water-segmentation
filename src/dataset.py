import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from src.transforms import get_train_transforms, get_val_transforms
import glob

logger = logging.getLogger(__name__)





class WaterBodyDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, cache=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.cache = cache
        self.cached_data = []

        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
            )

        if cache:
            logger.info("Caching dataset into RAM...")
            for ip, mp in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
                image = np.array(Image.open(ip).convert("RGB"))
                mask = np.array(Image.open(mp).convert("L"))
                mask = (mask > 127).astype(np.float32)
                self.cached_data.append((image, mask))
            logger.info("Cache complete.")

        logger.info(f"Dataset ready — {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cache:
            image, mask = self.cached_data[idx]
        else:
            image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
            mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
            mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)

        return image, mask


def build_dataloaders_from_raw(config):

    img_dir = config["data"]["raw_image_dir"]
    mask_dir = config["data"]["raw_mask_dir"]

    image_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.png"))
        + glob.glob(os.path.join(img_dir, "*.jpg"))
    )
    mask_paths = sorted(
        glob.glob(os.path.join(mask_dir, "*.png"))
        + glob.glob(os.path.join(mask_dir, "*.jpg"))
    )

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    total = len(image_paths)
    train_end = int(total * config["data"]["train_split"])
    val_end = train_end + int(total * config["data"]["val_split"])

    rng = np.random.RandomState(42)
    indices = rng.permutation(total)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def pick(paths, idx):
        return [paths[i] for i in idx]

    image_size = config["data"]["patch_size"]
    batch_size = config["training"]["batch_size"]
    workers = config["training"]["num_workers"]

    train_ds = WaterBodyDataset(
        pick(image_paths, train_idx),
        pick(mask_paths, train_idx),
        transform=get_train_transforms(image_size),
        cache=True,
    )
    val_ds = WaterBodyDataset(
        pick(image_paths, val_idx),
        pick(mask_paths, val_idx),
        transform=get_val_transforms(image_size),
        cache=True,
    )
    test_ds = WaterBodyDataset(
        pick(image_paths, test_idx),
        pick(mask_paths, test_idx),
        transform=get_val_transforms(image_size),
        cache=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    logger.info(
        f"Raw — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader

