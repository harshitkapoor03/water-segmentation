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


# class WaterBodyDataset(Dataset):
#     """
#     PyTorch Dataset for water body segmentation patches.

#     PyTorch's Dataset contract requires exactly two methods:
#     __len__  : return the total number of samples
#     __getitem__ : return one (image, mask) pair given an index

#     DataLoader calls these automatically to build batches.
#     We store paths not arrays — loading 2841+ images into RAM
#     at once would crash most machines. Loading on demand
#     (lazy loading) keeps memory usage constant regardless
#     of dataset size.
#     """

#     def __init__(self, image_paths, mask_paths, transform=None):
#         self.image_paths = image_paths
#         self.mask_paths = mask_paths
#         self.transform = transform

#         if len(image_paths) != len(mask_paths):
#             raise ValueError(
#                 f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
#             )

#         logger.info(f"Dataset ready — {len(self.image_paths)} samples")

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         # Load image as RGB — shape (H, W, 3), values 0-255
#         image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))

#         # Load mask as grayscale — shape (H, W), values 0 or 255
#         mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))

#         # Convert mask to binary float — 0.0 for land, 1.0 for water
#         # We threshold at 127 not 0 because JPEG compression can create
#         # intermediate values. Anything above 127 we call water.
#         mask = (mask > 127).astype(np.float32)

#         if self.transform:
#             # albumentations expects keyword arguments image= and mask=
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented["image"]
#             # mask comes out as (H, W) — we add a channel dimension
#             # to make it (1, H, W) to match model output shape
#             mask = augmented["mask"].unsqueeze(0)

#         return image, mask

# def build_dataloaders(config):
#     """
#     Loads patch file paths, shuffles and splits them,
#     then builds train/val/test DataLoaders.

#     We shuffle with a fixed random seed (42) so the split is
#     random but identical every time you run the code.
#     This matters for reproducibility — you always evaluate
#     on the exact same test set.
#     """
#     from src.transforms import get_train_transforms, get_val_transforms

#     img_dir = config["data"]["patches_image_dir"]
#     mask_dir = config["data"]["patches_mask_dir"]

#     # glob finds all files matching a pattern
#     # sorted() ensures consistent ordering across operating systems
#     image_paths = sorted(
#         glob.glob(os.path.join(img_dir, "*.png"))
#         + glob.glob(os.path.join(img_dir, "*.jpg"))
#     )
#     mask_paths = sorted(
#         glob.glob(os.path.join(mask_dir, "*.png"))
#         + glob.glob(os.path.join(mask_dir, "*.jpg"))
#     )

#     if len(image_paths) == 0:
#         raise RuntimeError(f"No patches found in {img_dir}. Run build_patches() first.")

#     total = len(image_paths)
#     train_end = int(total * config["data"]["train_split"])
#     val_end = train_end + int(total * config["data"]["val_split"])

#     # np.random.RandomState(42) creates a seeded random number generator
#     # .permutation returns a shuffled list of indices
#     # Using a fixed seed means the same indices get shuffled the same way
#     rng = np.random.RandomState(42)
#     indices = rng.permutation(total)

#     train_idx = indices[:train_end]
#     val_idx = indices[train_end:val_end]
#     test_idx = indices[val_end:]

#     def pick(paths, idx):
#         # list comprehension — builds a new list by selecting items at each index
#         return [paths[i] for i in idx]

#     image_size = config["data"]["patch_size"]
#     batch_size = config["training"]["batch_size"]
#     workers = config["training"]["num_workers"]

#     train_ds = WaterBodyDataset(
#         pick(image_paths, train_idx),
#         pick(mask_paths, train_idx),
#         transform=get_train_transforms(image_size),
#     )
#     val_ds = WaterBodyDataset(
#         pick(image_paths, val_idx),
#         pick(mask_paths, val_idx),
#         transform=get_val_transforms(image_size),
#     )
#     test_ds = WaterBodyDataset(
#         pick(image_paths, test_idx),
#         pick(mask_paths, test_idx),
#         transform=get_val_transforms(image_size),
#     )

#     # pin_memory=True speeds up data transfer from CPU RAM to GPU VRAM
#     # Only useful if training on GPU — harmless if on CPU
#     # shuffle=True for train so model sees batches in different order each epoch
#     # shuffle=False for val/test so results are consistent and comparable
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=workers,
#         pin_memory=True,
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=workers,
#         pin_memory=True,
#     )
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=workers,
#         pin_memory=True,
#     )

#     logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
#     return train_loader, val_loader, test_loader
