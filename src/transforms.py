import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=256):
    """
    Augmentations applied randomly to training patches.
    Each transform has a probability p :it won't always fire.

    Why each one:
    HorizontalFlip/VerticalFlip — satellites view from above so
        there's no natural up/down orientation. A river looks
        the same flipped. Free extra training data.

    RandomRotate90 — same reason. North is arbitrary from space.

    ShiftScaleRotate — simulates slightly different viewing angles
        and altitudes across different Sentinel-2 captures.

    ElasticTransform/GridDistortion — simulates subtle terrain
        distortions and lens effects in satellite imagery.

    ColorJitter — Sentinel-2 images vary in brightness and contrast
        due to atmospheric conditions, cloud cover, season.
        The model should be robust to these variations.

    Normalize — subtracts ImageNet mean and divides by ImageNet std.
        We do this because our ResNet34 encoder was pretrained on
        ImageNet. It learned to process inputs in that specific
        numerical range. Different normalization would break its
        pretrained feature detectors.

    ToTensorV2 — converts numpy (H, W, C) to PyTorch tensor (C, H, W).
        PyTorch's convention is channels first. This transform does
        the axis swap automatically.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
       translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
       scale=(0.8, 1.2),
       rotate=(-30, 30),
       p=0.5
        ),

        A.OneOf([
            A.ElasticTransform(p=1.0),
            A.GridDistortion(p=1.0),
        ], p=0.3),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.4
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size=256):
    """
    No augmentation for validation or test.
    Only resize and normalize.

    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])
