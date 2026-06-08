# Water Body Segmentation

This project trains a deep learning model to detect and segment water bodies in satellite imagery. Given a satellite image as input, the model outputs a binary mask where white pixels represent water and black pixels represent land.

The model is a UNet architecture with a ResNet34 encoder pretrained on ImageNet. It was trained on 2841 satellite images using a combination of Binary Cross Entropy and Dice loss. The best run achieved a Test IoU of 0.7884 and Test Accuracy of 0.936.

The project includes a Flask web application served via Docker so anyone can run inference without setting up Python or installing dependencies manually.

---

## Results

| Metric | Score |
|---|---|
| Test IoU | 0.7884 |
| Test Dice | 0.8422 |
| Test Accuracy | 0.9360 |
| Test Precision | 0.9258 |
| Test Recall | 0.8831 |

---

## Project Structure

```
water-segmentation/
├── app.py                  Flask web application
├── predictor.py            Loads model and runs inference
├── logger.py               Inference timing decorator
├── train.py                Training entry point
├── configs/
│   └── config.yaml         All hyperparameters and paths
├── src/
│   ├── model.py            UNet model and combined loss
│   ├── dataset.py          Dataset class and dataloaders
│   ├── trainer.py          Training loop with MLflow logging
│   ├── transforms.py       Augmentation pipelines
│   ├── metrics.py          IoU, Dice, accuracy, precision, recall
│   └── tiling.py           Patch tiling utilities(commented out, future work)
├── tests/
│   ├── test_dataset.py     Dataset unit tests
│   └── test_metrics.py     Metrics unit tests
├── checkpoints/            Saved model weights
├── logs/                   Training logs
├── Dockerfile
└── requirements.txt
```

---

## Running the App with Docker

This is the recommended way to run the project. You do not need Python installed.

### Prerequisites

**Step 1: Install Git**

Go to https://git-scm.com/download/win, download and run the installer. Click Next through everything, default settings are fine.

Verify it worked by opening PowerShell and running:
```
git --version
```

**Step 2: Install Git LFS**

The model weights file is 280MB so it is stored using Git Large File Storage. Without this the model will not download correctly.

Go to https://git-lfs.com, download and run the installer. Then open PowerShell and run:
```
git lfs install
```
You should see: `Git LFS initialized`

**Step 3: Update WSL**

Open PowerShell and run:
```
wsl --version
```
If it says that you need to update, run:
```
wsl --update
```
If there is no wsl installed on your device, come back to step 3 after step 4

**Step 4: Install Docker Desktop**

Go to https://www.docker.com/products/docker-desktop and download Docker Desktop for Windows. Run the installer and restart your computer when asked.

After restart, open Docker Desktop from the Start menu and wait until the bottom left says Engine running with a green dot.

Then go to Settings (gear icon top right) and under Resources, click Network. Make sure Enable host networking is turned on. Click Apply and Restart.

### Running the App

Open PowerShell, go to whichever directory you want to clone the repo into and run these commands one by one:

```
git clone https://github.com/harshitkapoor03/water-segmentation
cd water-segmentation
docker build -t water-segmentation .
docker run -p 5000:5000 water-segmentation
```

The build step takes 10 to 15 minutes the first time because it downloads Python, PyTorch and all dependencies. Do not close the window.

When you see `Model ready.` in the terminal, open your browser and go to:
```
http://localhost:5000
```

Upload any satellite image, click Run Segmentation, and the water mask will appear on the right side of the screen.

To stop the app press Ctrl+C in the PowerShell window.

---

## Training

To retrain the model you need Python 3.14.5 with the packages from `requirements.txt` installed. Use `requirements_gpu.txt` for the GPU version of PyTorch.

### Data Setup

Place your satellite images in `data/Images/` and corresponding masks in `data/Masks/`. Images and masks must have matching filenames. The dataset is split automatically into 80% train, 10% validation, 10% test using a fixed random seed.

### Config File

All settings are controlled through `configs/config.yaml`. Here is what each parameter does:

**Data settings**
```yaml
data:
  raw_image_dir: "data/Images"       # folder containing input satellite images
  raw_mask_dir: "data/Masks"         # folder containing binary mask images
  patch_size: 256                    # images are resized to this during training
  train_split: 0.8                   # 80% of data used for training
  val_split: 0.1                     # 10% used for validation
  test_split: 0.1                    # 10% held out for final evaluation
```

**Training settings**
```yaml
training:
  model_name: "unet"                 # architecture: unet, unet++
  encoder: "resnet34"                # backbone: resnet18, resnet34, resnet50
  encoder_weights: "imagenet"        # start from ImageNet pretrained weights
  batch_size: 32                     # number of images per gradient update
  num_epochs: 50                     # maximum training epochs
  learning_rate: 0.0001              # initial learning rate for AdamW optimizer
  weight_decay: 0.0001               # L2 regularisation strength
  bce_weight: 0.3                    # how much BCE loss vs Dice loss (0.3 = 30% BCE, 70% Dice)
  early_stopping_patience: 5        # stop if val IoU does not improve for this many epochs
  checkpoint_path: "checkpoints/best_model.pth"   # where to save the best model
  device: "cuda"                     # cuda or cpu
  num_workers: 0                     # dataloader worker processes
  freeze_encoder: false              # whether to freeze the encoder during training
  freeze_layers: "all"               # if freezing: all (entire encoder) or partial (layer1+layer2 only)
```

**Inference settings**
```yaml
inference:
  model_path: "checkpoints/resnet34_lr0.0001_bce0.3_resize_epochs50/best_model.pth"
  threshold: 0.5                     # probability threshold for classifying a pixel as water
  use_tiling: false                  # whether to use patch tiling for large images(future work)
```

### Running Training

Basic training run using config defaults:
```
python train.py
```

All config values can be overridden from the command line without editing the yaml file:
```
python train.py --lr 0.0001 --encoder resnet34 --bce_weight 0.3 --epochs 50
```

Training with full encoder freeze (encoder weights stay fixed, only decoder trains):
```
python train.py --lr 0.0001 --encoder resnet34 --bce_weight 0.3 --epochs 50 --freeze_encoder --freeze_layers all
```

Training with partial encoder freeze (only early layers frozen, deeper layers adapt to satellite data):
```
python train.py --lr 0.0001 --encoder resnet34 --bce_weight 0.3 --epochs 50 --freeze_encoder --freeze_layers partial
```

**Available command line arguments**

| Argument | Type | Description |
|---|---|---|
| `--lr` | float | Learning rate |
| `--encoder` | string | Encoder backbone (resnet18, resnet34, resnet50) |
| `--bce_weight` | float | Weight for BCE component of the combined loss |
| `--batch_size` | int | Batch size |
| `--epochs` | int | Number of training epochs |
| `--freeze_encoder` | flag | Freeze encoder weights during training |
| `--freeze_layers` | string | Which layers to freeze: all or partial |

The run name and checkpoint folder are generated automatically from the hyperparameters so each run saves to its own folder and nothing gets overwritten.

### Viewing Training Results with MLflow

MLflow tracks all metrics and hyperparameters automatically during training. To view the dashboard run:
```
mlflow ui 
```
or if that doesnt work :
```
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Then open the link that it gives you in the terminal in your browser. You can compare runs side by side, plot val IoU curves across epochs, and filter by any hyperparameter.

---

## Running Tests

```
pytest tests/
```

The test suite covers dataset loading, output shapes, mask binarisation, and all metric functions.

---

## Model Architecture

The model is a UNet with a ResNet34 encoder. UNet uses an encoder decoder structure with skip connections. The encoder progressively downsamples the image extracting features at multiple scales. The decoder upsamples back to the original resolution using those features. Skip connections pass encoder features directly to the corresponding decoder layer which helps recover fine spatial detail lost during downsampling.

ResNet34 was pretrained on ImageNet which gives the encoder a strong starting point for recognising visual features, even though satellite imagery looks different from natural photos. The final layer outputs one channel per pixel which is passed through a sigmoid to get a water probability between 0 and 1.

Loss is a weighted combination of Binary Cross Entropy and Dice loss. BCE penalises each pixel independently. Dice loss directly optimises the overlap between predicted and actual water regions which matters more for segmentation quality than pixel accuracy alone.
