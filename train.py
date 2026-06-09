import argparse
import logging
import os

import mlflow
import yaml
import torch


from src.dataset import build_dataloaders_from_raw
from src.model import build_model, CombinedLoss
from src.trainer import Trainer


def setup_logging(config):
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config["logging"]["log_file"]),
        ],
    )


def parse_args():

    parser = argparse.ArgumentParser(description="Train water segmentation model")

    # default=None means if not provided, we use the config.yaml value
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--bce_weight", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--freeze_encoder", action="store_true", default=None)
    parser.add_argument("--freeze_layers", type=str, default=None)  # "all" or "partial"
    return parser.parse_args()


def main():
    # Load base config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Override config with any command line arguments provided
    args = parse_args()
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.encoder is not None:
        config["training"]["encoder"] = args.encoder
    if args.bce_weight is not None:
        config["training"]["bce_weight"] = args.bce_weight
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.freeze_encoder:
        config["training"]["freeze_encoder"] = True
    if args.freeze_layers is not None:
        config["training"]["freeze_layers"] = args.freeze_layers
        h
    encoder = config["training"]["encoder"]
    lr = config["training"]["learning_rate"]
    bce = config["training"]["bce_weight"]
    
    epochs = config["training"]["num_epochs"]

    setup_logging(config)
    logger = logging.getLogger(__name__)

    
    device_str = config["training"]["device"]
    device = torch.device(
        "cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders_from_raw(config)

    
    model = build_model(config)

    if config["training"]["freeze_encoder"]:
        freeze_layers = config["training"].get("freeze_layers", "all")
        
        if freeze_layers == "all":
            for param in model.encoder.parameters():
                param.requires_grad = False
            mode = "resize_frozen"

        elif freeze_layers == "partial":
            # Freeze only early layers 
            for param in model.encoder.layer1.parameters():
                param.requires_grad = False
            for param in model.encoder.layer2.parameters():
                param.requires_grad = False
            mode = "resize_frozen_partial"
    else:
        mode = "resize"

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"Total:     {total:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Frozen:    {frozen:,}")
    
    run_name = f"{encoder}_lr{lr}_bce{bce}_{mode}_epochs{epochs}"

    config["training"]["checkpoint_path"] = f"checkpoints/{run_name}/best_model.pth"
    config["mlflow"]["run_name"] = run_name

    os.makedirs(f"checkpoints/{run_name}", exist_ok=True)

    
    logger.info(
        f"Model: {config['training']['model_name']} "
        f"with {config['training']['encoder']} encoder"
    )

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config["training"]["learning_rate"],
    weight_decay=config["training"]["weight_decay"],
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["num_epochs"], eta_min=1e-6
    )

    loss_fn = CombinedLoss(bce_weight=config["training"]["bce_weight"])

    #  Train

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=run_name):
        trainer = Trainer(model, optimizer, scheduler, loss_fn, device, config)
        trainer.fit(train_loader, val_loader)

        #Final evaluation on test set
        logger.info("Evaluating best model on test set...")
        checkpoint = torch.load(config["training"]["checkpoint_path"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        from src.metrics import compute_all_metrics

        test_totals = {k: 0.0 for k in ["iou", "dice", "accuracy", "precision", "recall"]}

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images)
                for k, v in compute_all_metrics(preds, masks).items():
                    test_totals[k] += v

        n_batches = len(test_loader)
        logger.info("Test set results:")
        for k, v in test_totals.items():
            logger.info(f"  {k}: {v / n_batches:.4f}")
        mlflow.log_metrics({f"test_{k}": v / n_batches for k, v in test_totals.items()})
    

if __name__ == "__main__":
    main()

 