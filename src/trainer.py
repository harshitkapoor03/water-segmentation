import os
import torch
import mlflow
import logging
from tqdm import tqdm
from src.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.best_iou = 0.0
        self.patience_counter = 0

        # Create checkpoints directory if it doesn't exist
        os.makedirs(
            os.path.dirname(config["training"]["checkpoint_path"]), exist_ok=True
        )

    def _run_epoch(self, loader, is_training):
        """
        Runs one pass through the dataset — either training or validation.

        is_training=True:  compute gradients, update weights
        is_training=False: just measure, don't touch weights

        """
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        # Dictionary comprehension — creates {key: 0.0} for each metric name
        all_metrics = {
            k: 0.0 for k in ["iou", "dice", "accuracy", "precision", "recall"]
        }

        # torch.no_grad() tells PyTorch not to track operations for gradients
        # During validation we never call .backward() so there's no reason
        # to build the computation graph — this halves memory usage
        context = torch.no_grad() if not is_training else torch.enable_grad()

        with context:
            for images, masks in tqdm(
                loader, desc="Train" if is_training else "Val", leave=False
            ):
                images = images.to(self.device)
                masks = masks.to(self.device)

                if is_training:
                    # Zero out gradients from the previous batch
                    # PyTorch accumulates gradients by default —
                    # if you don't zero them, batch N's gradients
                    # add to batch N-1's, corrupting the update
                    self.optimizer.zero_grad()

                predictions = self.model(images)
                loss = self.loss_fn(predictions, masks)

                if is_training:
                    loss.backward()

                    # Gradient clipping: if any gradient vector has magnitude > 1.0,
                    # scale it down to exactly 1.0. Prevents exploding gradients
                    # early in training when loss is high and gradients are large.
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.optimizer.step()

                total_loss += loss.item()

                # .detach() disconnects predictions from computation graph
                # We don't need gradients for metric computation
                batch_metrics = compute_all_metrics(predictions.detach(), masks)
                for k in all_metrics:
                    all_metrics[k] += batch_metrics[k]

        n = len(loader)
        avg_loss = total_loss / n
        avg_metrics = {k: v / n for k, v in all_metrics.items()}

        return avg_loss, avg_metrics

    def save_checkpoint(self, epoch, metrics):
        path = self.config["training"]["checkpoint_path"]
        # Save everything needed to resume training or run inference
        # Including config means predictor.py can rebuild the model
        # architecture without needing a separate config file
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved checkpoint -> {path} (IoU: {metrics['iou']:.4f})")

    def fit(self, train_loader, val_loader):
        # mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        # mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        # run_name = self.config["mlflow"].get("run_name", None)
        # with mlflow.start_run(run_name=run_name):
        # Log all hyperparameters at the start of the run
        # This is what makes MLflow useful — you can compare
        # these values across all 4 of your experiment runs
        mlflow.log_params(
            {
                "model": self.config["training"]["model_name"],
                "encoder": self.config["training"]["encoder"],
                "lr": self.config["training"]["learning_rate"],
                "weight_decay": self.config["training"]["weight_decay"],
                "batch_size": self.config["training"]["batch_size"],
                "freeze_encoder": self.config["training"]["freeze_encoder"],
                "freeze_layers": self.config["training"].get("freeze_layers", "none"),
                "bce_weight": self.config["training"]["bce_weight"],  # ,
                # "patch_size":  self.config["data"]["patch_size"],
                # "stride":      self.config["data"]["stride"],
            }
        )

        epochs = self.config["training"]["num_epochs"]

        for epoch in range(1, epochs + 1):
            train_loss, train_m = self._run_epoch(train_loader, is_training=True)
            val_loss, val_m = self._run_epoch(val_loader, is_training=False)

            # Step the scheduler after each epoch
            # CosineAnnealingLR adjusts learning rate along a cosine curve
            # Large steps early in training, tiny steps late
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val IoU: {val_m['iou']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Log metrics to MLflow — one data point per epoch
            # These become the curves you see in the MLflow UI
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "val_iou": val_m["iou"],
                    "val_dice": val_m["dice"],
                    "val_accuracy": val_m["accuracy"],
                    "val_precision": val_m["precision"],
                    "val_recall": val_m["recall"],
                    "train_iou": train_m["iou"],
                    "epoch": epoch,
                },
                step=epoch,
            )

            # Save checkpoint only when IoU improves
            if val_m["iou"] > self.best_iou:
                self.best_iou = val_m["iou"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_m)
                mlflow.log_artifact(self.config["training"]["checkpoint_path"])
            else:
                self.patience_counter += 1
                if (
                    self.patience_counter
                    >= self.config["training"]["early_stopping_patience"]
                ):
                    logger.info(
                        f"Early stopping at epoch {epoch}. "
                        f"Best Val IoU: {self.best_iou:.4f}"
                    )
                    break

        mlflow.log_metric("best_val_iou", self.best_iou)

        logger.info(f"Training complete. Best Val IoU: {self.best_iou:.4f}")
