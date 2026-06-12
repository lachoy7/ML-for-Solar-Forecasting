"""Temporal Fusion Transformer training and evaluation."""

from __future__ import annotations

import argparse

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tensorflow as tf
import tensorboard as tb  # noqa: F401
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from sklearn.metrics import r2_score

from ml_for_solar.config import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    TFT_HPARAMS,
    TRAIN_X_PATH,
    ensure_dirs,
)
from ml_for_solar.dataloading import build_dataloaders, build_datasets, load_dataframe, load_dataloaders, save_dataloaders

tf.io.gfile = tf.compat.v1.io.gfile


class TFTLightningModel(pl.LightningModule):
    def __init__(self, dataset, train_dataloader, val_dataloader, **hparams):
        super().__init__()
        self.model = TemporalFusionTransformer.from_dataset(dataset, **hparams)
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "dataset"])
        self.dataset_params = dataset.get_parameters()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def _shared_step(self, batch, stage: str):
        x, y = batch
        x = {k: v.to(self.device) for k, v in x.items()}
        y = tuple(v.to(self.device) if v is not None else None for v in y)
        output = self.model(x)
        loss = self.model.loss(output.prediction, y)
        batch_size = x["encoder_lengths"].shape[0]
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"), batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["dataset_params"] = self.dataset_params

    def on_load_checkpoint(self, checkpoint):
        self.dataset_params = checkpoint["dataset_params"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader


def _predictions_to_arrays(predictions):
    y_pred = predictions.output.detach().cpu().numpy().flatten()
    if isinstance(predictions.y, tuple):
        y_true = predictions.y[0].detach().cpu().numpy().flatten()
    else:
        y_true = predictions.y.detach().cpu().numpy().flatten()
    return y_true, y_pred


def train_tft(
    max_epochs: int = 50,
    accelerator: str = "auto",
    num_workers: int = 0,
    rebuild_dataloaders: bool = False,
) -> str:
    ensure_dirs()
    pl.seed_everything(42)

    if rebuild_dataloaders or not (PROCESSED_DIR / "train_dataloader.pth").exists():
        data_df = load_dataframe(TRAIN_X_PATH)
        training, validation, test = build_datasets(data_df)
        dataloaders = build_dataloaders(training, validation, test, num_workers=num_workers)
        save_dataloaders(*dataloaders)
    else:
        train_dataloader, val_dataloader, test_dataloader = load_dataloaders()
        data_df = load_dataframe(TRAIN_X_PATH)
        training, _, _ = build_datasets(data_df)
        dataloaders = (train_dataloader, val_dataloader, test_dataloader)

    train_dataloader, val_dataloader, test_dataloader = dataloaders

    tft_lightning = TFTLightningModel(
        training,
        train_dataloader,
        val_dataloader,
        **TFT_HPARAMS,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(CHECKPOINTS_DIR),
        filename="tft-best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        callbacks=[
            LearningRateMonitor(),
            EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, mode="min"),
            checkpoint_callback,
        ],
        logger=TensorBoardLogger(str(LOGS_DIR), name="tft"),
    )

    trainer.fit(tft_lightning, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        raise RuntimeError("No checkpoint was saved during training.")

    print(f"Best model saved at: {best_model_path}")
    evaluate_tft(best_model_path, train_dataloader, val_dataloader, test_dataloader, training)
    return best_model_path


def evaluate_tft(
    checkpoint_path: str,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    training,
) -> None:
    ensure_dirs()
    best_tft = TFTLightningModel.load_from_checkpoint(
        checkpoint_path,
        dataset=training,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **TFT_HPARAMS,
    )
    best_tft.model = TemporalFusionTransformer.from_dataset(training, **best_tft.hparams)

    for split_name, dataloader in [
        ("val", val_dataloader),
        ("train", train_dataloader),
        ("test", test_dataloader),
    ]:
        predictions = best_tft.model.predict(
            dataloader,
            return_y=True,
            trainer_kwargs=dict(accelerator="cpu"),
        )
        y_true, y_pred = _predictions_to_arrays(predictions)
        mae = MAE()(predictions.output, predictions.y)
        r2 = r2_score(y_true, y_pred)
        print(f"{split_name.title()} MAE: {mae:.4f}")
        print(f"{split_name.title()} R^2: {r2:.4f}")

        pred_path = PREDICTIONS_DIR / f"tft_{split_name}_predictions.csv"
        pd.DataFrame({"predicted": y_pred}).to_csv(pred_path, index=False)

    raw_predictions = best_tft.model.predict(
        test_dataloader,
        mode="raw",
        return_x=True,
        trainer_kwargs=dict(accelerator="cpu"),
    )
    best_tft.model.plot_prediction(
        x=raw_predictions.x,
        out=raw_predictions.output,
        idx=0,
        add_loss_to_title=True,
    )
    plt.savefig(PREDICTIONS_DIR / "tft_prediction_plot.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the TFT model.")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rebuild-dataloaders", action="store_true")
    args = parser.parse_args()

    train_tft(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        num_workers=args.num_workers,
        rebuild_dataloaders=args.rebuild_dataloaders,
    )


if __name__ == "__main__":
    main()
