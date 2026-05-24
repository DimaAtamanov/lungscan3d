"""Training entry point."""

import logging
from typing import Any

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger

from lungscan3d.data.datamodule import LungScanDataModule
from lungscan3d.models import build_model
from lungscan3d.training.callbacks import MetricsHistoryCallback
from lungscan3d.training.lightning_module import LungScanLightningModule
from lungscan3d.training.plots import save_training_plots
from lungscan3d.utils.git import get_git_commit
from lungscan3d.utils.paths import ensure_dir

LOGGER = logging.getLogger(__name__)


def _build_loggers(config: Any) -> list[Logger] | bool:
    """Build optional experiment loggers from ``logging.mode``."""
    mode = str(getattr(config.logging, "mode", "none")).lower()
    if mode in {"none", "off", "disabled", "false"}:
        LOGGER.info("Experiment logging is disabled (logging.mode=none)")
        return False

    loggers: list[Logger] = []
    if mode in {"mlflow", "all"}:
        loggers.append(
            MLFlowLogger(
                experiment_name=str(config.logging.experiment_name),
                tracking_uri=str(config.logging.mlflow_tracking_uri),
            )
        )
    if mode in {"tensorboard", "all"}:
        loggers.append(
            TensorBoardLogger(
                save_dir=str(config.logging.tensorboard_save_dir),
                name=str(config.project_name),
            )
        )
    if not loggers:
        raise ValueError("logging.mode must be one of: none, mlflow, tensorboard, all")

    hyperparameters = OmegaConf.to_container(config, resolve=True)
    metadata: dict[str, str] = {}
    if bool(config.logging.log_hyperparameters):
        metadata["config"] = str(hyperparameters)
    if bool(config.logging.log_git_commit):
        metadata["git_commit"] = get_git_commit()
    if metadata:
        LOGGER.info("Logging hyperparameters and git metadata")
        for logger in loggers:
            logger.log_hyperparams(metadata)
    return loggers


def train(config: Any) -> None:
    """Train a LungScan3D model."""
    LOGGER.info(
        "Starting training: project=%s, data=%s, model=%s",
        config.project_name,
        config.data.name,
        config.model.name,
    )
    pl.seed_everything(int(config.seed), workers=True)
    ensure_dir(config.paths.checkpoints_dir)
    ensure_dir(config.paths.plots_dir)

    LOGGER.info("Building DataModule")
    datamodule = LungScanDataModule(config)
    LOGGER.info("Building model: %s", config.model.name)
    model = build_model(config)
    lightning_module = LungScanLightningModule(model=model, config=config)

    loggers = _build_loggers(config)
    metrics_history = MetricsHistoryCallback()
    checkpoint = ModelCheckpoint(
        dirpath=str(config.paths.checkpoints_dir),
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=int(config.trainer.max_epochs),
        accelerator=str(config.trainer.accelerator),
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        num_sanity_val_steps=int(config.trainer.num_sanity_val_steps),
        fast_dev_run=bool(config.trainer.fast_dev_run),
        deterministic=bool(config.trainer.deterministic),
        benchmark=bool(config.trainer.benchmark),
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="val/loss", mode="min", patience=5),
            LearningRateMonitor(logging_interval="epoch"),
            metrics_history,
        ],
        logger=loggers,
    )
    LOGGER.info(
        "Launching Lightning trainer for %s epoch(s)", config.trainer.max_epochs
    )
    trainer.fit(lightning_module, datamodule=datamodule)
    best_checkpoint_path = checkpoint.best_model_path
    LOGGER.info(
        "Training finished. Best checkpoint: %s",
        best_checkpoint_path or "not available",
    )

    if best_checkpoint_path:
        LOGGER.info(
            "Loading best checkpoint for test evaluation: %s", best_checkpoint_path
        )
        test_module = LungScanLightningModule.load_from_checkpoint(
            checkpoint_path=str(best_checkpoint_path),
            model=model,
            config=config,
            weights_only=False,
        )
    else:
        LOGGER.warning(
            "Best checkpoint is not available; running test evaluation with current model weights"
        )
        test_module = lightning_module

    LOGGER.info("Running test evaluation")
    trainer.test(test_module, datamodule=datamodule, ckpt_path=None)

    save_training_plots(metrics_history.history, config.paths.plots_dir)
    LOGGER.info("Training plots saved to %s", config.paths.plots_dir)
