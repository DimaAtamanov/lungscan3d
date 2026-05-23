"""PyTorch Lightning data module."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

from lungscan3d.data.dataset import AugmentationConfig, PatchDataset, load_patch_arrays
from lungscan3d.data.download import download_data
from lungscan3d.data.preprocessing import preprocess
from lungscan3d.data.splits import split_indices

LOGGER = logging.getLogger(__name__)


class LungScanDataModule(pl.LightningDataModule if pl is not None else object):
    """Lightning data module for synthetic and preprocessed LungScan3D datasets."""

    def __init__(self, config: Any) -> None:
        """Initialize data module.

        Args:
        ----
            config: Hydra configuration object.

        """
        super().__init__()
        self.config = config
        self.train_dataset: PatchDataset | None = None
        self.val_dataset: PatchDataset | None = None
        self.test_dataset: PatchDataset | None = None
        self.train_labels: np.ndarray | None = None

    def prepare_data(self) -> None:
        """Ensure data exists locally before setup."""
        if not bool(self.config.data.ensure_data):
            LOGGER.info("Data availability checks are disabled by config")
            return
        data_name = str(self.config.data.name)
        LOGGER.info("Preparing data for mode: %s", data_name)
        if data_name == "synthetic":
            download_data(self.config)
            return

        processed_dir = Path(self.config.data.processed_dir)
        if (processed_dir / "volumes.npy").exists() and (processed_dir / "labels.npy").exists():
            LOGGER.info("Processed dataset already exists: %s", processed_dir)
            return

        LOGGER.info("Processed dataset is missing; trying DVC/internet/manual data step")
        download_data(self.config)
        if not (processed_dir / "volumes.npy").exists():
            LOGGER.info("Running preprocessing into %s", processed_dir)
            preprocess(self.config)

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for train, validation, and test stages.

        Args:
        ----
            stage: Optional Lightning stage name.

        """
        del stage
        if str(self.config.data.name) == "synthetic":
            processed_dir = Path(self.config.paths.processed_dir) / "synthetic"
        else:
            processed_dir = Path(self.config.data.processed_dir)
        LOGGER.info("Loading patch arrays from %s", processed_dir)
        volumes, labels = load_patch_arrays(processed_dir)
        train_idx, val_idx, test_idx = split_indices(
            num_items=len(labels),
            train_fraction=float(self.config.data.train_fraction),
            val_fraction=float(self.config.data.val_fraction),
            test_fraction=float(self.config.data.test_fraction),
            seed=int(self.config.seed),
        )
        augmentation = self._build_train_augmentation()
        self.train_dataset = PatchDataset(volumes[train_idx], labels[train_idx], augmentation)
        self.val_dataset = PatchDataset(volumes[val_idx], labels[val_idx])
        self.test_dataset = PatchDataset(volumes[test_idx], labels[test_idx])
        self.train_labels = labels[train_idx].astype(np.float32)
        LOGGER.info(
            "Dataset split sizes: train=%d, val=%d, test=%d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )
        LOGGER.info(
            "Train class balance: positive=%d, negative=%d",
            int(self.train_labels.sum()),
            int(len(self.train_labels) - self.train_labels.sum()),
        )

    def _build_train_augmentation(self) -> AugmentationConfig | None:
        """Build augmentation configuration for the training dataset.

        Returns
        -------
            Augmentation config, or ``None`` when augmentations are disabled.

        """
        if not bool(self.config.preprocessing.augment.enabled):
            LOGGER.info("Training augmentations are disabled")
            return None
        LOGGER.info("Training augmentations are enabled")
        return AugmentationConfig(
            random_flip=bool(self.config.preprocessing.augment.random_flip),
            random_rotate90=bool(self.config.preprocessing.augment.random_rotate90),
            gaussian_noise_std=float(self.config.preprocessing.augment.gaussian_noise_std),
            random_shift_voxels=int(self.config.preprocessing.augment.random_shift_voxels),
        )

    def _build_weighted_sampler(self) -> WeightedRandomSampler | None:
        """Build a class-balanced weighted sampler for imbalanced candidate labels.

        Returns
        -------
            Weighted sampler for the train split, or ``None`` when disabled.

        """
        if not bool(self.config.data.weighted_sampling.enabled):
            LOGGER.info("Weighted sampling is disabled")
            return None
        if self.train_labels is None:
            raise RuntimeError("setup() must be called before creating sampler")
        labels = torch.as_tensor(self.train_labels, dtype=torch.float32)
        positive_count = torch.clamp(labels.sum(), min=1.0)
        negative_count = torch.clamp((1.0 - labels).sum(), min=1.0)
        positive_weight = len(labels) / (2.0 * positive_count)
        negative_weight = len(labels) / (2.0 * negative_count)
        sample_weights = torch.where(labels > 0.5, positive_weight, negative_weight)
        sample_weights = self._apply_hard_negative_weights(sample_weights, labels)
        LOGGER.info(
            "Using weighted sampling: positive_weight=%.4f, negative_weight=%.4f",
            float(positive_weight),
            float(negative_weight),
        )
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def _apply_hard_negative_weights(
        self, sample_weights: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Increase sampling weights for previously selected hard negatives.

        Args:
        ----
            sample_weights: Base class-balanced sample weights.
            labels: Train labels with shape ``(N,)``.

        Returns:
        -------
            Updated sample weights.

        """
        hard_negative_config = getattr(self.config.data, "hard_negative_mining", None)
        if hard_negative_config is None or not bool(hard_negative_config.enabled):
            return sample_weights
        indices_path = Path(str(hard_negative_config.indices_path))
        if not indices_path.exists():
            LOGGER.info(
                "Hard negative indices were not found at %s; using class-balanced sampler only",
                indices_path,
            )
            return sample_weights
        indices = np.load(indices_path).astype(np.int64)
        valid_indices = indices[(indices >= 0) & (indices < len(sample_weights))]
        if len(valid_indices) == 0:
            LOGGER.info("Hard negative file %s contains no valid train indices", indices_path)
            return sample_weights
        negative_mask = labels[torch.as_tensor(valid_indices)] < 0.5
        selected_indices = torch.as_tensor(valid_indices, dtype=torch.long)[negative_mask]
        multiplier = float(hard_negative_config.weight_multiplier)
        sample_weights[selected_indices] = sample_weights[selected_indices] * multiplier
        LOGGER.info(
            "Applied hard negative mining weights to %d negatives with multiplier=%.2f",
            len(selected_indices),
            multiplier,
        )
        return sample_weights

    def train_dataloader(self) -> DataLoader[tuple[Any, Any]]:
        """Return train dataloader.

        Returns
        -------
            Dataloader over training patches.

        """
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        sampler = self._build_weighted_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config.data.batch_size),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(self.config.data.num_workers),
        )

    def val_dataloader(self) -> DataLoader[tuple[Any, Any]]:
        """Return validation dataloader.

        Returns
        -------
            Dataloader over validation patches.

        """
        if self.val_dataset is None:
            raise RuntimeError("setup() must be called before val_dataloader()")
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.config.data.batch_size),
            shuffle=False,
            num_workers=int(self.config.data.num_workers),
        )

    def test_dataloader(self) -> DataLoader[tuple[Any, Any]]:
        """Return test dataloader.

        Returns
        -------
            Dataloader over test patches.

        """
        if self.test_dataset is None:
            raise RuntimeError("setup() must be called before test_dataloader()")
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.config.data.batch_size),
            shuffle=False,
            num_workers=int(self.config.data.num_workers),
        )
