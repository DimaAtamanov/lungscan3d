"""PyTorch Lightning data module."""

from __future__ import annotations

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

from lungscan3d.data.dataset import (
    AugmentationConfig,
    LazyChunkedPatchDataset,
    PatchDataset,
    has_chunked_patch_arrays,
    load_patch_arrays,
    load_patch_labels,
)
from lungscan3d.data.download import download_data
from lungscan3d.data.preprocessing import preprocess
from lungscan3d.data.splits import split_indices

LOGGER = logging.getLogger(__name__)


class LungScanDataModule(pl.LightningDataModule if pl is not None else object):
    """Lightning data module for synthetic and preprocessed LungScan3D datasets."""

    def __init__(self, config: Any) -> None:
        """Initialize data module."""
        super().__init__()
        self.config = config
        self.train_dataset: PatchDataset | LazyChunkedPatchDataset | None = None
        self.val_dataset: PatchDataset | LazyChunkedPatchDataset | None = None
        self.test_dataset: PatchDataset | LazyChunkedPatchDataset | None = None
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
        if self._processed_dataset_exists(processed_dir):
            LOGGER.info("Processed dataset already exists: %s", processed_dir)
            return

        LOGGER.info("Processed dataset is missing; trying DVC/internet/manual data step")
        download_data(self.config)
        if not self._processed_dataset_exists(processed_dir):
            LOGGER.info("Running preprocessing into %s", processed_dir)
            preprocess(self.config)

    @staticmethod
    def _processed_dataset_exists(processed_dir: Path) -> bool:
        legacy_exists = (processed_dir / "volumes.npy").exists() and (
            processed_dir / "labels.npy"
        ).exists()
        chunked_exists = (processed_dir / "manifest.csv").exists() and (
            processed_dir / "labels.npy"
        ).exists()
        return legacy_exists or chunked_exists

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for train, validation, and test stages."""
        del stage
        if str(self.config.data.name) == "synthetic":
            processed_dir = Path(self.config.paths.processed_dir) / "synthetic"
        else:
            processed_dir = Path(self.config.data.processed_dir)

        LOGGER.info("Loading patch metadata from %s", processed_dir)
        uses_chunked_arrays = has_chunked_patch_arrays(processed_dir)
        volumes: np.ndarray | None = None
        if uses_chunked_arrays:
            labels = load_patch_labels(processed_dir)
        else:
            volumes, labels = load_patch_arrays(processed_dir)

        train_idx, val_idx, test_idx = split_indices(
            num_items=len(labels),
            train_fraction=float(self.config.data.train_fraction),
            val_fraction=float(self.config.data.val_fraction),
            test_fraction=float(self.config.data.test_fraction),
            seed=int(self.config.seed),
        )
        augmentation = self._build_train_augmentation()

        if uses_chunked_arrays:
            self.train_dataset = LazyChunkedPatchDataset(processed_dir, train_idx, augmentation)
            self.val_dataset = LazyChunkedPatchDataset(processed_dir, val_idx)
            self.test_dataset = LazyChunkedPatchDataset(processed_dir, test_idx)
        else:
            if volumes is None:
                raise RuntimeError("Legacy arrays were not loaded")
            self.train_dataset = PatchDataset(volumes[train_idx], labels[train_idx], augmentation)
            self.val_dataset = PatchDataset(volumes[val_idx], labels[val_idx])
            self.test_dataset = PatchDataset(volumes[test_idx], labels[test_idx])

        self.train_labels = labels[train_idx].astype(np.float32, copy=False)
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
        """Build augmentation configuration for the training dataset."""
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
        """Build a class-balanced weighted sampler for imbalanced candidate labels."""
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

    def train_dataloader(self) -> DataLoader[tuple[Any, Any]]:
        """Return train dataloader."""
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
        """Return validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("setup() must be called before val_dataloader()")
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.config.data.batch_size),
            shuffle=False,
            num_workers=int(self.config.data.num_workers),
        )

    def test_dataloader(self) -> DataLoader[tuple[Any, Any]]:
        """Return test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("setup() must be called before test_dataloader()")
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.config.data.batch_size),
            shuffle=False,
            num_workers=int(self.config.data.num_workers),
        )
