"""Dataset definitions and 3D augmentations for LungScan3D."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for lightweight 3D CT augmentations."""

    random_flip: bool = False
    random_rotate90: bool = False
    gaussian_noise_std: float = 0.0
    random_shift_voxels: int = 0


def _random_shift(volume: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Randomly shift a 3D volume and zero-fill uncovered areas."""
    if max_shift <= 0:
        return volume
    shifts = [int(torch.randint(-max_shift, max_shift + 1, (1,)).item()) for _ in range(3)]
    shifted = torch.zeros_like(volume)
    source_slices: list[slice] = [slice(None)]
    target_slices: list[slice] = [slice(None)]
    for axis_size, shift in zip(volume.shape[1:], shifts, strict=True):
        if shift >= 0:
            source_slices.append(slice(0, axis_size - shift))
            target_slices.append(slice(shift, axis_size))
        else:
            source_slices.append(slice(-shift, axis_size))
            target_slices.append(slice(0, axis_size + shift))
    shifted[tuple(target_slices)] = volume[tuple(source_slices)]
    return shifted


def apply_augmentations(volume: torch.Tensor, config: AugmentationConfig) -> torch.Tensor:
    """Apply stochastic 3D augmentations to a CT patch."""
    augmented = volume.clone()
    if config.random_flip:
        for axis in (1, 2, 3):
            if torch.rand(()) < 0.5:
                augmented = torch.flip(augmented, dims=(axis,))
    if config.random_rotate90:
        k = int(torch.randint(0, 4, (1,)).item())
        augmented = torch.rot90(augmented, k=k, dims=(2, 3))
    augmented = _random_shift(augmented, int(config.random_shift_voxels))
    if config.gaussian_noise_std > 0:
        noise = torch.randn_like(augmented) * float(config.gaussian_noise_std)
        augmented = augmented + noise
    return augmented


class PatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset backed by preprocessed 3D patches stored as NumPy arrays."""

    def __init__(
        self,
        volumes: np.ndarray,
        labels: np.ndarray,
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
        ----
            volumes: Array with shape ``(N, C, D, H, W)``.
            labels: Binary labels with shape ``(N,)``.
            augmentation: Optional stochastic augmentation config.

        """
        if volumes.ndim != 5:
            raise ValueError("volumes must have shape (N, C, D, H, W)")
        if labels.ndim != 1:
            raise ValueError("labels must have shape (N,)")
        if len(volumes) != len(labels):
            raise ValueError("volumes and labels must have the same length")
        self.volumes = volumes.astype(np.float32, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.augmentation = augmentation

    def __len__(self) -> int:
        """Return number of samples."""
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single volume and label."""
        volume = torch.from_numpy(np.asarray(self.volumes[index], dtype=np.float32))
        if self.augmentation is not None:
            volume = apply_augmentations(volume, self.augmentation)
        label = torch.tensor([self.labels[index]], dtype=torch.float32)
        return volume, label


class LazyChunkedPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset that loads preprocessed LUNA16 patch chunks lazily from disk."""

    def __init__(
        self,
        processed_dir: str | Path,
        indices: np.ndarray,
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        """Initialize a lazy chunked patch dataset.

        Args:
        ----
            processed_dir: Directory with preprocessed LUNA16 files. It must
                contain ``manifest.csv``, ``labels.npy`` and the ``chunks/``
                directory with chunked volume arrays.
            indices: Global sample indices that belong to this dataset split,
                for example train, validation or test indices.
            augmentation: Optional stochastic augmentation config.

        """
        self.processed_dir = Path(processed_dir)
        self.manifest_path = self.processed_dir / "manifest.csv"
        self.labels_path = self.processed_dir / "labels.npy"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {self.labels_path}")

        self.manifest = pd.read_csv(self.manifest_path)
        self.labels = np.load(self.labels_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augmentation = augmentation
        self._cached_volume_path: Path | None = None
        self._cached_volumes: np.ndarray | None = None

        if len(self.manifest) != len(self.labels):
            raise ValueError(
                "manifest.csv and labels.npy describe different numbers of samples: "
                f"{len(self.manifest)} != {len(self.labels)}"
            )

    def __len__(self) -> int:
        """Return number of samples."""
        return int(len(self.indices))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single volume and label."""
        global_index = int(self.indices[index])
        row = self.manifest.iloc[global_index]
        volume_path = self.processed_dir / str(row.volume_path)
        local_index = int(row.local_index)
        volumes = self._load_volume_chunk(volume_path)
        volume = torch.from_numpy(np.asarray(volumes[local_index], dtype=np.float32).copy())
        if self.augmentation is not None:
            volume = apply_augmentations(volume, self.augmentation)
        label = torch.tensor([float(self.labels[global_index])], dtype=torch.float32)
        return volume, label

    def _load_volume_chunk(self, volume_path: Path) -> np.ndarray:
        if self._cached_volume_path != volume_path or self._cached_volumes is None:
            LOGGER.debug("Memory-mapping preprocessed chunk: %s", volume_path)
            self._cached_volumes = np.load(volume_path, mmap_mode="r")
            self._cached_volume_path = volume_path
        return self._cached_volumes


def has_chunked_patch_arrays(processed_dir: str | Path) -> bool:
    """Return whether processed_dir contains the streamed chunked format."""
    directory = Path(processed_dir)
    return (directory / "manifest.csv").exists() and (directory / "labels.npy").exists()


def load_patch_labels(processed_dir: str | Path) -> np.ndarray:
    """Load labels without loading all patch volumes."""
    directory = Path(processed_dir)
    LOGGER.info("Loading labels from %s", directory / "labels.npy")
    return np.load(directory / "labels.npy").astype(np.float32, copy=False)


def load_patch_arrays(processed_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load legacy monolithic preprocessed patch arrays from disk."""
    directory = Path(processed_dir)
    LOGGER.info("Loading volumes from %s", directory / "volumes.npy")
    LOGGER.info("Loading labels from %s", directory / "labels.npy")
    volumes = np.load(directory / "volumes.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy")
    LOGGER.info("Loaded %d patches with shape %s", len(labels), volumes.shape[1:])
    return volumes, labels
