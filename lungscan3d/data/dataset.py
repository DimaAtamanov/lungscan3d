"""Dataset definitions and 3D augmentations for LungScan3D."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for lightweight 3D CT augmentations.

    Attributes
    ----------
        random_flip: Whether to randomly flip the patch along spatial axes.
        random_rotate90: Whether to randomly rotate the patch by multiples of 90 degrees.
        gaussian_noise_std: Standard deviation of additive Gaussian noise.
        random_shift_voxels: Maximum absolute spatial shift in voxels for each axis.

    """

    random_flip: bool = False
    random_rotate90: bool = False
    gaussian_noise_std: float = 0.0
    random_shift_voxels: int = 0


def _random_shift(volume: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Randomly shift a 3D volume and zero-fill uncovered areas.

    Args:
    ----
        volume: Tensor with shape ``(C, D, H, W)``.
        max_shift: Maximum absolute shift in voxels for each spatial axis.

    Returns:
    -------
        Shifted tensor with the same shape.

    """
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
    """Apply stochastic 3D augmentations to a CT patch.

    Args:
    ----
        volume: Tensor with shape ``(C, D, H, W)``.
        config: Augmentation configuration.

    Returns:
    -------
        Augmented tensor with shape ``(C, D, H, W)``.

    """
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
        self.volumes = volumes.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.augmentation = augmentation

    def __len__(self) -> int:
        """Return number of samples."""
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single volume and label.

        Args:
        ----
            index: Sample index.

        Returns:
        -------
            Tuple ``(volume, label)`` where volume is a float tensor and label has shape ``(1,)``.

        """
        volume = torch.from_numpy(self.volumes[index])
        if self.augmentation is not None:
            volume = apply_augmentations(volume, self.augmentation)
        label = torch.tensor([self.labels[index]], dtype=torch.float32)
        return volume, label


def load_patch_arrays(processed_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed patch arrays from disk.

    Args:
    ----
        processed_dir: Directory containing ``volumes.npy`` and ``labels.npy``.

    Returns:
    -------
        Tuple with volumes and labels arrays.

    """
    directory = Path(processed_dir)
    LOGGER.info("Loading volumes from %s", directory / "volumes.npy")
    LOGGER.info("Loading labels from %s", directory / "labels.npy")
    volumes = np.load(directory / "volumes.npy")
    labels = np.load(directory / "labels.npy")
    LOGGER.info("Loaded %d patches with shape %s", len(labels), volumes.shape[1:])
    return volumes, labels
