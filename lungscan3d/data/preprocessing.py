"""CT preprocessing utilities."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lungscan3d.utils.paths import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CtMetadata:
    """Metadata needed to map world coordinates into voxel coordinates.

    Attributes
    ----------
        spacing_zyx: Voxel spacing in ``(Z, Y, X)`` millimeters.
        origin_xyz: CT origin in LUNA16 world coordinates.

    """

    spacing_zyx: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]


def normalize_hu(volume_hu: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """Clip and normalize CT Hounsfield units to ``[-1, 1]``.

    Args:
    ----
        volume_hu: CT volume in HU units.
        clip_min: Lower HU clipping value.
        clip_max: Upper HU clipping value.

    Returns:
    -------
        Normalized float32 volume.

    Raises:
    ------
        ValueError: If the clipping range is invalid.

    """
    if clip_max <= clip_min:
        raise ValueError("clip_max must be greater than clip_min")
    clipped = np.clip(volume_hu.astype(np.float32), clip_min, clip_max)
    normalized = (clipped - clip_min) / (clip_max - clip_min)
    return (2.0 * normalized - 1.0).astype(np.float32)


def crop_or_pad_patch(
    volume: np.ndarray,
    center_zyx: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> np.ndarray:
    """Crop a 3D patch and pad with zeros near borders.

    Args:
    ----
        volume: Input volume with shape ``(D, H, W)``.
        center_zyx: Center coordinate in voxel order.
        patch_size: Output patch size in ``(D, H, W)`` order.

    Returns:
    -------
        Patch with shape ``patch_size``.

    Raises:
    ------
        ValueError: If input dimensions are invalid.

    """
    if volume.ndim != 3:
        raise ValueError("volume must have shape (D, H, W)")
    if len(center_zyx) != 3 or len(patch_size) != 3:
        raise ValueError("center_zyx and patch_size must contain exactly three values")
    patch = np.zeros(patch_size, dtype=volume.dtype)
    starts = [center - size // 2 for center, size in zip(center_zyx, patch_size, strict=True)]
    ends = [start + size for start, size in zip(starts, patch_size, strict=True)]

    src_slices = []
    dst_slices = []
    for axis, (start, end) in enumerate(zip(starts, ends, strict=True)):
        src_start = max(start, 0)
        src_end = min(end, volume.shape[axis])
        dst_start = max(-start, 0)
        dst_end = dst_start + max(src_end - src_start, 0)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    patch[tuple(dst_slices)] = volume[tuple(src_slices)]
    return patch


def world_to_voxel_zyx(
    coord_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    spacing_zyx: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Convert CT world coordinates to voxel coordinates.

    Args:
    ----
        coord_xyz: Candidate center in world coordinates ``(X, Y, Z)``.
        origin_xyz: Image origin in world coordinates ``(X, Y, Z)``.
        spacing_zyx: Voxel spacing in ``(Z, Y, X)`` order.

    Returns:
    -------
        Rounded voxel coordinate in ``(Z, Y, X)`` order.

    """
    spacing_xyz = np.array([spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]], dtype=np.float32)
    voxel_xyz = (
        np.array(coord_xyz, dtype=np.float32) - np.array(origin_xyz, dtype=np.float32)
    ) / spacing_xyz
    voxel_zyx = voxel_xyz[::-1]
    return tuple(int(round(value)) for value in voxel_zyx)


def preprocess(config: Any) -> None:
    """Run preprocessing for configured dataset.

    Synthetic mode only ensures that generated arrays exist. LUNA16 mode converts
    raw ``.mhd/.raw`` volumes plus ``candidates.csv`` metadata into cached NumPy
    arrays with shape ``(N, 1, D, H, W)`` and labels with shape ``(N,)``.

    Args:
    ----
        config: Hydra configuration object.

    """
    data_name = str(config.data.name)
    LOGGER.info("Starting preprocessing for data mode: %s", data_name)
    if data_name == "synthetic":
        from lungscan3d.data.download import download_data

        download_data(config)
        return
    if data_name == "luna16":
        preprocess_candidate_csv(
            raw_dir=Path(config.data.raw_dir),
            metadata_csv=Path(config.data.candidates_csv),
            output_dir=Path(config.data.processed_dir),
            patch_size=tuple(int(value) for value in config.data.patch_size),
            clip_min=float(config.preprocessing.clip_hu_min),
            clip_max=float(config.preprocessing.clip_hu_max),
            task=str(config.data.task),
        )
        return
    raise ValueError(f"Unknown dataset mode: {data_name}")


def preprocess_candidate_csv(
    raw_dir: Path,
    metadata_csv: Path,
    output_dir: Path,
    patch_size: tuple[int, int, int],
    clip_min: float,
    clip_max: float,
    task: str,
) -> None:
    """Preprocess candidate rows into arrays suitable for training.

    Args:
    ----
        raw_dir: Directory containing LUNA16 subset folders with ``.mhd/.raw`` files.
        metadata_csv: CSV containing at least ``seriesuid``, ``coordX``, ``coordY``,
            and ``coordZ``. It should contain ``class`` for LUNA16 candidate labels
            or a generic binary ``label`` column.
        output_dir: Directory where ``volumes.npy`` and ``labels.npy`` are written.
        patch_size: Output patch size in ``(D, H, W)`` order.
        clip_min: Lower HU clipping value.
        clip_max: Upper HU clipping value.
        task: Task name used for label extraction.

    Raises:
    ------
        FileNotFoundError: If metadata or CT files are missing.
        ValueError: If required columns or labels are invalid.

    """
    LOGGER.info("Preprocessing candidates: raw_dir=%s, metadata_csv=%s", raw_dir, metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    mhd_index = _index_mhd_files(raw_dir)
    if not mhd_index:
        raise FileNotFoundError(f"No .mhd files found under {raw_dir}")
    LOGGER.info("Indexed %d CT volumes", len(mhd_index))
    rows = pd.read_csv(metadata_csv)
    _validate_candidate_columns(rows)
    LOGGER.info("Loaded %d candidate rows", len(rows))
    output_path = ensure_dir(output_dir)

    volumes: list[np.ndarray] = []
    labels: list[float] = []
    volume_cache: dict[str, tuple[np.ndarray, CtMetadata]] = {}

    for row in rows.itertuples(index=False):
        seriesuid = str(row.seriesuid)
        if seriesuid not in mhd_index:
            LOGGER.warning("Skipping candidate because CT file is missing: seriesuid=%s", seriesuid)
            continue
        label = _extract_label(row=row, task=task)
        if label is None:
            continue
        if seriesuid not in volume_cache:
            ct_array, metadata = read_mhd_volume(mhd_index[seriesuid])
            ct_array = normalize_hu(ct_array, clip_min=clip_min, clip_max=clip_max)
            volume_cache[seriesuid] = (ct_array, metadata)
        ct_array, metadata = volume_cache[seriesuid]
        coord_xyz = (
            float(row.coordX),
            float(row.coordY),
            float(row.coordZ),
        )
        center_zyx = world_to_voxel_zyx(
            coord_xyz=coord_xyz,
            origin_xyz=metadata.origin_xyz,
            spacing_zyx=metadata.spacing_zyx,
        )
        patch = crop_or_pad_patch(ct_array, center_zyx=center_zyx, patch_size=patch_size)
        volumes.append(patch[None, ...].astype(np.float32))
        labels.append(float(label))

    if not volumes:
        raise ValueError("No training samples were produced from the metadata CSV")
    volumes_array = np.stack(volumes, axis=0).astype(np.float32)
    labels_array = np.asarray(labels, dtype=np.float32)
    np.save(output_path / "volumes.npy", volumes_array)
    np.save(output_path / "labels.npy", labels_array)
    LOGGER.info(
        "Preprocessing finished: samples=%d, positives=%d, output_dir=%s",
        len(labels_array),
        int(labels_array.sum()),
        output_path,
    )


def read_mhd_volume(path: str | Path) -> tuple[np.ndarray, CtMetadata]:
    """Read an MHD CT volume with SimpleITK.

    Args:
    ----
        path: Path to ``.mhd`` file.

    Returns:
    -------
        Tuple with array in ``(D, H, W)`` order and metadata.

    """
    import SimpleITK as sitk

    LOGGER.info("Reading CT volume: %s", path)
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    origin_raw = image.GetOrigin()
    origin_xyz = (float(origin_raw[0]), float(origin_raw[1]), float(origin_raw[2]))
    return array, CtMetadata(spacing_zyx=spacing_zyx, origin_xyz=origin_xyz)


def _index_mhd_files(raw_dir: Path) -> dict[str, Path]:
    """Index MHD files by stem/series UID.

    Args:
    ----
        raw_dir: Directory containing MHD files recursively.

    Returns:
    -------
        Mapping from series UID to MHD path.

    """
    return {path.stem: path for path in raw_dir.rglob("*.mhd")}


def _validate_candidate_columns(rows: pd.DataFrame) -> None:
    """Validate core metadata columns.

    Args:
    ----
        rows: Candidate metadata dataframe.

    Raises:
    ------
        ValueError: If required columns are missing.

    """
    required_columns = {"seriesuid", "coordX", "coordY", "coordZ"}
    missing_columns = required_columns.difference(rows.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")


def _extract_label(row: Any, task: str) -> float:
    """Extract a LUNA16 binary candidate label from a metadata row.

    Args:
    ----
        row: Row returned by ``DataFrame.itertuples``.
        task: Task name. The supported production task is
            ``nodule_vs_non_nodule``.

    Returns:
    -------
        Binary label where ``1.0`` means true nodule candidate and ``0.0`` means
        false positive candidate.

    Raises:
    ------
        ValueError: If required labels are unavailable or the task is unsupported.

    """
    if hasattr(row, "label"):
        return float(row.label)
    if task == "nodule_vs_non_nodule" and hasattr(row, "class"):
        return float(getattr(row, "class"))
    raise ValueError("Could not extract a LUNA16 binary candidate label from metadata row")
