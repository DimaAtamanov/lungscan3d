"""CT preprocessing utilities."""

import csv
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from lungscan3d.utils.paths import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CtMetadata:
    """Metadata needed to map world coordinates into voxel coordinates.

    Attributes
    ----------
    spacing_zyx:
        Voxel spacing in ``(Z, Y, X)`` millimeters.
    origin_xyz:
        CT origin in LUNA16 world coordinates.

    """

    spacing_zyx: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]


def normalize_hu(volume_hu: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """Clip and normalize CT Hounsfield units to ``[-1, 1]``."""
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
    """Crop a 3D patch and pad with zeros near borders."""
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
    """Convert CT world coordinates to voxel coordinates."""
    spacing_xyz = np.array([spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]], dtype=np.float32)
    voxel_xyz = (
        np.array(coord_xyz, dtype=np.float32) - np.array(origin_xyz, dtype=np.float32)
    ) / spacing_xyz
    voxel_zyx = voxel_xyz[::-1]
    return tuple(int(round(value)) for value in voxel_zyx)


def preprocess(config: Any) -> None:
    """Run preprocessing for configured dataset."""
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
            chunk_size=int(getattr(config.preprocessing, "chunk_size", 512)),
            max_cached_ct_volumes=int(getattr(config.preprocessing, "max_cached_ct_volumes", 1)),
            progress=bool(getattr(config.preprocessing, "progress", True)),
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
    chunk_size: int = 512,
    max_cached_ct_volumes: int = 1,
    progress: bool = True,
) -> None:
    """Preprocess candidate rows into disk-backed chunks suitable for training.

    Instead of materializing every candidate patch in RAM and writing one huge
    ``volumes.npy``, this function keeps only ``chunk_size`` patches in memory,
    flushes them to ``chunks/volumes_XXXXXX.npy``, and records a ``manifest.csv``.
    """
    LOGGER.info("Preprocessing candidates: raw_dir=%s, metadata_csv=%s", raw_dir, metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if max_cached_ct_volumes <= 0:
        raise ValueError("max_cached_ct_volumes must be a positive integer")

    mhd_index = _index_mhd_files(raw_dir)
    if not mhd_index:
        raise FileNotFoundError(f"No .mhd files found under {raw_dir}")
    LOGGER.info("Indexed %d CT volumes", len(mhd_index))

    rows = pd.read_csv(metadata_csv)
    rows = rows.rename(columns={"class": "label"})
    _validate_candidate_columns(rows)
    rows = rows.sort_values("seriesuid", kind="stable").reset_index(drop=True)
    LOGGER.info("Candidate metadata columns: %s", list(rows.columns))
    LOGGER.info("Loaded %d candidate rows", len(rows))
    LOGGER.info("Class distribution: %s", rows["label"].value_counts().to_dict())

    output_path = ensure_dir(output_dir)
    chunks_dir = ensure_dir(output_path / "chunks")
    _remove_stale_chunk_files(chunks_dir)

    labels_path = output_path / "labels.npy"
    manifest_path = output_path / "manifest.csv"
    legacy_volumes_path = output_path / "volumes.npy"
    if legacy_volumes_path.exists():
        LOGGER.warning(
            "Removing legacy monolithic volumes.npy so the streamed format is used: %s",
            legacy_volumes_path,
        )
        legacy_volumes_path.unlink()

    chunk_volumes: list[np.ndarray] = []
    chunk_labels: list[float] = []
    chunk_metadata: list[tuple[str, float]] = []
    all_labels: list[float] = []
    manifest_rows: list[dict[str, Any]] = []
    volume_cache: OrderedDict[str, tuple[np.ndarray, CtMetadata]] = OrderedDict()
    chunk_index = 0

    iterator = rows.itertuples(index=False)
    if progress:
        iterator = tqdm(
            iterator,
            total=len(rows),
            desc="Preprocessing LUNA16 candidates",
            unit="candidate",
        )

    for row in iterator:
        seriesuid = str(row.seriesuid)
        if seriesuid not in mhd_index:
            continue

        label = int(row.label)
        if label is None:
            continue

        ct_array, metadata = _get_cached_normalized_volume(
            seriesuid=seriesuid,
            mhd_path=mhd_index[seriesuid],
            volume_cache=volume_cache,
            clip_min=clip_min,
            clip_max=clip_max,
            max_cached_ct_volumes=max_cached_ct_volumes,
        )
        coord_xyz = (float(row.coordX), float(row.coordY), float(row.coordZ))
        center_zyx = world_to_voxel_zyx(
            coord_xyz=coord_xyz,
            origin_xyz=metadata.origin_xyz,
            spacing_zyx=metadata.spacing_zyx,
        )
        patch = crop_or_pad_patch(ct_array, center_zyx=center_zyx, patch_size=patch_size)
        chunk_volumes.append(patch[None, ...].astype(np.float32))
        chunk_labels.append(float(label))
        chunk_metadata.append((seriesuid, float(label)))
        all_labels.append(float(label))

        if len(chunk_volumes) >= chunk_size:
            manifest_rows.extend(
                _write_preprocessed_chunk(
                    chunks_dir=chunks_dir,
                    chunk_index=chunk_index,
                    volumes=chunk_volumes,
                    labels=chunk_labels,
                    metadata=chunk_metadata,
                )
            )
            LOGGER.info(
                "Wrote preprocessing chunk %06d with %d samples",
                chunk_index,
                len(chunk_volumes),
            )
            chunk_index += 1
            chunk_volumes.clear()
            chunk_labels.clear()
            chunk_metadata.clear()

    if chunk_volumes:
        manifest_rows.extend(
            _write_preprocessed_chunk(
                chunks_dir=chunks_dir,
                chunk_index=chunk_index,
                volumes=chunk_volumes,
                labels=chunk_labels,
                metadata=chunk_metadata,
            )
        )
        LOGGER.info(
            "Wrote preprocessing chunk %06d with %d samples",
            chunk_index,
            len(chunk_volumes),
        )

    if not all_labels:
        raise ValueError("No training samples were produced from the metadata CSV")

    labels_array = np.asarray(all_labels, dtype=np.float32)
    np.save(labels_path, labels_array)
    _write_manifest(manifest_path, manifest_rows)

    LOGGER.info(
        "Preprocessing finished: samples=%d, positives=%d, chunks=%d, output_dir=%s",
        len(labels_array),
        int(labels_array.sum()),
        len({row["chunk_index"] for row in manifest_rows}),
        output_path,
    )


def read_mhd_volume(path: str | Path) -> tuple[np.ndarray, CtMetadata]:
    """Read an MHD CT volume with SimpleITK."""
    import SimpleITK as sitk

    # LOGGER.info("Reading CT volume: %s", path)
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    origin_raw = image.GetOrigin()
    origin_xyz = (float(origin_raw[0]), float(origin_raw[1]), float(origin_raw[2]))
    return array, CtMetadata(spacing_zyx=spacing_zyx, origin_xyz=origin_xyz)


def _get_cached_normalized_volume(
    seriesuid: str,
    mhd_path: Path,
    volume_cache: OrderedDict[str, tuple[np.ndarray, CtMetadata]],
    clip_min: float,
    clip_max: float,
    max_cached_ct_volumes: int,
) -> tuple[np.ndarray, CtMetadata]:
    if seriesuid in volume_cache:
        volume_cache.move_to_end(seriesuid)
        return volume_cache[seriesuid]

    ct_array, metadata = read_mhd_volume(mhd_path)
    ct_array = normalize_hu(ct_array, clip_min=clip_min, clip_max=clip_max)
    volume_cache[seriesuid] = (ct_array, metadata)
    volume_cache.move_to_end(seriesuid)

    while len(volume_cache) > max_cached_ct_volumes:
        evicted_seriesuid, _ = volume_cache.popitem(last=False)
        LOGGER.debug(
            "Evicted normalized CT volume from preprocessing cache: %s",
            evicted_seriesuid,
        )

    return ct_array, metadata


def _write_preprocessed_chunk(
    chunks_dir: Path,
    chunk_index: int,
    volumes: list[np.ndarray],
    labels: list[float],
    metadata: list[tuple[str, float]],
) -> list[dict[str, Any]]:
    volume_path = chunks_dir / f"volumes_{chunk_index:06d}.npy"
    label_path = chunks_dir / f"labels_{chunk_index:06d}.npy"

    volumes_array = np.stack(volumes, axis=0).astype(np.float32)
    labels_array = np.asarray(labels, dtype=np.float32)
    np.save(volume_path, volumes_array)
    np.save(label_path, labels_array)

    rows = []
    for local_index, (seriesuid, label) in enumerate(metadata):
        rows.append(
            {
                "global_index": None,
                "chunk_index": chunk_index,
                "local_index": local_index,
                "volume_path": str(volume_path.relative_to(chunks_dir.parent)),
                "label_path": str(label_path.relative_to(chunks_dir.parent)),
                "seriesuid": seriesuid,
                "label": float(label),
            }
        )
    return rows


def _write_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "global_index",
        "chunk_index",
        "local_index",
        "volume_path",
        "label_path",
        "seriesuid",
        "label",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for global_index, row in enumerate(rows):
            row = dict(row)
            row["global_index"] = global_index
            writer.writerow(row)


def _remove_stale_chunk_files(chunks_dir: Path) -> None:
    for pattern in ("volumes_*.npy", "labels_*.npy"):
        for path in chunks_dir.glob(pattern):
            path.unlink()


def _index_mhd_files(raw_dir: Path) -> dict[str, Path]:
    """Index MHD files by stem/series UID."""
    return {path.stem: path for path in raw_dir.rglob("*.mhd")}


def _validate_candidate_columns(rows: pd.DataFrame) -> None:
    """Validate core metadata columns."""
    required_columns = {"seriesuid", "coordX", "coordY", "coordZ"}
    missing_columns = required_columns.difference(rows.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")
