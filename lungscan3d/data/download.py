"""Data download and synthetic data generation utilities."""

import logging
import urllib.request
import zipfile
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from lungscan3d.utils.dvc import dvc_pull
from lungscan3d.utils.paths import ensure_dir

LOGGER = logging.getLogger(__name__)

LUNA16_PART1_RECORD_URL = "https://zenodo.org/records/3723295/files"
LUNA16_PART2_RECORD_URL = "https://zenodo.org/records/4121926/files"
LUNA16_METADATA_FILES = ("annotations.csv", "candidates.csv")
LUNA16_SUBSET_FILES = {index: f"subset{index}.zip" for index in range(10)}


def generate_synthetic_dataset(
    output_dir: str | Path,
    num_samples: int,
    patch_size: tuple[int, int, int],
    positive_fraction: float,
    seed: int,
) -> None:
    """Generate a small synthetic 3D nodule-like dataset.

    Positive samples contain a bright spherical blob near the center. Negative samples
    contain only noise and weak low-frequency background. This dataset is intended for
    CI, smoke tests, and assignment validation, not for medical conclusions.

    Args:
    ----
        output_dir: Directory where ``volumes.npy`` and ``labels.npy`` are written.
        num_samples: Number of generated examples.
        patch_size: Patch shape in ``(D, H, W)`` order.
        positive_fraction: Share of positive examples.
        seed: Random seed for reproducibility.

    """
    LOGGER.info(
        "Generating synthetic dataset: output_dir=%s, num_samples=%s",
        output_dir,
        num_samples,
    )
    rng = np.random.default_rng(seed)
    output_path = ensure_dir(output_dir)
    volumes = rng.normal(loc=0.0, scale=0.18, size=(num_samples, 1, *patch_size)).astype(np.float32)
    labels = (rng.random(num_samples) < positive_fraction).astype(np.float32)

    depth, height, width = patch_size
    z_grid, y_grid, x_grid = np.ogrid[:depth, :height, :width]
    center = np.array([depth, height, width], dtype=np.float32) / 2.0

    iterator = tqdm(
        enumerate(labels),
        total=len(labels),
        desc="Generating synthetic positives",
        unit="sample",
        leave=False,
    )
    for sample_index, label in iterator:
        if label < 0.5:
            continue
        jitter = rng.normal(loc=0.0, scale=2.0, size=3)
        radius = rng.uniform(3.0, 5.0)
        distance = (
            (z_grid - center[0] - jitter[0]) ** 2
            + (y_grid - center[1] - jitter[1]) ** 2
            + (x_grid - center[2] - jitter[2]) ** 2
        )
        blob = np.exp(-distance / (2.0 * radius**2)).astype(np.float32)
        volumes[sample_index, 0] += 1.2 * blob

    np.save(output_path / "volumes.npy", volumes)
    np.save(output_path / "labels.npy", labels)

    examples_dir = ensure_dir(Path(output_path).parents[0] / "examples")
    np.save(examples_dir / "sample_patch.npy", volumes[int(labels.argmax())])
    LOGGER.info(
        "Synthetic dataset saved: volumes=%s, labels=%s, example=%s",
        output_path / "volumes.npy",
        output_path / "labels.npy",
        examples_dir / "sample_patch.npy",
    )


def parse_subset_selection(
    subsets: str | int | Sequence[int] | None, max_subsets: int | None
) -> list[int]:
    """Parse requested LUNA16 subset ids.

    Args:
    ----
        subsets: Comma-separated subset ids, an integer id, or a list like ``[0, 1, 2]``.
            ``None`` means use ``max_subsets`` or all subsets.
        max_subsets: Optional number of first subsets to download.

    Returns:
    -------
        Sorted list of subset ids.

    Raises:
    ------
        ValueError: If subset ids are invalid.

    """
    if isinstance(subsets, str) and subsets:
        subset_ids = sorted({int(value.strip()) for value in subsets.split(",") if value.strip()})
    elif isinstance(subsets, int):
        subset_ids = [int(subsets)]
    elif subsets:
        subset_ids = sorted({int(value) for value in subsets})
    elif max_subsets is not None:
        subset_ids = list(range(int(max_subsets)))
    else:
        subset_ids = list(range(10))
    invalid_ids = [subset_id for subset_id in subset_ids if subset_id < 0 or subset_id > 9]
    if invalid_ids:
        raise ValueError(f"LUNA16 subset ids must be in [0, 9], got: {invalid_ids}")
    return subset_ids


def _zenodo_url(filename: str) -> str:
    """Build a Zenodo direct download URL for a LUNA16 file.

    Args:
    ----
        filename: LUNA16 file name on Zenodo.

    Returns:
    -------
        Direct download URL.

    """
    base_url = (
        LUNA16_PART2_RECORD_URL
        if filename in {"subset7.zip", "subset8.zip", "subset9.zip"}
        else LUNA16_PART1_RECORD_URL
    )
    return f"{base_url}/{filename}?download=1"


def _download_file(url: str, target_path: Path, overwrite: bool) -> None:
    """Download a file unless it already exists.

    Args:
    ----
        url: Source URL.
        target_path: Local destination path.
        overwrite: Whether to redownload existing files.

    """
    if target_path.exists() and not overwrite:
        LOGGER.info("File already exists, skipping download: %s", target_path)
        return
    ensure_dir(target_path.parent)
    LOGGER.info("Downloading %s -> %s", url, target_path)
    with urllib.request.urlopen(url) as response, target_path.open("wb") as output_file:
        total = int(response.headers.get("Content-Length") or 0)
        with tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=target_path.name,
        ) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
                progress.update(len(chunk))
    LOGGER.info("Downloaded %s (%.2f MB)", target_path, target_path.stat().st_size / 1024 / 1024)


def _extract_zip(zip_path: Path, output_dir: Path, overwrite: bool) -> None:
    """Extract a zip archive to a directory.

    Args:
    ----
        zip_path: Archive path.
        output_dir: Extraction directory.
        overwrite: Whether to extract again when output directory already exists.

    """
    subset_dir = output_dir / zip_path.stem
    if subset_dir.exists() and any(subset_dir.iterdir()) and not overwrite:
        LOGGER.info("Archive already extracted, skipping: %s", subset_dir)
        return
    LOGGER.info("Extracting %s -> %s", zip_path, output_dir)
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in tqdm(members, desc=f"Extracting {zip_path.name}", unit="file"):
            archive.extract(member, output_dir)
    LOGGER.info("Extracted %s", zip_path.name)


def download_luna16_dataset(
    raw_dir: str | Path,
    subsets: str | None = None,
    max_subsets: int | None = None,
    include_metadata: bool = True,
    extract: bool = True,
    keep_archives: bool = True,
    overwrite: bool = False,
) -> None:
    """Download LUNA16 metadata and selected subset archives from Zenodo.

    The full LUNA16 image data is large. For development it is usually enough to start
    with one or two subsets, for example ``subsets="0"`` or ``max_subsets=2``.

    Args:
    ----
        raw_dir: Destination directory, normally ``data/raw/luna16``.
        subsets: Subset ids to download, for example ``"0,1"`` or ``[0, 1]``.
        max_subsets: Optional number of first subsets to download when ``subsets`` is
            not provided.
        include_metadata: Whether to download ``annotations.csv`` and ``candidates.csv``.
        extract: Whether to unzip downloaded subset archives.
        keep_archives: Whether to keep ``subset*.zip`` files after extraction.
        overwrite: Whether to redownload existing files and re-extract archives.

    """
    output_dir = ensure_dir(raw_dir)
    subset_ids = parse_subset_selection(subsets=subsets, max_subsets=max_subsets)
    LOGGER.info("Preparing LUNA16 download into %s", output_dir)
    LOGGER.info("Selected subsets: %s", subset_ids)

    if include_metadata:
        for filename in tqdm(LUNA16_METADATA_FILES, desc="LUNA16 metadata", unit="file"):
            _download_file(_zenodo_url(filename), output_dir / filename, overwrite=overwrite)

    for subset_id in tqdm(subset_ids, desc="LUNA16 subsets", unit="subset"):
        filename = LUNA16_SUBSET_FILES[subset_id]
        archive_path = output_dir / filename
        _download_file(_zenodo_url(filename), archive_path, overwrite=overwrite)
        if extract:
            _extract_zip(archive_path, output_dir=output_dir, overwrite=overwrite)
            if not keep_archives:
                LOGGER.info("Removing archive after extraction: %s", archive_path)
                archive_path.unlink(missing_ok=True)

    LOGGER.info("LUNA16 download step completed")


def download_data(config: Any) -> None:
    """Ensure that configured data exists locally.

    Args:
    ----
        config: Hydra configuration object with ``data`` and ``paths`` sections.

    """
    data_name = str(config.data.name)
    LOGGER.info("Ensuring dataset is available: %s", data_name)
    if data_name == "synthetic":
        processed_dir = Path(config.paths.processed_dir) / "synthetic"
        if not (processed_dir / "volumes.npy").exists():
            generate_synthetic_dataset(
                output_dir=processed_dir,
                num_samples=int(config.data.num_samples),
                patch_size=tuple(int(value) for value in config.data.patch_size),
                positive_fraction=float(config.data.positive_fraction),
                seed=int(config.seed),
            )
        else:
            LOGGER.info("Synthetic dataset already exists: %s", processed_dir)
        return

    dvc_target = getattr(config.data, "dvc_target", None)
    if dvc_pull(dvc_target):
        LOGGER.info("Dataset restored via DVC target: %s", dvc_target)
        return

    if data_name == "luna16" and bool(getattr(config.data, "allow_internet_download", False)):
        download_luna16_dataset(
            raw_dir=Path(config.data.raw_dir),
            subsets=getattr(config.data, "download_subsets", None),
            max_subsets=getattr(config.data, "download_max_subsets", None),
            include_metadata=bool(getattr(config.data, "download_metadata", True)),
            extract=bool(getattr(config.data, "extract_archives", True)),
            keep_archives=bool(getattr(config.data, "keep_archives", True)),
            overwrite=bool(getattr(config.data, "overwrite_downloads", False)),
        )
        return

    raise FileNotFoundError(
        f"Dataset '{data_name}' is not bundled and DVC pull did not restore it. "
        "Either run `lungscan3d download-luna16 data=luna16 data.download_subsets=0`, "
        "prepare LUNA16 raw files "
        "manually as described in README.md, or use data=synthetic for a smoke test."
    )
