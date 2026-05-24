from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from lungscan3d.data.datamodule import LungScanDataModule


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_lazy_chunked_datamodule_splits_by_patient(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed" / "luna16"
    chunks_dir = processed_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    np.save(chunks_dir / "volumes_000000.npy", np.zeros((8, 1, 4, 4, 4), dtype=np.float32))
    np.save(chunks_dir / "labels_000000.npy", np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32))
    np.save(processed_dir / "labels.npy", np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32))
    pd.DataFrame(
        {
            "global_index": list(range(8)),
            "chunk_index": [0] * 8,
            "local_index": list(range(8)),
            "volume_path": ["chunks/volumes_000000.npy"] * 8,
            "label_path": ["chunks/labels_000000.npy"] * 8,
            "seriesuid": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    ).to_csv(processed_dir / "manifest.csv", index=False)

    config = _ns(
        seed=42,
        paths=_ns(processed_dir=str(tmp_path / "processed"), splits_dir=str(tmp_path / "splits")),
        data=_ns(
            name="luna16",
            processed_dir=str(processed_dir),
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            split_by_patient=True,
            group_column="seriesuid",
            save_splits=True,
            weighted_sampling=_ns(enabled=False),
            batch_size=2,
            num_workers=0,
        ),
        preprocessing=_ns(
            augment=_ns(
                enabled=False,
                random_flip=False,
                random_rotate90=False,
                gaussian_noise_std=0.0,
                random_shift_voxels=0,
            )
        ),
    )

    datamodule = LungScanDataModule(config)
    datamodule.setup()

    split_summary = (tmp_path / "splits" / "luna16" / "groups.json").read_text(encoding="utf-8")
    assert "p1" in split_summary or "p2" in split_summary

    manifest = pd.read_csv(processed_dir / "manifest.csv")
    train_groups = set(manifest.iloc[datamodule.train_dataset.indices].seriesuid)
    val_groups = set(manifest.iloc[datamodule.val_dataset.indices].seriesuid)
    test_groups = set(manifest.iloc[datamodule.test_dataset.indices].seriesuid)
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
