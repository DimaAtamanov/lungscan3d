import numpy as np

from lungscan3d.data.preprocessing import (
    crop_or_pad_patch,
    normalize_hu,
    world_to_voxel_zyx,
)


def test_normalize_hu_range():
    volume = np.array([-1200.0, -1000.0, -300.0, 400.0, 1000.0], dtype=np.float32)
    normalized = normalize_hu(volume, clip_min=-1000.0, clip_max=400.0)

    assert normalized.min() >= -1.0
    assert normalized.max() <= 1.0


def test_crop_or_pad_patch_shape():
    volume = np.ones((5, 5, 5), dtype=np.float32)
    patch = crop_or_pad_patch(volume, center_zyx=(0, 0, 0), patch_size=(4, 4, 4))

    assert patch.shape == (4, 4, 4)
    assert patch.sum() > 0


def test_world_to_voxel_zyx_uses_luna_coordinate_order():
    voxel = world_to_voxel_zyx(
        coord_xyz=(12.0, 24.0, 36.0),
        origin_xyz=(10.0, 20.0, 30.0),
        spacing_zyx=(3.0, 2.0, 1.0),
    )

    assert voxel == (2, 2, 2)
