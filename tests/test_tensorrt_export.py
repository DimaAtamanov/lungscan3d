from pathlib import Path
from types import SimpleNamespace

from lungscan3d.inference.trt_export import _build_trtexec_command


def test_build_trtexec_command_uses_dynamic_shapes():
    config = SimpleNamespace(
        infer=SimpleNamespace(input_name="input"),
        model=SimpleNamespace(in_channels=1),
        data=SimpleNamespace(patch_size=[32, 48, 48]),
        tensorrt=SimpleNamespace(
            min_batch_size=1,
            opt_batch_size=16,
            max_batch_size=64,
            workspace_mb=4096,
            precision="fp16",
            extra_args=[],
        ),
    )

    command = _build_trtexec_command(config, "trtexec", Path("model.onnx"), Path("model.plan"))

    assert "--minShapes=input:1x1x32x48x48" in command
    assert "--optShapes=input:16x1x32x48x48" in command
    assert "--maxShapes=input:64x1x32x48x48" in command
    assert "--fp16" in command
