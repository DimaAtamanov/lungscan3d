"""Triton inference client."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

LOGGER = logging.getLogger(__name__)


def call_triton(config: Any, input: str) -> None:
    """Call Triton HTTP endpoint for a preprocessed patch.

    Args:
    ----
        config: Hydra configuration object.
        input: Path to NumPy patch.
        url: Triton HTTP endpoint URL.

    """
    LOGGER.info("Calling Triton server: url=%s, input=%s", config.triton_client, input)

    patch = np.load(Path(input)).astype(np.float32)
    if patch.ndim == 4:
        patch = patch[None, ...]
    client = httpclient.InferenceServerClient(url=config.triton_client)
    infer_input = httpclient.InferInput(
        str(config.infer.input_name),
        patch.shape,
        np_to_triton_dtype(patch.dtype),
    )
    infer_input.set_data_from_numpy(patch)
    infer_output = httpclient.InferRequestedOutput(str(config.infer.output_name))
    response = client.infer("lungscan3d", inputs=[infer_input], outputs=[infer_output])
    logits = response.as_numpy(str(config.infer.output_name))
    LOGGER.info("Triton inference finished")
    print(json.dumps({"logit": logits.reshape(-1).tolist()}))
