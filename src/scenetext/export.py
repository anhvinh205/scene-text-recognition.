"""Export the recogniser to ONNX and verify numerical parity with PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scenetext.recognition.checkpoint import load_recognizer


def export_recognizer_onnx(ckpt: str, out: str, verify: bool = True, opset: int = 17) -> Path:
    model, _, (h, w) = load_recognizer(ckpt, "cpu")
    dummy = torch.randn(1, 1, h, w)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["log_probs"],
        dynamic_axes={"image": {0: "batch"}, "log_probs": {1: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    if verify:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        x = torch.randn(3, 1, h, w)
        ref = model(x).detach().numpy()
        got = sess.run(None, {"image": x.numpy()})[0]
        np.testing.assert_allclose(ref, got, rtol=1e-3, atol=1e-4)
    return out_path
