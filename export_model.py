r"""Export the trained DRNet checkpoint to a fixed-shape ONNX model.

Run from the DEAD RECKONING MODEL repository root:

    .venv\Scripts\python.exe export_model.py

The exported model accepts one window at a time:
    imu_window: (1, 6, 40) float32
    v0:         (1,)      float32

and returns:
    disp_and_vend: (1, 3) float32  -- dx, dy, v_end
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from training.train_model import DRNet


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "preprocessing" / "output" / "models" / "dr_model.pt"
FALLBACK_CHECKPOINT = ROOT / "preprocessing" / "output" / "dr_model.pt"
TEST_SPLIT = ROOT / "preprocessing" / "output" / "test_splits" / "Vta1a_test_split.npz"
FALLBACK_TEST_SPLIT = ROOT / "preprocessing" / "output" / "Vta1a_test_split.npz"
ONNX_PATH = ROOT / "preprocessing" / "output" / "models" / "dr_model.onnx"


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    choices = " or ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find required file: {choices}")


def load_model(checkpoint_path: Path) -> DRNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = DRNet(in_channels=6)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main() -> int:
    checkpoint_path = first_existing(CHECKPOINT, FALLBACK_CHECKPOINT)
    test_split_path = first_existing(TEST_SPLIT, FALLBACK_TEST_SPLIT)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint_path)
    test = np.load(test_split_path)
    if test["X"].ndim != 3 or test["X"].shape[1:] != (40, 6):
        raise ValueError(f"Expected test X shape (N, 40, 6), got {test['X'].shape}")

    # Training stores windows as (N, T, C); DRNet consumes (N, C, T).
    imu_window = torch.from_numpy(test["X"][0:1].astype(np.float32)).permute(0, 2, 1)
    v0 = torch.from_numpy(test["v_start"][0:1].astype(np.float32))

    with torch.no_grad():
        torch_output = model(imu_window, v0).cpu().numpy()

    torch.onnx.export(
        model,
        (imu_window, v0),
        str(ONNX_PATH),
        input_names=["imu_window", "v0"],
        output_names=["disp_and_vend"],
        opset_version=17,
        # No dynamic_axes: this is intentionally fixed to batch=1 and T=40.
        dynamo=False,
    )

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_output = session.run(
        ["disp_and_vend"],
        {
            "imu_window": imu_window.numpy(),
            "v0": v0.numpy(),
        },
    )[0]
    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    size_kb = ONNX_PATH.stat().st_size / 1024

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test window: {test_split_path}")
    print(f"PyTorch output: {torch_output.tolist()}")
    print(f"ONNX output: {onnx_output.tolist()}")
    print(f"Max absolute parity difference: {max_abs_diff:.9g}")
    if max_abs_diff >= 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max difference {max_abs_diff:.9g} >= 1e-4")
    print(f"ONNX model: {ONNX_PATH}")
    print(f"ONNX file size: {size_kb:.1f} KB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
