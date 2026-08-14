"""action_gate ONNX 导出与推理。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experts.action_gate import ActionGate
from experts.skl2onnx_histgbdt_patch import apply_histgbdt_skl2onnx_patch

JOBLIB = ROOT / "output/train/action_gate_v1/model.joblib"
ONNX = ROOT / "output/train/action_gate_v1/model.onnx"


@pytest.fixture(scope="module")
def ensure_onnx_model():
    if not JOBLIB.is_file():
        pytest.skip(f"缺少 {JOBLIB}")
    if not ONNX.is_file():
        from scripts.export_action_gate_onnx import export_onnx

        export_onnx(JOBLIB, ONNX, n_samples=200, seed=0)
    return ONNX


def test_skl2onnx_export_parity(ensure_onnx_model):
    import joblib
    import onnxruntime as ort

    clf = joblib.load(JOBLIB)
    rng = np.random.default_rng(42)
    x64 = rng.standard_normal((300, int(clf.n_features_in_)), dtype=np.float64)
    x32 = x64.astype(np.float32)
    sk = clf.predict_proba(x64)[:, 1]

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    ort_p = sess.run(None, {inp: x32})[1][:, 1]
    assert float(np.max(np.abs(sk - ort_p))) < 1e-5


def test_action_gate_onnx_backend_matches_sklearn(ensure_onnx_model):
    rng = np.random.default_rng(7)
    feat = rng.standard_normal(68, dtype=np.float64)

    sk_gate = ActionGate(
        {
            "enabled": True,
            "backend": "sklearn",
            "threshold": 0.30,
            "model_path": str(JOBLIB.relative_to(ROOT)),
        }
    )
    onnx_gate = ActionGate(
        {
            "enabled": True,
            "backend": "onnx",
            "threshold": 0.30,
            "model_path": str(JOBLIB.relative_to(ROOT)),
            "onnx_path": str(ONNX.relative_to(ROOT)),
        }
    )
    p_sk = sk_gate.score(feat)
    p_on = onnx_gate.score(feat)
    assert abs(p_sk - p_on) < 1e-5
    ok_sk, _ = sk_gate.allow(feat)
    ok_on, _ = onnx_gate.allow(feat)
    assert ok_sk == ok_on


def test_export_script_idempotent():
    apply_histgbdt_skl2onnx_patch()
    if not JOBLIB.is_file():
        pytest.skip(f"缺少 {JOBLIB}")
    from scripts.export_action_gate_onnx import export_onnx

    tmp = ROOT / "output/train/action_gate_v1/model.onnx"
    report = export_onnx(JOBLIB, tmp, n_samples=50, seed=1)
    assert report["parity"]["max_abs_diff"] < 1e-5
    assert tmp.is_file()
