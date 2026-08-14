#!/usr/bin/env python3
"""将 action_gate HistGBDT 导出为 ONNX（skl2onnx + patch + zipmap=False）并做数值对齐。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experts.skl2onnx_histgbdt_patch import apply_histgbdt_skl2onnx_patch


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_file():
        p = ROOT / path
    if not p.is_file():
        raise FileNotFoundError(path)
    return p


def export_onnx(
    joblib_path: Path,
    out_path: Path,
    *,
    n_samples: int,
    seed: int,
) -> dict:
    apply_histgbdt_skl2onnx_patch()

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort

    clf = joblib.load(joblib_path)
    n_feat = int(clf.n_features_in_)
    opts = {type(clf): {"zipmap": False}}

    t0 = time.perf_counter()
    onnx_model = convert_sklearn(
        clf,
        initial_types=[("X", FloatTensorType([None, n_feat]))],
        options=opts,
    )
    convert_ms = (time.perf_counter() - t0) * 1000.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onnx_model.SerializeToString())

    rng = np.random.default_rng(seed)
    x64 = rng.standard_normal((n_samples, n_feat), dtype=np.float64)
    x32 = x64.astype(np.float32)
    sk_prob = clf.predict_proba(x64)[:, 1]

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ort_prob = sess.run(None, {inp_name: x32})[1][:, 1]
    diff = np.abs(sk_prob - ort_prob)

    # 微基准（单条）
    t0 = time.perf_counter()
    for _ in range(500):
        clf.predict_proba(x64[:1])[:, 1]
    sklearn_1_ms = (time.perf_counter() - t0) * 1000.0 / 500.0

    t0 = time.perf_counter()
    for _ in range(500):
        sess.run(None, {inp_name: x32[:1]})[1][:, 1]
    ort_1_ms = (time.perf_counter() - t0) * 1000.0 / 500.0

    report = {
        "joblib_path": str(joblib_path.relative_to(ROOT) if joblib_path.is_relative_to(ROOT) else joblib_path),
        "onnx_path": str(out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path),
        "n_features": n_feat,
        "onnx_bytes": out_path.stat().st_size,
        "convert_ms": round(convert_ms, 1),
        "parity": {
            "n_samples": n_samples,
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
        },
        "bench_ms_per_call": {
            "sklearn_1x": round(sklearn_1_ms, 4),
            "onnxruntime_1x": round(ort_1_ms, 4),
            "speedup_x": round(sklearn_1_ms / max(ort_1_ms, 1e-9), 1),
        },
        "zipmap": False,
        "skl2onnx_patch": "histgbdt_missing_bool_to_int",
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 action_gate ONNX 并验证对齐")
    ap.add_argument(
        "--joblib",
        default=str(ROOT / "output/train/action_gate_v1/model.joblib"),
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "output/train/action_gate_v1/model.onnx"),
    )
    ap.add_argument("--report", default="", help="对齐报告 JSON 路径")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    joblib_path = _resolve(args.joblib)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    report = export_onnx(joblib_path, out_path, n_samples=args.samples, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    report_path = Path(args.report) if args.report else out_path.with_suffix(".onnx.json")
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] ONNX -> {out_path}")
    print(f"[ok] report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
