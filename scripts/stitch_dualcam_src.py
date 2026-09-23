#!/usr/bin/env python3
"""将两路独立 mp4 拼成 dualcam 用的 src.mp4：左=144、右=24，各 1280×720，总长取较短一路。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _duration_sec(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_L = ROOT / "data" / "test-144.mp4"
DEFAULT_R = ROOT / "data" / "test-24.mp4"
DEFAULT_OUT = ROOT / "output" / "dualcam" / "src.mp4"
HALF_W, HALF_H = 1280, 720


def main() -> int:
    ap = argparse.ArgumentParser(description="144/24 → 2560×720 src.mp4（-shortest）")
    ap.add_argument("--left", type=Path, default=DEFAULT_L, help="左路（144）")
    ap.add_argument("--right", type=Path, default=DEFAULT_R, help="右路（24）")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if not args.left.is_file() or not args.right.is_file():
        print("缺少输入视频", file=sys.stderr)
        return 1
    if shutil.which("ffmpeg") is None:
        print("需要 ffmpeg", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dur = min(_duration_sec(args.left), _duration_sec(args.right))
    filt = (
        f"[0:v]scale={HALF_W}:{HALF_H}:flags=lanczos[left];"
        f"[1:v]scale={HALF_W}:{HALF_H}:flags=lanczos[right];"
        f"[left][right]hstack=inputs=2[vout]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(args.left),
        "-i", str(args.right),
        "-filter_complex", filt,
        "-map", "[vout]",
        "-an",
        "-t", f"{dur:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(args.out),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"wrote {args.out} ({HALF_W * 2}×{HALF_H}, duration={dur:.2f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
