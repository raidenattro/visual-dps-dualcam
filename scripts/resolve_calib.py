#!/usr/bin/env python3
"""按新的实测尺寸重跑已存标定的相机反解，四角标注原样复用。

拣货面底沿贴地时 base 必须填 0：填错只会让整个场景连同相机竖直平移，
反投影残差一点都不变，画面上看不出来。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.solve_scene import solve as solve_scene  # noqa: E402

CALIB_DIR = ROOT / "output/calib"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=float, help="底沿离地 m，贴地填 0")
    ap.add_argument("--width", type=float, help="面宽 m")
    ap.add_argument("--height", type=float, help="面高 m")
    ap.add_argument("--aisle", type=float, help="巷道净宽 m")
    ap.add_argument("--cam-h", type=float, help="相机高先验 m")
    ap.add_argument("--cam-dist", type=float, help="相机到近端水平距离先验 m")
    ap.add_argument("--slug", help="只处理这个机位")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写回")
    args = ap.parse_args()

    files = sorted(CALIB_DIR.glob("*.json"))
    if args.slug:
        files = [f for f in files if f.stem == args.slug]
    if not files:
        print("没有找到标定文件")
        return

    for path in files:
        calib = json.loads(path.read_text())
        walls = [w for w in calib.get("walls") or [] if len(w.get("quad") or []) == 4]
        if not walls:
            print(f"{path.stem}: 没有完整四角，跳过")
            continue
        if args.aisle is not None:
            calib["aisle"] = args.aisle
        prior = dict(calib.get("prior") or {})
        if args.cam_h is not None:
            prior["camH"] = args.cam_h
        if args.cam_dist is not None:
            prior["camDist"] = args.cam_dist
        if prior:
            calib["prior"] = prior
        for w in walls:
            for key, val in (("base", args.base), ("width", args.width), ("height", args.height)):
                if val is not None:
                    w[key] = val

        img_w, img_h = calib.get("image_size") or [852, 480]
        old = (calib.get("solved") or {}).get("camera") or {}
        res = solve_scene(calib, int(img_w), int(img_h))
        if not res.get("ok"):
            print(f"{path.stem}: 求解失败 {res.get('error')}")
            continue
        cam = res["camera"]
        delta = f" (原 {old['camH']:.3f})" if "camH" in old else ""
        print(
            f"{path.stem}: 残差 {res['resid_px']:.2f}px · FOV {cam['fovH']:.1f} · "
            f"相机高 {cam['camH']:.3f}{delta} · 俯仰 {cam['pitch']:.2f} · 滚转 {cam['roll']:.2f}"
        )
        if args.dry_run:
            continue
        calib["solved"] = res
        path.write_text(json.dumps(calib, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
