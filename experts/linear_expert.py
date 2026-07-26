"""标准化特征 + logistic 权重 → pick_score ∈ (0,1)。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class LinearPickExpert:
    name = "linear_expert"

    def __init__(self, cfg: dict[str, Any]):
        model_path = cfg.get("model_path")
        if model_path:
            path = Path(model_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            model = json.loads(path.read_text(encoding="utf-8"))
        else:
            model = cfg.get("model") or {}
        self.feature_keys = list(model.get("feature_keys") or [])
        self.mean = [float(x) for x in (model.get("scaler_mean") or [])]
        self.scale = [float(x) for x in (model.get("scaler_scale") or [])]
        self.coef = [float(x) for x in (model.get("coef") or [])]
        self.intercept = float(model.get("intercept") or 0.0)
        if not (len(self.feature_keys) == len(self.coef) == len(self.mean) == len(self.scale)):
            raise ValueError("linear_expert model 维度不一致")

    def reset(self) -> None:
        return

    def score(self, row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        z = self.intercept
        contrib: dict[str, float] = {}
        missing = False
        for i, key in enumerate(self.feature_keys):
            raw = row.get(key)
            if raw is None:
                missing = True
                break
            try:
                x = float(raw)
            except (TypeError, ValueError):
                missing = True
                break
            scale = self.scale[i] if self.scale[i] else 1.0
            xs = (x - self.mean[i]) / scale
            c = self.coef[i] * xs
            contrib[key] = round(c, 4)
            z += c

        if missing:
            # 特征不全：fail-open（不挡），分偏低但不过激
            return 0.55, {"missing": True, "contrib": {}}

        # sigmoid
        if z >= 0:
            prob = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            prob = ez / (1.0 + ez)
        top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        return float(prob), {"z": round(z, 4), "contrib": contrib, "top": top}
