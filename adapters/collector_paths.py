"""只读访问 visual-dps-data-collector/localdata。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_COLLECTOR_ROOT = Path("/home/hqit/workspace/visual-dps-data-collector")


@dataclass(frozen=True)
class CollectorPaths:
    root: Path
    localdata: Path
    review_dir: Path
    json_dir: Path
    baseline_manifest: Path

    def ensure_readable(self) -> list[str]:
        """返回缺失路径列表；空列表表示可读检查通过。"""
        missing: list[str] = []
        for p in (self.root, self.localdata, self.review_dir, self.json_dir, self.baseline_manifest):
            if not p.exists():
                missing.append(str(p))
        return missing


def _repo_configs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def load_paths(config_path: Path | None = None) -> CollectorPaths:
    cfg_path = config_path or (_repo_configs_dir() / "paths.default.json")
    data: dict[str, Any] = {}
    if cfg_path.is_file():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))

    root = Path(
        os.environ.get("PICK_STATE_COLLECTOR_ROOT")
        or data.get("collector_root")
        or DEFAULT_COLLECTOR_ROOT
    ).expanduser().resolve()
    localdata = root / str(data.get("localdata_rel") or "localdata")
    return CollectorPaths(
        root=root,
        localdata=localdata,
        review_dir=localdata / str(data.get("review_rel") or "review"),
        json_dir=localdata / str(data.get("json_rel") or "json"),
        baseline_manifest=localdata
        / str(data.get("baseline_manifest_rel") or "export/rule-baseline-local-prod-test/_manifest.json"),
    )


def load_baseline_manifest(paths: CollectorPaths | None = None) -> dict[str, Any]:
    p = paths or load_paths()
    return json.loads(p.baseline_manifest.read_text(encoding="utf-8"))
