"""只读归一化 event_review 标真条目（schema2 bindings）。

不写回 collector。
"""

from __future__ import annotations

import re
from typing import Any


def normalize_box_token(raw: Any) -> str:
    t = str(raw or "").strip()
    if not t:
        return ""
    if t.startswith("Box_"):
        return t
    if ":" in t:
        return f"Box_{t.split(':')[-1]}"
    return f"Box_{t}" if t.isdigit() else t


def _norm_track(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _tokens_and_tracks_from_entry(entry: dict[str, Any]) -> tuple[list[str], list[str]]:
    """返回 (tokens, track_ids)；按 binding 对齐。"""
    pairs: list[tuple[str, str]] = []
    bindings = entry.get("bindings")
    if isinstance(bindings, list) and bindings:
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            track = _norm_track(binding.get("person_track_id"))
            toks: list[Any] = []
            for key in ("confirmed_box_tokens", "box_tokens"):
                v = binding.get(key)
                if isinstance(v, list) and v:
                    toks = v
                    break
            for item in toks:
                tok = normalize_box_token(item)
                if tok:
                    pairs.append((tok, track))
    if not pairs:
        raw: list[Any] = []
        for key in ("confirmed_box_tokens", "box_tokens"):
            v = entry.get(key)
            if isinstance(v, list) and v:
                raw = v
                break
        for item in raw:
            tok = normalize_box_token(item)
            if tok:
                pairs.append((tok, ""))

    tokens: list[str] = []
    tracks: list[str] = []
    seen: set[tuple[str, str]] = set()
    for tok, track in pairs:
        key = (tok, track)
        if key in seen:
            continue
        seen.add(key)
        tokens.append(tok)
        tracks.append(track)
    return tokens, tracks


def normalize_verified_true(verified_true: list[Any] | None) -> list[dict[str, Any]]:
    """扁平化 token/track；输入不被修改。"""
    out: list[dict[str, Any]] = []
    for entry in verified_true or []:
        if not isinstance(entry, dict):
            continue
        tokens, tracks = _tokens_and_tracks_from_entry(entry)
        if not tokens:
            continue
        norm = dict(entry)
        norm["box_tokens"] = tokens
        norm["confirmed_box_tokens"] = tokens
        norm["person_track_ids"] = tracks
        out.append(norm)
    return out


def camera_review_aliases(camera_slug: str) -> list[str]:
    """json 相机名 ↔ review 目录名：1-6-2-(4) ↔ 1-6-2-_4。"""
    cam = str(camera_slug or "").strip()
    if not cam:
        return []
    out = [cam]
    # 1-6-2-(4) -> 1-6-2-_4
    out.append(cam.replace("-(", "-_").replace(")", ""))
    # 1-6-2-_4 -> 1-6-2-(4)
    out.append(re.sub(r"-_(\d+)$", r"-(\1)", cam))
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq
