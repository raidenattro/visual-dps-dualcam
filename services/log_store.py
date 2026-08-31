"""SQLite 操作审计与业务事件日志。"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any

DEFAULT_DB_PATH = os.environ.get("LOG_DB_PATH", "localdata/logs/visual_dps.db")
_RETENTION_DAYS = int(os.environ.get("LOG_RETENTION_DAYS", "90"))

_lock = threading.Lock()


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_log_db(db_path: str = DEFAULT_DB_PATH) -> None:
    with _lock:
        conn = _connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    result TEXT NOT NULL,
                    detail TEXT,
                    ip TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_logs(ts DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_logs(actor);

                CREATE TABLE IF NOT EXISTS event_logs (
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    camera_id TEXT,
                    severity TEXT NOT NULL DEFAULT 'info',
                    summary TEXT,
                    detail TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_event_ts ON event_logs(ts DESC);
                CREATE INDEX IF NOT EXISTS idx_event_type ON event_logs(event_type);
                CREATE INDEX IF NOT EXISTS idx_event_camera ON event_logs(camera_id);
                """
            )
            conn.commit()
        finally:
            conn.close()


def _prune_old(conn: sqlite3.Connection) -> None:
    if _RETENTION_DAYS <= 0:
        return
    cutoff = time.time() - _RETENTION_DAYS * 86400
    conn.execute("DELETE FROM audit_logs WHERE ts < ?", (cutoff,))
    conn.execute("DELETE FROM event_logs WHERE ts < ?", (cutoff,))


def insert_audit(
    *,
    actor: str,
    action: str,
    resource_type: str = "",
    resource_id: str = "",
    result: str = "success",
    detail: dict | None = None,
    ip: str = "",
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    row_id = uuid.uuid4().hex
    now = time.time()
    detail_json = json.dumps(detail or {}, ensure_ascii=False)
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO audit_logs (id, ts, actor, action, resource_type, resource_id, result, detail, ip)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (row_id, now, actor, action, resource_type, resource_id, result, detail_json, ip),
            )
            _prune_old(conn)
            conn.commit()
        finally:
            conn.close()
    return row_id


def insert_event(
    *,
    event_type: str,
    camera_id: str = "",
    severity: str = "info",
    summary: str = "",
    detail: dict | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    row_id = uuid.uuid4().hex
    now = time.time()
    detail_json = json.dumps(detail or {}, ensure_ascii=False)
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO event_logs (id, ts, event_type, camera_id, severity, summary, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (row_id, now, event_type, camera_id, severity, summary, detail_json),
            )
            _prune_old(conn)
            conn.commit()
        finally:
            conn.close()
    return row_id


_AUDIT_SORT_COLUMNS = frozenset({"ts", "actor", "action", "result", "resource_type", "resource_id"})
_EVENT_SORT_COLUMNS = frozenset({"ts", "event_type", "camera_id", "severity", "summary"})


def _safe_sort_column(column: str, allowed: frozenset[str], default: str = "ts") -> str:
    return column if column in allowed else default


def _sort_direction(order: str) -> str:
    return "ASC" if str(order or "").lower() == "asc" else "DESC"


def _paginate(query: str, count_query: str, params: list, page: int, page_size: int, db_path: str) -> dict:
    page = max(1, page)
    page_size = max(1, min(200, page_size))
    offset = (page - 1) * page_size
    with _lock:
        conn = _connect(db_path)
        try:
            total = conn.execute(count_query, params).fetchone()[0]
            rows = conn.execute(f"{query} LIMIT ? OFFSET ?", [*params, page_size, offset]).fetchall()
        finally:
            conn.close()
    items = [dict(r) for r in rows]
    for item in items:
        if item.get("detail"):
            try:
                item["detail"] = json.loads(item["detail"])
            except json.JSONDecodeError:
                pass
        item["ts_iso"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.get("ts", 0)))
    return {"status": "success", "page": page, "page_size": page_size, "total": total, "items": items}


def query_audit_logs(
    *,
    page: int = 1,
    page_size: int = 50,
    actor: str = "",
    action: str = "",
    result: str = "",
    resource_id: str = "",
    sort_by: str = "ts",
    sort_order: str = "desc",
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    where = ["1=1"]
    params: list[Any] = []
    if actor.strip():
        where.append("actor LIKE ?")
        params.append(f"%{actor.strip()}%")
    if action.strip():
        where.append("action LIKE ?")
        params.append(f"%{action.strip()}%")
    if result.strip():
        where.append("result = ?")
        params.append(result.strip())
    if resource_id.strip():
        where.append("resource_id LIKE ?")
        params.append(f"%{resource_id.strip()}%")
    w = " AND ".join(where)
    sort_col = _safe_sort_column(sort_by, _AUDIT_SORT_COLUMNS)
    sort_dir = _sort_direction(sort_order)
    return _paginate(
        f"SELECT * FROM audit_logs WHERE {w} ORDER BY {sort_col} {sort_dir}",
        f"SELECT COUNT(*) FROM audit_logs WHERE {w}",
        params,
        page,
        page_size,
        db_path,
    )


def query_event_logs(
    *,
    page: int = 1,
    page_size: int = 50,
    event_type: str = "",
    camera_id: str = "",
    severity: str = "",
    summary: str = "",
    sort_by: str = "ts",
    sort_order: str = "desc",
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    where = ["1=1"]
    params: list[Any] = []
    if event_type.strip():
        where.append("event_type LIKE ?")
        params.append(f"%{event_type.strip()}%")
    if camera_id.strip():
        where.append("camera_id LIKE ?")
        params.append(f"%{camera_id.strip()}%")
    if severity.strip():
        where.append("severity = ?")
        params.append(severity.strip())
    if summary.strip():
        where.append("summary LIKE ?")
        params.append(f"%{summary.strip()}%")
    w = " AND ".join(where)
    sort_col = _safe_sort_column(sort_by, _EVENT_SORT_COLUMNS)
    sort_dir = _sort_direction(sort_order)
    return _paginate(
        f"SELECT * FROM event_logs WHERE {w} ORDER BY {sort_col} {sort_dir}",
        f"SELECT COUNT(*) FROM event_logs WHERE {w}",
        params,
        page,
        page_size,
        db_path,
    )
