from datetime import datetime, timezone
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Mock Java Backend")

# Use the PDF-defined callback path here. You can override at runtime:
# set MOCK_CALLBACK_PATH=/your/pdf/path
MOCK_CALLBACK_PATH = os.getenv("MOCK_CALLBACK_PATH", "/api/pick/finish").strip() or "/api/pick/finish"
LEGACY_CALLBACK_ALIASES = ["/api/pick/finish"]

# Keep recent callback payloads in memory for quick verification.
RECEIVED_EVENTS: list[dict[str, Any]] = []
MAX_EVENTS = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "mock-java-backend"}


async def pick_finish(request: Request):
    payload = await request.json()
    print(f"[MOCK_JAVA][RECV] {_now_iso()} payload={payload}")

    RECEIVED_EVENTS.append({"received_at": _now_iso(), "payload": payload})
    if len(RECEIVED_EVENTS) > MAX_EVENTS:
        del RECEIVED_EVENTS[0 : len(RECEIVED_EVENTS) - MAX_EVENTS]

    # Optional test hook: if payload contains simulateStatus, use it as HTTP status.
    simulate_status = payload.get("simulateStatus")
    if isinstance(simulate_status, int) and simulate_status >= 400:
        return JSONResponse(
            status_code=simulate_status,
            content={
                "success": False,
                "message": "simulated failure",
                "simulateStatus": simulate_status,
                "eventId": payload.get("eventId") or payload.get("event_id"),
                "serverTime": _now_iso(),
            },
        )

    return {
        "success": True,
        "message": "callback accepted",
        "eventId": payload.get("eventId") or payload.get("event_id"),
        "serverTime": _now_iso(),
        "received": payload,
    }


app.add_api_route(MOCK_CALLBACK_PATH, pick_finish, methods=["POST"])

for alias in LEGACY_CALLBACK_ALIASES:
    if alias != MOCK_CALLBACK_PATH:
        app.add_api_route(alias, pick_finish, methods=["POST"])


@app.get("/api/pick/records")
async def pick_records(limit: int = 20):
    limit = max(1, min(limit, 200))
    return {
        "count": len(RECEIVED_EVENTS),
        "items": RECEIVED_EVENTS[-limit:],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
