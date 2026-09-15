"""Event Worker（Legacy）：单路 2D 碰撞，消费 event-workers-legacy 组。"""

import asyncio
import os
import signal

from core.config import load_app_config
from services.callback_reporter import CollisionCallbackReporter
from services.event_engine.sharding import shard_label
from services.event_engine.worker import EventRedisWorker
from services.pipeline_log import configure_process_logging


async def _run():
    app_config = load_app_config()
    log_role = os.environ.get("EVENT_WORKER_CONSUMER_NAME", "legacy-a").strip() or "legacy-a"
    configure_process_logging(role=log_role, app_config=app_config)

    enable_cb = os.environ.get("EVENT_WORKER_ENABLE_CALLBACKS", "1").strip() not in (
        "0",
        "false",
        "False",
        "no",
    )
    reporter = None
    if enable_cb:
        reporter = CollisionCallbackReporter(app_config.get("reporting", {}))
        await reporter.start()

    worker = EventRedisWorker(app_config, callback_reporter=reporter)
    await worker.start()
    from services.pose_bus import POSE_STREAM_GROUP, pose_delivery_mode

    instance_id = (
        os.environ.get("EVENT_WORKER_INSTANCE_ID", "").strip()
        or os.environ.get("HOSTNAME", "")
    )
    delivery = pose_delivery_mode()
    if delivery == "stream":
        print(
            f"ℹ️ Event worker 已启动 delivery=stream legacy-2d "
            f"streams={worker._owned_stream_keys} group={POSE_STREAM_GROUP} "
            f"consumer={worker._consumer_name} ({shard_label()}) "
            f"id={instance_id or 'local'} callbacks={'on' if enable_cb else 'off'}"
        )
    else:
        print(
            f"ℹ️ Event worker 已启动 delivery=pubsub legacy-2d ({shard_label()}) "
            f"id={instance_id or 'local'}"
        )

    stopping = False

    def _stop(*_args):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while not stopping:
        await asyncio.sleep(1)

    await worker.stop()
    if reporter is not None:
        await reporter.stop()
    print("ℹ️ Event worker legacy 已停止")


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
