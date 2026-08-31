"""Event Worker-2（dualcam-exp）：双路 3D contact_slots，按 aisle_id 分片。"""

import asyncio
import os
import signal

from core.config import load_app_config
from services.callback_reporter import CollisionCallbackReporter
from services.event_engine.dualcam_worker import DualcamRedisWorker
from services.event_engine.sharding import shard_label


async def _run():
    app_config = load_app_config()

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

    worker = DualcamRedisWorker(app_config, callback_reporter=reporter)
    await worker.start()
    from services.pose_bus import POSE_STREAM_GROUP, pose_delivery_mode

    instance_id = (
        os.environ.get("EVENT_WORKER_INSTANCE_ID", "").strip()
        or os.environ.get("HOSTNAME", "")
    )
    delivery = pose_delivery_mode()
    if delivery == "stream":
        print(
            f"ℹ️ Event worker-2 已启动 delivery=stream dualcam-3d "
            f"streams={worker._owned_stream_keys} group={POSE_STREAM_GROUP} "
            f"consumer={worker._consumer_name} ({shard_label()}) "
            f"id={instance_id or 'local'} callbacks={'on' if enable_cb else 'off'}"
        )
    else:
        print(
            f"ℹ️ Event worker-2 已启动 delivery=pubsub dualcam-3d ({shard_label()}) "
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
    print("ℹ️ Event worker-2 已停止")


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
