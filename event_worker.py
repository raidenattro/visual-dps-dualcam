"""全局事件 Worker：本仓只跑双路 3D DualcamRedisWorker，不再启动 2D CollisionProcessor。"""

from event_worker_2 import main

if __name__ == "__main__":
    main()
