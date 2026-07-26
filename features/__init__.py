"""特征提取器注册（后续接入 collector skeleton_features / 自研加速度等）。"""

FEATURE_CATALOG = [
    "ankle_max_speed_norm",
    "ankle_max_speed",
    "wrist_speed_norm",
    "torso_speed_norm",
    "arm_torso_angle_max",
    "elbow_angle_mean",
    "wrist_elevation_angle_max",
    "shoulder_hip_knee_angle_min",
    # 规划中：加速度与分向速度
    "ankle_accel_norm",
    "wrist_accel_norm",
]
