"""请求处理和业务服务共享的轻量运行态模块。

这里保留的是当前视频会话需要的最小状态，方便在上传、标注、推理
之间传递，不再把散落的全局变量放在主入口里。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppState:
    """当前视频会话的可变状态。"""
    video_path: str = ""
    source_type: str = "file"
    source_url: str = ""
    is_inferencing: bool = False
    upload_id: int = 0
    upload_tag: str = ""
    json_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


STATE = AppState()
