/** 将 API/异常信息转为面向终端用户的简短说明 */
export function formatUserError(message) {
  if (message == null || message === '') return '操作失败，请稍后重试';
  const raw = String(message).trim();
  const lower = raw.toLowerCase();

  const exact = {
    'camera not found': '未找到该摄像头',
    'camera id is required': '摄像头信息不完整',
    'camera url is required': '请填写视频流地址',
    'url is required': '请填写视频流地址',
    'annotation not found': '尚未配置标注，请先在标注页完成设置',
    'last frame not found': '暂无预览画面，请先抓取一帧',
    'failed to open camera stream': '无法连接视频流，请检查地址与网络',
    'failed to read frame from camera': '无法从摄像头读取画面',
    'no video': '请先上传视频',
  };
  if (exact[lower]) return exact[lower];

  if (/路径已存在/.test(raw)) return '该通道编号已被使用，请换一个';
  if (/path 不能为空|通道.*空/.test(raw)) return '请填写通道编号';
  if (/path 仅支持|通道编号/.test(raw)) return '通道编号仅支持字母、数字、下划线、中划线';
  if (/pull_url|上游/.test(raw) && /填/.test(raw)) return '请填写上游视频流地址';
  if (/device|设备/.test(raw) && /填/.test(raw)) return '请填写本地摄像头设备路径';
  if (/外部.*rtsp|播放地址/.test(raw)) return '请填写完整的视频流地址';
  if (/启动推理容器|停止推理容器|docker|container|HOST_PROJECT|mediamtx|\.sh|traceback|sdk/i.test(raw)) {
    return '智能检测服务暂时不可用，请稍后重试';
  }
  if (/failed to fetch|network|无法连接/i.test(raw)) return '无法连接服务器，请检查网络后重试';

  return raw;
}

/** 推理状态附言（不展示容器名、日志等实现细节） */
export function formatInferenceMessage(message) {
  if (!message) return '';
  const raw = String(message).trim();
  const mapped = {
    推理运行中: '运行中',
    容器已创建: '正在启动…',
    容器异常退出: '检测已异常停止，请重试',
    已手动停止: '已停止',
    已在运行: '已在运行',
  };
  if (mapped[raw]) return mapped[raw];
  if (/debug-info|MMDet|MMPose|visual-dps-infer|docker/i.test(raw)) return '';
  if (raw.length > 80 || /^[A-Za-z_\-./:\\]+$/.test(raw)) return '';
  return formatUserError(raw);
}

export const STREAM_CONFIG_SAVED_HINT = '配置已保存，视频流设置已自动生效。';
