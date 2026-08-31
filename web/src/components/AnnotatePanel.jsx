import { useRef } from 'react';
import { useAnnotateTool } from '../features/annotate/useAnnotateTool';
import '../pages/AnnotatePage.css';

/** 货架网格标注面板；embedded 时绑定单路摄像头，仅手动刷新画面 */
export default function AnnotatePanel({ embedded = false, fixedCamera = null }) {
  const canvasRef = useRef(null);
  const tool = useAnnotateTool(canvasRef, { fixedCamera, embedded });

  return (
    <div className={`annotate-page${embedded ? ' annotate-panel-embedded' : ''}`}>
      <div className="panel">
        <div className="controls-grid">
          <div className="group">
            <div className="group-title">网格设置</div>
            <div className="row">
              <label className="field">
                行数
                <select
                  value={tool.gridRows}
                  onChange={(e) => tool.setGridRows(Number(e.target.value))}
                >
                  {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                列数
                <select
                  value={tool.gridCols}
                  onChange={(e) => tool.setGridCols(Number(e.target.value))}
                >
                  {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
              <button type="button" className="secondary" onClick={tool.applyGridShape}>
                应用网格
              </button>
            </div>
          </div>

          {!embedded && (
            <>
              <div className="group">
                <div className="group-title">本地视频</div>
                <div className="row">
                  <input ref={tool.videoFileRef} type="file" accept="video/*" />
                  <button type="button" onClick={tool.uploadVideo}>
                    上传并开始标注
                  </button>
                </div>
              </div>

              <div className="group">
                <div className="group-title">网络摄像头</div>
                <div className="row">
                  <select
                    value={tool.cameraIp}
                    onChange={(e) => tool.setCameraIp(e.target.value)}
                  >
                    <option value="">选择已保存的摄像头</option>
                    {tool.cameraIpList.map((item) => (
                      <option key={item.url} value={item.url}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                  <input
                    type="text"
                    value={tool.cameraIp}
                    onChange={(e) => tool.setCameraIp(e.target.value)}
                    placeholder="输入视频流地址"
                  />
                  <input
                    type="text"
                    value={tool.cameraName}
                    onChange={(e) => tool.setCameraName(e.target.value)}
                    placeholder="货架名称（必填）"
                  />
                  <button type="button" className="secondary" onClick={tool.deleteCameraIp}>
                    删除所选摄像头
                  </button>
                  <button type="button" className="secondary" onClick={tool.addCameraIp}>
                    保存摄像头
                  </button>
                  <button type="button" onClick={tool.captureFrame}>
                    通过摄像头抓帧标注
                  </button>
                </div>
              </div>
            </>
          )}

          {embedded && (
            <div className="group">
              <div className="group-title">画面</div>
              <div className="row">
                <label className="field">
                  货架名称
                  <input
                    type="text"
                    value={tool.cameraName}
                    onChange={(e) => tool.setCameraName(e.target.value)}
                    placeholder="用于保存标注"
                  />
                </label>
                <button type="button" onClick={tool.captureFrame}>
                  刷新画面
                </button>
              </div>
              <p className="annotate-embedded-hint">标注模式使用静止画面，请点击「刷新画面」获取最新一帧。</p>
            </div>
          )}

          <div className="group">
            <div className="group-title">标注操作</div>
            <div className="row">
              <button
                type="button"
                className="secondary"
                disabled={!tool.canSave}
                onClick={tool.saveAnnotation}
              >
                保存标注
              </button>
              <button type="button" className="secondary" onClick={tool.finishAnnotation}>
                结束标注
              </button>
              <button type="button" className="secondary" onClick={tool.resetAnnotation}>
                重置标注
              </button>
            </div>
          </div>
        </div>
        <div
          className={`status ${tool.statusClass}`}
          dangerouslySetInnerHTML={{ __html: tool.statusHtml }}
        />
      </div>

      <canvas ref={canvasRef} className="annotator-canvas" />

      {tool.boxEditorCells.length > 0 && (
        <div className="panel box-editor-panel">
          <div className="group-title">货位编号</div>
          <div className="row shelf-row">
            <label className="field">
              货架
              <select
                value={tool.selectedShelfCode}
                onChange={(e) => tool.setSelectedShelfCode(e.target.value)}
              >
                {tool.shelfSelectOptions.map((opt) => (
                  <option key={opt.value || 'empty'} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div
            className="box-editor-grid"
            style={{
              gridTemplateColumns: `repeat(${tool.gridCols}, minmax(110px, 1fr))`,
            }}
          >
            {tool.boxEditorCells.map((cell) => (
              <div key={cell.key} className="box-item">
                <div className="box-item-label">{cell.label}</div>
                <input
                  type="text"
                  value={cell.value}
                  title={`默认编号：${cell.defaultId}`}
                  onChange={(e) => tool.onBoxIdChange(cell.rowIdx, cell.colIdx, e.target.value)}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
