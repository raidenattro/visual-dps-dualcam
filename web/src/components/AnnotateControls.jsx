import GridPicker from './GridPicker';
import '../pages/AnnotatePage.css';

/** 标注工具栏（无画布），用于监控页抽屉 */
export default function AnnotateControls({ tool, embedded = false }) {
  const shelfReady = tool.shelfCornersReady;
  const gridReady = tool.gridGenerated;
  const selected = tool.selectedBoxCell;
  const activeShelf = tool.shelves?.find(
    (s) => String(s.shelf_code || '').trim() === String(tool.selectedShelfCode || '').trim(),
  );
  const activeShelfLabel =
    String(activeShelf?.shelf_name || '').trim() ||
    String(tool.selectedShelfCode || '').trim();

  return (
    <div className="annotate-controls-sidebar">
      {embedded && tool.selectedShelfCode ? (
        <div className="group annotate-shelf-panel">
          <div className="annotate-shelf-head">
            <span className="group-title">当前货架</span>
            <span className="annotate-shelf-name">{activeShelfLabel}</span>
          </div>
        </div>
      ) : null}

      <div className="group annotate-grid-group">
        <div className="annotate-section-head">
          <span className="group-title">货位矩阵</span>
          <button
            type="button"
            className="monitor-panel-btn monitor-panel-btn-primary"
            disabled={!shelfReady}
            title={
              shelfReady
                ? gridReady
                  ? `按当前 ${tool.gridRows}×${tool.gridCols} 重新生成货位`
                  : `生成 ${tool.gridRows}×${tool.gridCols} 货位`
                : '请先在画面上标定货架四角'
            }
            onClick={tool.confirmGenerateGrid}
          >
            {gridReady ? `重新生成 ${tool.gridRows}×${tool.gridCols}` : '生成货位'}
          </button>
        </div>
        <GridPicker
          rows={tool.gridRows}
          cols={tool.gridCols}
          onChange={tool.setGridDimensions}
        />
      </div>

      {!embedded ? (
        <div className="group">
          <div className="group-title">画面</div>
          <label className="field">
            货架名称
            <input
              type="text"
              value={tool.cameraName}
              onChange={(e) => tool.setCameraName(e.target.value)}
              placeholder="保存标注时的货架标识"
            />
          </label>
          <button type="button" className="secondary annotate-btn-block" onClick={tool.captureFrame}>
            刷新画面
          </button>
          <p className="annotate-embedded-hint">在预览上标注；静止帧请点「刷新画面」。</p>
        </div>
      ) : null}

      {gridReady ? (
        <div className="annotate-box-panel group">
          {selected ? (
            <>
              <div className="annotate-section-head annotate-box-head-row">
                <div className="annotate-box-head-text">
                  <span className="group-title">货位编号</span>
                  <p className="annotate-box-pos">
                    第 {selected.row} 层 · 第 {selected.col} 列
                  </p>
                </div>
                <button
                  type="button"
                  className="monitor-panel-btn monitor-panel-btn-danger"
                  title={`删除第 ${selected.row} 层 · 第 ${selected.col} 列货位`}
                  onClick={() => tool.promptDeleteSelectedCell()}
                >
                  删除货位
                </button>
              </div>
              <label className="field annotate-box-id-field">
                编号
                <input
                  type="text"
                  value={selected.value}
                  placeholder={selected.defaultId}
                  onChange={(e) =>
                    tool.onBoxIdChange(selected.rowIdx, selected.colIdx, e.target.value)
                  }
                />
              </label>
            </>
          ) : (
            <>
              <span className="group-title">货位编号</span>
              <p className="annotate-box-empty-hint">
                点击画面中的货位进行编辑；选中后点「删除货位」可移除该 ROI。
              </p>
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}
