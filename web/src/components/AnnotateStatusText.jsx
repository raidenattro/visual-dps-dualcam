/** 标注页状态条：纯文本展示，避免 dangerouslySetInnerHTML 误解析错误信息中的 < */

export default function AnnotateStatusText({ html, className = '' }) {
  if (!html) return null;
  const text = String(html).replace(/<br\s*\/?>/gi, '\n');
  return (
    <div className={className} style={{ whiteSpace: 'pre-line' }}>
      {text}
    </div>
  );
}
