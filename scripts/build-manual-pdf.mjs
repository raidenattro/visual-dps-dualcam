#!/usr/bin/env node
/**
 * 生成可外发的单文件 PDF：合并 USER_MANUAL + DEPLOY + PIPELINE_SPLIT 要点 + .env
 * 输出: docs/DiDPS-使用手册.pdf
 */
import { readFile, writeFile } from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { chromium } from 'playwright';
import { marked } from 'marked';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const DOCS = path.join(ROOT, 'docs');
const OUT_MD = path.join(DOCS, 'DiDPS-使用手册-完整版.md');
const OUT_PDF = path.join(DOCS, 'DiDPS-使用手册.pdf');
const IMG_DIR = path.join(DOCS, 'images/user-manual');

async function read(rel) {
  return readFile(path.join(ROOT, rel), 'utf8');
}

async function embedImages(md) {
  const re = /!\[([^\]]*)\]\(\.\/images\/user-manual\/([^)]+)\)/g;
  let out = md;
  const seen = new Map();
  let m;
  while ((m = re.exec(md)) !== null) {
    const file = m[2];
    if (!seen.has(file)) {
      const buf = await readFile(path.join(IMG_DIR, file));
      const ext = path.extname(file).slice(1) || 'png';
      seen.set(file, `data:image/${ext};base64,${buf.toString('base64')}`);
    }
  }
  out = out.replace(re, (_, alt, file) => `![${alt}](${seen.get(file)})`);
  return out;
}

function buildMergedMd(manual, deploy, pipeline, envExample) {
  const version = 'v26.1.1.build211';
  let body = manual
    .replace(/^# DiDPS 使用手册\n/, '')
    .replace(
      /\*\*相关文档\*\*[\s\S]*$/,
      '',
    )
    .replace(
      /### 监控页 HLS 404 \/ WebRTC 黑屏\n\n- 见 \[DEPLOY\.md\].*?\n/,
      '### 监控页 HLS 404 / WebRTC 黑屏\n\n- 详见本文 **附录 A** 播放排障表。\n',
    )
    .replace(/### 更新配图（维护者）[\s\S]*?```\n\n/, '')
    .replace(/### Final 推荐路径/, '### 推荐路径');

  return `# DiDPS 使用手册（完整版）

> **版本 ${version}** · 现场工程师与运维 · 单文件发行版（含部署与管道附录）  
> 生成时间：${new Date().toISOString().slice(0, 10)}

${body}

---

<div class="page-break"></div>

# 附录 A · Docker 交付与播放排障

${deploy.replace(/^# Docker Compose 交付部署\n\n/, '')}

---

<div class="page-break"></div>

# 附录 B · 推理 / 事件管道与 Redis

${pipeline.replace(/^# 推理 \/ 事件 管道拆分.*\n\n/, '')}

---

<div class="page-break"></div>

# 附录 C · 环境变量参考（\`.env\`）

将项目根目录 \`.env.example\` 复制为 \`.env\` 后按需修改：

\`\`\`bash
${envExample.trim()}
\`\`\`

| 变量 | 说明 |
|------|------|
| \`REDIS_PASSWORD\` | Redis 密码，compose 各服务须一致 |
| \`UI_PORT\` | Web 访问端口，默认 8045 |
| \`MEDIAMTX_PUBLIC_HOST\` | 浏览器访问 WebRTC ICE 的宿主机 IP |
| \`MEDIAMTX_RTSP_PORT\` | RTSP 推流/拉流端口，默认 8554 |
| \`MEDIAMTX_WEBRTC_ICE_PORT\` | ICE UDP/TCP，默认 8189，须与 compose 端口映射一致 |
| \`INFERENCE_LITE_IMAGE\` | 轻量推理镜像名（MediaPipe / RTMPose-T） |
| \`INFERENCE_USE_GPU\` | 1 时优先使用 lite-gpu 镜像（NVIDIA 解码） |

---

*本文档由 visual-dps 仓库自动构建，配图与界面以实际部署为准。*
`;
}

const HTML_CSS = `
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Droid Sans Fallback", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #1a1a1a;
  max-width: 100%;
}
h1 { font-size: 22pt; color: #0d3d38; border-bottom: 2px solid #3dd6c6; padding-bottom: 8px; margin-top: 0; }
h2 { font-size: 15pt; color: #0d3d38; margin-top: 1.4em; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 1.1em; page-break-after: avoid; }
h4 { font-size: 11pt; }
blockquote {
  margin: 0.8em 0; padding: 8px 14px;
  background: #f0f7f6; border-left: 4px solid #3dd6c6;
  color: #333;
}
table { border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 9.5pt; }
th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }
th { background: #e8f4f2; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 8.5pt; line-height: 1.4; }
pre code { background: none; color: inherit; padding: 0; }
img { max-width: 100%; height: auto; display: block; margin: 12px auto; border: 1px solid #ddd; border-radius: 4px; }
.page-break { page-break-before: always; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
ul, ol { padding-left: 1.4em; }
a { color: #0d6e63; text-decoration: none; }
.cover-meta { color: #555; font-size: 10pt; margin-bottom: 1.5em; }
`;

async function mdToPdf(md, pdfPath) {
  const htmlBody = marked.parse(md, { gfm: true, breaks: false });
  const html = `<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><style>${HTML_CSS}</style></head><body>${htmlBody}</body></html>`;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.setContent(html, { waitUntil: 'load' });
  await page.pdf({
    path: pdfPath,
    format: 'A4',
    printBackground: true,
    margin: { top: '16mm', bottom: '18mm', left: '14mm', right: '14mm' },
  });
  await browser.close();
}

async function main() {
  const [manual, deploy, pipeline, envExample] = await Promise.all([
    read('docs/USER_MANUAL.md'),
    read('docs/DEPLOY.md'),
    read('docs/PIPELINE_SPLIT.md'),
    read('.env.example'),
  ]);

  let merged = buildMergedMd(manual, deploy, pipeline, envExample);
  merged = await embedImages(merged);
  await writeFile(OUT_MD, merged, 'utf8');
  console.log('wrote', OUT_MD);

  await mdToPdf(merged, OUT_PDF);
  console.log('wrote', OUT_PDF);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
