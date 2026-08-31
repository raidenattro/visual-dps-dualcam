#!/usr/bin/env node
/**
 * 抓取 docs/images/user-manual/ 配图（需 UI 已启动：http://127.0.0.1:8045）
 */
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'docs/images/user-manual');
const BASE = process.env.UI_BASE || 'http://127.0.0.1:8045';
const USER = process.env.MANUAL_USER || 'admin';
const PASS = process.env.MANUAL_PASS || 'admin123';

async function shot(page, name, opts = {}) {
  const file = path.join(OUT, `${name}.png`);
  if (opts.before) await opts.before(page);
  if (opts.url) {
    await page.goto(opts.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  }
  if (opts.waitMs) await page.waitForTimeout(opts.waitMs);
  await page.screenshot({ path: file, fullPage: Boolean(opts.fullPage) });
  console.log('wrote', file);
}

async function login(page) {
  await page.fill('input[type="text"], input[autocomplete="username"]', USER);
  await page.fill('input[type="password"]', PASS);
  await page.getByRole('button', { name: /登录/ }).click();
  await page.waitForURL((url) => !url.pathname.includes('/login'), { timeout: 15000 });
}

async function renderTerminalShot(browser, name, lines) {
  const html = `<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{margin:0;background:#1e1e1e;font-family:ui-monospace,Menlo,Consolas,monospace;padding:24px}
pre{color:#d4d4d4;font-size:14px;line-height:1.45;white-space:pre-wrap;margin:0}
.ok{color:#4ec9b0}.hi{color:#dcdcaa}.url{color:#9cdcfe}
</style></head><body><pre>${lines
    .map((l) => l.replace(/&/g, '&amp;').replace(/</g, '&lt;'))
    .join('\n')}</pre></body></html>`;
  const p = await browser.newPage({ viewport: { width: 920, height: 420 } });
  await p.setContent(html);
  await p.screenshot({ path: path.join(OUT, `${name}.png`) });
  await p.close();
  console.log('wrote', path.join(OUT, `${name}.png`));
}

async function main() {
  await mkdir(OUT, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    locale: 'zh-CN',
  });
  const page = await context.newPage();

  await page.goto(`${BASE}/login`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(400);
  await shot(page, '01-login');

  await login(page);
  await shot(page, '02-dashboard', { url: `${BASE}/`, waitMs: 2500 });
  await shot(page, '03-monitor', { url: `${BASE}/monitor?camera=cam4`, waitMs: 4000 });
  await shot(page, '04-monitor-annotate', {
    url: `${BASE}/monitor?camera=cam4&mode=annotate`,
    waitMs: 3000,
  });
  await shot(page, '05-matrix', { url: `${BASE}/matrix`, waitMs: 2000 });
  await shot(page, '06-settings', { url: `${BASE}/settings`, waitMs: 1500 });

  await shot(page, '07-camera-drawer', {
    url: `${BASE}/`,
    waitMs: 1000,
    before: async (p) => {
      const settingsBtn = p.getByTitle('设置').first();
      await settingsBtn.waitFor({ state: 'visible', timeout: 10000 });
      await settingsBtn.click();
      await p.locator('.drawer, [class*="drawer"]').first().waitFor({ state: 'visible', timeout: 8000 }).catch(() => {});
      await p.waitForTimeout(600);
    },
  });

  const ffprobe = `ffprobe -v error -show_entries stream=codec_name -of default=noprint_wrappers=1 rtsp://127.0.0.1:8554/cam4`;
  await renderTerminalShot(browser, '09-architecture', [
    '┌─────────────────────────────────────────────────────────────────┐',
    '│  ffmpeg / 摄像头  ──RTSP──►  MediaMTX (:8554)                   │',
    '│       cam1 … camN          rtsp://127.0.0.1:8554/<path>          │',
    '└───────────────────────────────┬─────────────────────────────────┘',
    '                                │',
    '        ┌───────────────────────┼───────────────────────┐',
    '        ▼                       ▼                       ▼',
    '  visual-dps-ui          infer-cam* (按路)      visual-dps-event-worker',
    '  Web :8045              检测+17点姿态          碰撞/报警/Java回调',
    '        │                       │                       │',
    '        └─────────── Redis (pose / event) ──────────────┘',
    '                                │',
    '                         浏览器 SSE 监控页',
    '└─────────────────────────────────────────────────────────────────┘',
  ]);

  await renderTerminalShot(browser, '08-ffmpeg-publish', [
    '$ ./scripts/start-mp4-rtsp.sh /path/to/demo.mp4 cam4',
    'MP4 推流已启动: cam4 (pid=12345)',
    '源文件: /path/to/demo.mp4',
    '<span class="url">RTSP: rtsp://127.0.0.1:8554/cam4</span>',
    '日志: scripts/.local/ffmpeg-mp4-rtsp-cam4.log',
    '',
    '$ ./scripts/start-mp4-rtsp-multi.sh cam1 cam2 cam3 cam4',
    '>>> 启动推流 cam1',
    '>>> 启动推流 cam2',
    '...',
    '<span class="ok">已处理 4 路。Dashboard 刷新后应显示在线。</span>',
    '',
    '$ ' + ffprobe,
    'codec_name=h264',
  ]);

  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
