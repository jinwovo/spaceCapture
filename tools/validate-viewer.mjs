// Local Chrome/CDP check; no browser automation package or remote service.
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { openSync } from 'node:fs';

const root = path.resolve(import.meta.dirname, '..');
const sceneName = process.argv[2] ?? 'synth';
if (!/^[\p{L}\p{N}_-]+$/u.test(sceneName)) throw new Error('장면 이름을 확인해 주세요.');
const output = path.join(root, 'artifacts', sceneName === 'synth' ? 'brush-validation' : 'photo-validation');
await fs.mkdir(output, { recursive: true });
const chrome = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', [
  '--headless=new', '--remote-debugging-port=9226',
  `--user-data-dir=${path.join(output, `chrome-profile-${Date.now()}`)}`,
  '--no-sandbox',
  '--no-first-run', '--no-default-browser-check', '--window-size=1440,1000', 'about:blank',
], { windowsHide: true, stdio: ['ignore', 'ignore', openSync(path.join(output, 'chrome.log'), 'w')] });
let ws;
try {
  const deadline = Date.now() + 30000;
  let targets;
  while (Date.now() < deadline) {
    try { targets = await (await fetch('http://127.0.0.1:9226/json')).json(); break; }
    catch { await new Promise(r => setTimeout(r, 500)); }
  }
  if (!targets) throw new Error('검증용 Chrome 연결 실패');
  ws = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  let id = 0;
  const pending = new Map();
  const errors = [];
  ws.onclose = () => {
    for (const p of pending.values()) { clearTimeout(p.timer); p.reject(new Error('Chrome 연결 종료')); }
    pending.clear();
  };
  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.id) {
      const p = pending.get(msg.id);
      pending.delete(msg.id);
      clearTimeout(p.timer);
      if (msg.error) p.reject(new Error(JSON.stringify(msg.error))); else p.resolve(msg.result);
    } else if (msg.method === 'Runtime.exceptionThrown') errors.push(msg.params.exceptionDetails);
  };
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const callId = ++id;
    const timer = setTimeout(() => { pending.delete(callId); reject(new Error(`Chrome 응답 시간 초과: ${method}`)); }, 30000);
    pending.set(callId, { resolve, reject, timer });
    ws.send(JSON.stringify({ id: callId, method, params }));
  });
  const evaluate = async expression => {
    const response = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
    return response.result.value;
  };
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: `http://localhost:5173/?scene=${encodeURIComponent(sceneName)}` });
  let ready = false;
  const loadingDeadline = Date.now() + 120000;
  while (Date.now() < loadingDeadline) {
    ready = await evaluate(`!!window.__spacecapture?.splat?.isInitialized && document.querySelector('#loading').hidden`);
    // Spark exposes initialized as a promise; numSplats is also checked below.
    if (!ready) ready = await evaluate(`!!window.__spacecapture?.splat && document.querySelector('#loading').hidden && !document.querySelector('#show-splat').disabled`);
    if (ready) break;
    if (errors.length) throw new Error(JSON.stringify(errors));
    await new Promise(r => setTimeout(r, 1000));
  }
  if (!ready) throw new Error('실사 3D 로딩 시간 초과');
  await evaluate(`new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))`);
  const state = await evaluate(`(() => {
    const s = window.__spacecapture, canvas = document.querySelector('canvas');
    const gl = canvas.getContext('webgl2'), ext = gl.getExtension('WEBGL_debug_renderer_info');
    return { scene: document.querySelector('#scene-select').value,
      splats: s.splat.numSplats, visible: s.splat.visible,
      renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      body: document.body.innerText, canvas: [canvas.width, canvas.height] };
  })()`);
  async function screenshot(name) {
    await new Promise(r => setTimeout(r, 2000));
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    await fs.writeFile(path.join(output, name), Buffer.from(shot.data, 'base64'));
  }
  await screenshot('viewer-3d.png');
  const buttons = await evaluate(`Array.from(document.querySelectorAll('button')).map(b => ({id:b.id,text:b.textContent}))`);
  await fs.writeFile(path.join(output, 'viewer-buttons.json'), JSON.stringify(buttons, null, 2));
  const top = buttons.find(b => b.text.includes('평면도'));
  if (!top) throw new Error('평면도 버튼 없음');
  await evaluate(`document.querySelector('[data-mode="top"]').click()`);
  await screenshot('viewer-plan.png');
  state.errors = errors;
  await fs.writeFile(path.join(output, 'viewer.json'), JSON.stringify(state, null, 2));
  console.log(JSON.stringify({ ...state, body: state.body.slice(0, 1400) }, null, 2));
  if (errors.length || state.scene !== sceneName || !state.visible || !state.splats) {
    throw new Error('실사 3D 상태 검증 실패');
  }
  await send('Browser.close').catch(() => {});
} finally {
  ws?.close();
  if (chrome.exitCode === null) chrome.kill();
}
