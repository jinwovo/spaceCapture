import fs from 'node:fs/promises';
import { openSync } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const output = path.join(root, 'artifacts/workflow-validation');
await fs.mkdir(output, { recursive: true });
const chrome = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', [
  '--headless=new', '--no-sandbox', '--remote-debugging-port=9339',
  `--user-data-dir=${path.join(output, `chrome-${Date.now()}`)}`,
  '--no-first-run', '--no-default-browser-check', '--window-size=1440,1100', 'about:blank',
], { windowsHide: true, stdio: ['ignore', 'ignore', openSync(path.join(output, 'chrome.log'), 'w')] });
let ws;
try {
  let targets;
  for (let i = 0; i < 60; i++) {
    try { targets = await (await fetch('http://127.0.0.1:9339/json')).json(); break; }
    catch { await new Promise(r => setTimeout(r, 500)); }
  }
  if (!targets) throw new Error('검증용 Chrome 연결 실패');
  ws = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  let id = 0;
  const pending = new Map(), errors = [];
  ws.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.id) {
      const call = pending.get(msg.id); if (!call) return;
      pending.delete(msg.id); clearTimeout(call.timer);
      if (msg.error) call.reject(new Error(JSON.stringify(msg.error))); else call.resolve(msg.result);
    } else if (msg.method === 'Runtime.exceptionThrown') errors.push(msg.params.exceptionDetails);
  };
  ws.onclose = () => {
    for (const call of pending.values()) { clearTimeout(call.timer); call.reject(new Error('Chrome 연결 종료')); }
    pending.clear();
  };
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const callId = ++id;
    const timer = setTimeout(() => { pending.delete(callId); reject(new Error(`응답 시간 초과: ${method}`)); }, 60000);
    pending.set(callId, { resolve, reject, timer }); ws.send(JSON.stringify({ id: callId, method, params }));
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const waitFor = async (expression, timeout = 30000) => {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      if (await evaluate(expression)) return;
      if (errors.length) throw new Error(JSON.stringify(errors));
      await new Promise(r => setTimeout(r, 500));
    }
    throw new Error(`화면 확인 시간 초과: ${expression}`);
  };
  const screenshot = async name => {
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    await fs.writeFile(path.join(output, name), Buffer.from(shot.data, 'base64'));
  };
  await send('Runtime.enable'); await send('Page.enable');
  await send('Page.navigate', { url: 'http://127.0.0.1:8765/?scene=synth' });
  await waitFor(`document.querySelector('#capture-status')?.textContent.includes('연결됨') && document.querySelector('#loading').hidden`);
  const title = `사무실 업로드 검증 ${Date.now()}`;
  const reuse = process.argv.includes('--reuse');
  await evaluate(`document.querySelector('#capture-title').value=${JSON.stringify(title)}; document.querySelector('#capture-space').value='office'`);
  const dom = await send('DOM.getDocument');
  const input = await send('DOM.querySelector', { nodeId: dom.root.nodeId, selector: '#capture-files' });
  const source = path.join(root, 'artifacts/photo-validation/input');
  const files = (await fs.readdir(source)).filter(n => n.endsWith('.jpg')).map(n => path.join(source, n));
  await send('DOM.setFileInputFiles', { nodeId: input.nodeId, files });
  const t0 = Date.now();
  if (!reuse) await evaluate(`document.querySelector('#capture-form').requestSubmit()`);
  let job, previous = '';
  while (Date.now() - t0 < 300000) {
    const jobs = await (await fetch('http://127.0.0.1:8765/api/jobs')).json();
    job = reuse ? jobs.find(j => j.title.startsWith('사무실 업로드 검증') && j.status === 'complete') : jobs.find(j => j.title === title);
    if (job && previous !== job.message) { console.log(job.message); previous = job.message; }
    if (job?.status === 'failed' || job?.status === 'needs_review') throw new Error(JSON.stringify(job));
    if (job?.status === 'complete') break;
    await new Promise(r => setTimeout(r, 2000));
  }
  if (job?.status !== 'complete') throw new Error('업로드 분석 시간 초과');
  await waitFor(`Array.from(document.querySelectorAll('#capture-result button')).some(b=>b.textContent.includes('3D 결과'))`);
  await evaluate(`document.querySelector('#capture-result button').click()`);
  await waitFor(`document.querySelector('#scene-select').value===${JSON.stringify(job.scene)} && document.querySelector('#loading').hidden && !document.querySelector('#show-splat').disabled`);
  await new Promise(r => setTimeout(r, 2000));
  const renderLoad = await evaluate(`(async () => {
    const proto = WebGL2RenderingContext.prototype, saved = {}, names = ['drawArrays','drawElements','drawArraysInstanced','drawElementsInstanced'];
    let calls = 0;
    for (const name of names) { saved[name] = proto[name]; proto[name] = function(...args) { calls++; return saved[name].apply(this,args); }; }
    try {
      window.dispatchEvent(new CustomEvent('spacecapture-processing',{detail:true})); calls=0;
      await new Promise(r=>setTimeout(r,1200)); const reduced=calls;
      window.dispatchEvent(new CustomEvent('spacecapture-processing',{detail:false})); calls=0;
      await new Promise(r=>setTimeout(r,1200)); const normal=calls;
      return {reducedDrawCalls:reduced,normalDrawCalls:normal};
    } finally { for (const name of names) proto[name]=saved[name]; }
  })()`);
  if (!(renderLoad.reducedDrawCalls > 0 && renderLoad.normalDrawCalls > renderLoad.reducedDrawCalls)) throw new Error('분석 중 GPU 미리보기 부하 제한 실패');
  await screenshot('uploaded-3d.png');
  await evaluate(`document.querySelector('[data-mode="top"]').click()`);
  await new Promise(r => setTimeout(r, 1500));
  await screenshot('uploaded-plan.png');
  const scene = await (await fetch(`http://127.0.0.1:8765/scenes/${job.scene}/scene.json`)).json();
  if (scene.source.declared_space_type !== 'office' || scene.assessment.checks.indoor_outdoor_and_use.status !== 'not_checked') {
    throw new Error('사용자 지정 용도를 자동 분류 결과로 취급했어요.');
  }
  const plan = await fetch(`http://127.0.0.1:8765/scenes/${job.scene}/floorplan.svg`);
  if (!plan.ok || !(await plan.text()).includes('<svg')) throw new Error('SVG 도면 다운로드 실패');
  await evaluate(`localStorage.setItem('spacecapture:validation-l-office:layout',JSON.stringify([{preserved:true}]))`);
  await send('Page.navigate', { url: 'http://127.0.0.1:8765/?scene=validation-l-office' });
  await waitFor(`document.querySelector('#scene-select')?.value==='validation-l-office' && document.querySelector('[data-mode="walk"]').disabled`);
  const inspection = await evaluate(`({area:document.querySelector('#dim-area').textContent,
    walkDisabled:document.querySelector('[data-mode="walk"]').disabled,
    furnitureDisabled:Array.from(document.querySelectorAll('#furniture-section button')).every(b=>b.disabled),
    savedLayoutPreserved:JSON.parse(localStorage.getItem('spacecapture:validation-l-office:layout'))?.[0]?.preserved===true})`);
  if (!inspection.area.includes('확정 보류') || !inspection.furnitureDisabled || !inspection.savedLayoutPreserved) throw new Error('불확실한 구조의 도면 제한 실패');
  await evaluate(`document.querySelector('[data-mode="top"]').click()`);
  await new Promise(r => setTimeout(r, 1500));
  await screenshot('l-office-inspection.png');
  const report = { scene: job.scene, uploadedFiles: files.length, elapsed_s: (Date.now() - t0) / 1000,
    job, inspection, renderLoad, browserErrors: errors };
  await fs.writeFile(path.join(output, reuse ? 'viewer-review.json' : 'result.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ scene: job.scene, uploadedFiles: files.length, elapsed_s: report.elapsed_s,
    registered: scene.quality.registered, layout: scene.layout.status, inspection, browserErrors: errors }, null, 2));
  if (errors.length) throw new Error('브라우저 오류 발생');
  await send('Browser.close').catch(() => {});
} finally {
  ws?.close(); if (chrome.exitCode === null) chrome.kill();
}
