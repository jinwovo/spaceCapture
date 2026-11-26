// Freeze only the automation-owned page while its GPU training job completes.
const targets = await (await fetch('http://127.0.0.1:9339/json')).json();
const target = targets.find(t => t.type === 'page');
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
let id = 0;
const pending = new Map();
ws.onmessage = event => { const m = JSON.parse(event.data); if (m.id) { pending.get(m.id)?.(m.result); pending.delete(m.id); } };
const send = (method, params) => new Promise(resolve => { const call = ++id; pending.set(call, resolve); ws.send(JSON.stringify({ id: call, method, params })); });
await send('Page.setWebLifecycleState', { state: 'frozen' });
console.log('검증용 뷰어 일시 정지 · 학습 GPU 부하 측정');
for (let i = 0; i < 180; i++) {
  const jobs = await (await fetch('http://127.0.0.1:8765/api/jobs')).json();
  if (!jobs.some(j => j.status === 'processing')) break;
  await new Promise(r => setTimeout(r, 1000));
}
await send('Page.setWebLifecycleState', { state: 'active' });
console.log('검증용 뷰어 재개');
ws.close();
