import fs from 'node:fs/promises';
import path from 'node:path';
import { openSync } from 'node:fs';
import { spawn } from 'node:child_process';
const root=path.resolve(import.meta.dirname,'..');
const sceneName=process.argv[2]??'property-rooms-corridor';
if(!/^[\p{L}\p{N}_-]+$/u.test(sceneName))throw new Error('장면 이름을 확인해 주세요.');
const output=path.join(root,'artifacts',sceneName.startsWith('procthor-')?'procthor-validation':'property-validation',sceneName.startsWith('procthor-')?'property-browser':'.');
await fs.mkdir(output,{recursive:true});
const graph=await(await fetch(`http://127.0.0.1:8765/scenes/${encodeURIComponent(sceneName)}/property_graph.json`)).json();
const supportedCount=graph.portals.filter(p=>p.status==='image_supported').length;
if(!supportedCount)throw new Error('검증할 사진 근거 연결이 없어요.');
const chrome=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',[
  '--headless=new','--no-sandbox','--remote-debugging-port=9340',
  `--user-data-dir=${path.join(output,`chrome-${Date.now()}`)}`,'--no-first-run','--window-size=1440,1100','about:blank',
],{windowsHide:true,stdio:['ignore','ignore',openSync(path.join(output,'chrome.log'),'w')]});
let ws;
try {
  let targets;
  for(let i=0;i<60;i++){try{targets=await(await fetch('http://127.0.0.1:9340/json')).json();break;}catch{await new Promise(r=>setTimeout(r,500));}}
  if(!targets)throw new Error('검증용 Chrome 연결 실패');
  ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  let id=0; const pending=new Map(),errors=[];
  ws.onmessage=event=>{const m=JSON.parse(event.data);if(m.id){const c=pending.get(m.id);if(!c)return;pending.delete(m.id);clearTimeout(c.timer);m.error?c.reject(new Error(JSON.stringify(m.error))):c.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);};
  ws.onclose=()=>{for(const c of pending.values()){clearTimeout(c.timer);c.reject(new Error('Chrome 연결 종료'));}pending.clear();};
  const send=(method,params={})=>new Promise((resolve,reject)=>{const call=++id;const timer=setTimeout(()=>{pending.delete(call);reject(new Error(`응답 시간 초과: ${method}`));},30000);pending.set(call,{resolve,reject,timer});ws.send(JSON.stringify({id:call,method,params}));});
  const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  const wait=async expression=>{for(let i=0;i<80;i++){if(await evaluate(expression))return;if(errors.length)throw new Error(JSON.stringify(errors));await new Promise(r=>setTimeout(r,500));}throw new Error(`화면 확인 실패: ${expression}`);};
  const screenshot=async name=>{await new Promise(r=>setTimeout(r,1500));const shot=await send('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(output,name),Buffer.from(shot.data,'base64'));};
  await send('Runtime.enable'); await send('Page.enable');
  await send('Page.navigate',{url:`http://127.0.0.1:8765/?scene=${encodeURIComponent(sceneName)}`});
  await wait(`document.querySelector('#property-summary')?.textContent.includes('관측 영역 후보 ${graph.spaces.length}개') && document.querySelector('#loading').hidden`);
  await evaluate(`document.querySelector('#property-section').scrollIntoView({block:'start'}); Array.from(document.querySelectorAll('.portal-evidence')).find(d=>d.querySelector('summary').textContent.startsWith('사진 근거 연결')).open=true`);
  await wait(`Array.from(Array.from(document.querySelectorAll('.portal-evidence')).find(d=>d.open).querySelectorAll('img')).every(i=>i.complete&&i.naturalWidth>0)`);
  const state=await evaluate(`({summary:document.querySelector('#property-summary').textContent,
    spaces:document.querySelectorAll('#property-spaces details').length,
    supported:Array.from(document.querySelectorAll('.portal-evidence summary')).filter(s=>s.textContent.startsWith('사진 근거 연결')).length,
    weak:Array.from(document.querySelectorAll('.portal-evidence summary')).filter(s=>s.textContent.startsWith('근거 부족')).length,
    evidenceImages:Array.from(document.querySelectorAll('.portal-evidence')).find(d=>d.open).querySelectorAll('img').length})`);
  if(state.spaces!==graph.spaces.length||state.supported!==supportedCount||state.evidenceImages<2)throw new Error('공간·문 연결 근거 표시 실패');
  await evaluate(`document.querySelector('[data-mode="top"]').click()`);
  await screenshot('property-plan-browser.png');
  await evaluate(`Array.from(document.querySelectorAll('.portal-evidence')).find(d=>d.open).querySelector('.photo-button').click()`);
  await wait(`document.querySelector('#property-active-photo').textContent.includes('현재 촬영 위치') && document.querySelector('[data-mode="orbit"]').classList.contains('on')`);
  const first=await evaluate(`document.querySelector('#property-active-photo').textContent`);
  await screenshot('opening-side-a.png');
  await evaluate(`Array.from(document.querySelectorAll('.portal-evidence')).find(d=>d.open).querySelectorAll('.photo-evidence')[1].querySelector('.photo-button').click()`);
  const second=await evaluate(`document.querySelector('#property-active-photo').textContent`);
  if(first===second)throw new Error('문 반대편 사진 이동 실패');
  await screenshot('opening-side-b.png');
  for(const p of graph.portals.filter(p=>p.status==='image_supported')){
    const sides=new Set(p.evidence_photos.map(photo=>graph.photo_spaces[photo]));
    if(!p.spaces.every(s=>sides.has(s)))throw new Error('문 양쪽 사진 근거 누락');
  }
  const report={scene:sceneName,...state,sideA:first,sideB:second,confirmedRoomCount:graph.confirmed_room_count,browserErrors:errors};
  await fs.writeFile(path.join(output,'browser.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
  if(errors.length)throw new Error('브라우저 오류 발생');
  await send('Browser.close').catch(()=>{});
}finally{ws?.close();if(chrome.exitCode===null)chrome.kill();}
