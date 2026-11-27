import { makeLabel } from './labels.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const USAGE_NAME={living:'거실',bedroom:'침실',bathroom:'욕실',kitchen:'주방'};
// 용도 추정(사물 근거 우선, 없으면 공간 크기·모양·연결 모델). 확정이 아니라 후보다.
export function usageText(space) {
  const e=space.usage_estimate;
  if (!e||!e.usage||!USAGE_NAME[e.usage]) return '';
  const p=e.probabilities?.[e.usage];
  return `${USAGE_NAME[e.usage]} 추정`+(e.source==='objects'?'(사물 근거)':p!=null?`(형태 ${Math.round(p*100)}%)`:'');
}

function area(poly) {
  let s=0; for (let i=0;i<poly.length;i++){ const [x0,z0]=poly[i], [x1,z1]=poly[(i+1)%poly.length]; s+=x0*z1-x1*z0; }
  return Math.abs(s)/2;
}
// 평면도(plan_render)와 같은 규칙의 공간 이름: 용도가 있으면 '침실1·욕실1·거실', 없으면 '공간 N'.
export function spaceNames(graph) {
  const names={}, count={};
  const order=[...graph.spaces].sort((a,b)=>(a.center_xz[1]-b.center_xz[1])||(a.center_xz[0]-b.center_xz[0]));
  for (const s of order) {
    const usage=s.usage_estimate?.usage, base=USAGE_NAME[usage]??'공간';
    count[base]=(count[base]??0)+1;
    let name=(base==='거실'||base==='주방')?base:(base==='공간'?`공간 ${count[base]}`:`${base}${count[base]}`);
    if (!USAGE_NAME[usage]) {  // 확정 임계값 미만이어도 가장 가능성 높은 용도를 물음표로 알려 준다
      const p=s.usage_estimate?.probabilities??{};
      const best=Object.entries(p).filter(([k])=>USAGE_NAME[k]).sort((x,y)=>y[1]-x[1])[0];
      if (best && best[1]>=.5) name+=` (${USAGE_NAME[best[0]]}?)`;
      else if (s.usage_estimate?.passage_like) name+=' (복도?)';
    }
    names[s.id]={name, estimated: !!USAGE_NAME[usage], area: area(s.polygon_xz)};
  }
  return names;
}

const el = (tag, text, className) => Object.assign(document.createElement(tag), {textContent:text ?? "", ...(className?{className}:{})});

export function renderProperty(data, name, onPhoto) {
  const summary=document.getElementById('property-summary');
  const spaces=document.getElementById('property-spaces'), portals=document.getElementById('property-portals');
  const download=document.getElementById('property-plan-download');
  spaces.replaceChildren(); portals.replaceChildren();
  document.getElementById('property-active-photo').textContent='';
  const graph=data.property_graph;
  download.hidden=!data.files?.property_plan;
  if (!graph) { summary.textContent='이 장면에는 집·공간 연결 검사 기록이 없어요.'; return; }
  const supported=graph.portals.filter(portal=>portal.status==='image_supported').length;
  const weak=graph.portals.length-supported;
  summary.textContent=`관측 영역 후보 ${graph.spaces.length}개 · 사진 근거가 있는 문/개구부 연결 ${supported}개${weak?` · 근거 부족 ${weak}개`:''} · 실제 방 개수·용도 미확정`;
  if (data.files?.property_plan) download.href=`/scenes/${encodeURIComponent(name)}/${encodeURIComponent(data.files.property_plan)}`;
  const cameras=new Map((data.cameras??[]).map(camera=>[camera.name,camera]));
  function photoButtons(names, limit=8) {
    const grid=el('div','','photo-evidence');
    for (const photo of names.slice(0,limit)) {
      const camera=cameras.get(photo); if (!camera) continue;
      const button=el('button','','photo-button'); button.type='button';
      button.title=`${camera.source_photo??photo} · 촬영 위치로 이동`;
      if (camera.preview) {
        const image=document.createElement('img'); image.loading='lazy';
        image.src=`/scenes/${encodeURIComponent(name)}/${encodeURIComponent(camera.preview)}`;
        image.alt=camera.source_photo??photo; button.append(image);
      }
      button.append(el('span',camera.source_photo??photo));
      button.onclick=()=>onPhoto(photo); grid.append(button);
    }
    return grid;
  }
  for (const space of graph.spaces) {
    const details=el('details','','space-evidence');
    const usage=usageText(space);
    details.append(el('summary',`${space.label}${usage?' · '+usage:''} · 촬영 위치 ${space.photos.length}개`));
    details.append(el('p','사진을 누르면 그 촬영 위치에서 3D를 볼 수 있어요. 방 용도는 추정이며 완성된 바닥 면적은 미확정이에요.','small muted'));
    details.append(photoButtons(space.photos,16)); spaces.append(details);
  }
  for (const portal of graph.portals) {
    const details=el('details','','space-evidence portal-evidence');
    details.append(el('summary',`${portal.status==='image_supported'?'사진 근거 연결':'근거 부족'} · ${portal.id} · ${portal.spaces.join(' ↔ ')}`));
    details.append(el('p',`같은 3D 특징 ${portal.shared_landmarks}개 · ${portal.status==='image_supported'?'양쪽 사진의 기하 근거 있음':'사진 근거 부족'}${portal.view_projection_checked?' · 촬영 방향·화각 검사':''} · 유리·거울·통행 가능 여부 미확정`,'small muted'));
    for (const spaceId of portal.spaces) {
      const space=graph.spaces.find(s=>s.id===spaceId); if (!space) continue;
      details.append(el('b',`${space.label} 쪽 근거 사진`));
      details.append(photoButtons(portal.evidence_photos.filter(photo=>graph.photo_spaces[photo]===spaceId),6));
    }
    portals.append(details);
  }
}

export function propertyOverlay(THREE, graph) {
  const group=new THREE.Group();
  const colors=[0x3970bc,0xa47b25,0x398350,0x915eac,0x31878e];
  const names=spaceNames(graph);
  for (const [index,space] of graph.spaces.entries()) {
    const geometry=new THREE.BufferGeometry().setFromPoints(space.polygon_xz.map(([x,z])=>new THREE.Vector3(x,.025,z)));
    group.add(new THREE.LineLoop(geometry,new THREE.LineBasicMaterial({color:colors[index%colors.length]})));
    const n=names[space.id];
    const label=makeLabel(`${n.name} · ${n.area.toFixed(1)}㎡${n.estimated?' (추정)':''}`,'room-name');
    label.position.set(space.center_xz[0],.05,space.center_xz[1]);
    group.add(label);
  }
  for (const portal of graph.portals) {
    // 사진 근거가 있는 문만 표시한다(근거 부족 후보는 목록에서만 확인).
    if (portal.status!=='image_supported') continue;
    const [x,z]=portal.center_xz;
    const geometry=new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x,.02,z),new THREE.Vector3(x,1.1,z),
      new THREE.Vector3(x-.12,.03,z),new THREE.Vector3(x+.12,.03,z),
      new THREE.Vector3(x,.03,z-.12),new THREE.Vector3(x,.03,z+.12),
    ]);
    group.add(new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:0xdd8124})));
    const label=makeLabel(`문 ${Math.round(portal.width_m*100)}cm`,'dim'); label.position.set(x,.2,z); group.add(label);
  }
  return group;
}



const FLOOR_COLOR={living:0xe8d6b4,bedroom:0xd4e2ef,bathroom:0xcde6e0,kitchen:0xefe0c4};
// 복원된 공간 윤곽으로 세운 구조 모형(모형 집): 사진 근거가 있는 문 자리는 비우고, 바닥은 용도별 색.
// 공유 벽선 도면 모델(scene.floorplan)로 만든 구조 모형: 두께 있는 벽, 벽 윗면 단면색, 용도별 바닥.
// theme: 'light'(모형 집) | 'ai'(AI 재구성 보기: 어두운 배경, 밝은 단색 모형, 하늘색 단면선)
export const THEMES={
  light:{wall:0xfaf8f4,wallEmissive:0x5a5852,cap:0x3b3d40,open:0x7d8287,patch:0xd2bc98,unobserved:0xb9bcc0,floorDefault:0xdccfbb,
    floor:{living:0xd9bf96,bedroom:0xe2cba6,bathroom:0xc8d1d4,kitchen:0xd6c09c}},
  ai:{wall:0xe6ebf1,wallEmissive:0x20262e,cap:0x46c2ff,open:0x46c2ff,patch:0x56606c,unobserved:0x262c34,floorDefault:0x5a6370,
    floor:{living:0x5d6979,bedroom:0x566b7e,bathroom:0x4f6b6c,kitchen:0x6c6755}},
};
function planModel(THREE, plan, height, theme='light') {
  const T=THEMES[theme]??THEMES.light, group=new THREE.Group();
  // 내려다보는 모형이라 벽을 1.1m에서 자른 단면으로 보여 준다(문 자리는 비움).
  const cut=Math.min(1.1,height);
  const sides=[], caps=[];
  const box=(x0,z0,x1,z1,y0,y1)=>{  // 옆면 4개(바깥 법선) + 윗면은 단면색으로 따로
    const q=(a,b,c,d,n)=>{ for (const v of [a,b,c,a,c,d]) sides.push(...v,...n); };
    q([x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],[0,0,1]);
    q([x1,y0,z0],[x0,y0,z0],[x0,y1,z0],[x1,y1,z0],[0,0,-1]);
    q([x1,y0,z1],[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[1,0,0]);
    q([x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0],[-1,0,0]);
    if (y0>0) q([x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1],[0,-1,0]);
    caps.push(x0,y1+.002,z0, x0,y1+.002,z1, x1,y1+.002,z1, x0,y1+.002,z0, x1,y1+.002,z1, x1,y1+.002,z0);
  };
  for (const w of plan.walls) { const [x0,z0,x1,z1]=w.rect_xz; box(x0,z0,x1,z1,0,cut); }
  const geo=new THREE.BufferGeometry(), arr=new Float32Array(sides);
  const ib=new THREE.InterleavedBuffer(arr,6);
  geo.setAttribute('position',new THREE.InterleavedBufferAttribute(ib,3,0));
  geo.setAttribute('normal',new THREE.InterleavedBufferAttribute(ib,3,3));
  const walls=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:T.wall,emissive:T.wallEmissive,roughness:.9,metalness:0}));
  walls.castShadow=walls.receiveShadow=true; walls.userData.part='walls'; group.add(walls);
  const capGeo=new THREE.BufferGeometry(); capGeo.setAttribute('position',new THREE.Float32BufferAttribute(caps,3)); capGeo.computeVertexNormals();
  group.add(new THREE.Mesh(capGeo,new THREE.MeshBasicMaterial({color:T.cap,side:THREE.DoubleSide})));
  const floorMesh=(poly,color,y)=>{
    // (x, -z)로 만든 도형을 -90° 돌리면 (x, 0, z)가 되고 법선이 위(+y)를 향한다(햇빛·그림자가 제대로 받힘).
    const shape=new THREE.Shape(poly.map(([x,z])=>new THREE.Vector2(x,-z)));
    const m=new THREE.Mesh(new THREE.ShapeGeometry(shape).rotateX(-Math.PI/2),
      new THREE.MeshStandardMaterial({color,roughness:.95,side:THREE.DoubleSide}));
    m.position.y=y; m.receiveShadow=true; m.userData.part='floor'; group.add(m);
  };
  for (const r of plan.rooms) floorMesh(r.interior_polygon_xz,T.floor[r.usage]??T.floorDefault,.004);
  for (const [x0,z0,x1,z1] of plan.floor_patches??[]) floorMesh([[x0,z0],[x1,z0],[x1,z1],[x0,z1]],T.patch,.003);
  for (const u of plan.unobserved??[]) floorMesh(u,T.unobserved,.003);
  const open=(plan.open_edges??[]).flatMap(([x0,z0,x1,z1])=>[x0,.02,z0,x1,.02,z1]);  // 벽 근거 없는 관측 경계
  if (open.length) {
    const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.Float32BufferAttribute(open,3));
    const line=new THREE.LineSegments(g,new THREE.LineDashedMaterial({color:T.open,dashSize:.15,gapSize:.1})); line.computeLineDistances(); group.add(line);
  }
  return group;
}

// 인식한 가구를 테슬라 화면처럼 깔끔한 모형으로 다시 그린다. box3d(크기·방향 맞춤)가 믿을 만한 것만.
// 로컬 좌표: 가로 x, 깊이 z(+z가 앞), 바닥 y=0. 앞쪽은 front_xz(벽 반대쪽). 모서리는 둥근 상자.
const FURN={
  light:{frame:0xa8845c,mattress:0xf6f4ef,duvet:0x9fb3c8,pillow:0xffffff,sofa:0x7d8896,cushion:0x97a2b0,seat:0x5f6b78,
    metal:0x8d949c,wood:0xc49a6c,screen:0x16181b,appliance:0xe9ebee,ceramic:0xf7f7f5,plant:0x6f9a64,pot:0xb07a55,other:0xc9cfd6},
  ai:{frame:0xb9c1cb,mattress:0xf2f5f8,duvet:0xd5dce4,pillow:0xffffff,sofa:0xc3cbd5,cushion:0xdde3ea,seat:0xb6bfca,
    metal:0x8d97a3,wood:0xcfd5dc,screen:0x0b0d10,appliance:0xe7ecf1,ceramic:0xf5f7f9,plant:0xa9c9b0,pot:0xbfc6ce,other:0xd2d8df},
};
function furnitureModel(THREE, obj, theme='light') {
  const P=FURN[theme]??FURN.light;
  const b=obj.box3d, [w,h,d]=b.size, g=new THREE.Group();
  const mats={};
  const mat=(k)=>mats[k]??=new THREE.MeshStandardMaterial({color:P[k],roughness:k==='screen'?.35:.8,metalness:k==='metal'?.4:0,
    emissive:k==='screen'&&theme==='ai'?0x0d2233:0x000000});
  const part=(k,x0,y0,z0,x1,y1,z1,r=.025)=>{
    const sx=Math.max(x1-x0,.005), sy=Math.max(y1-y0,.005), sz=Math.max(z1-z0,.005);
    const rad=Math.min(r,sx/2.2,sy/2.2,sz/2.2);
    const geo=rad>.004?new RoundedBoxGeometry(sx,sy,sz,2,rad):new THREE.BoxGeometry(sx,sy,sz);
    const m=new THREE.Mesh(geo,mat(k)); m.position.set((x0+x1)/2,(y0+y1)/2,(z0+z1)/2); m.castShadow=m.receiveShadow=true; g.add(m); return m;
  };
  const W=w/2, D=d/2, leg=(k,top,t=.04,inset=t)=>{ for (const sx of [-1,1]) for (const sz of [-1,1]) part(k,sx*(W-inset)-t/2,0,sz*(D-inset)-t/2,sx*(W-inset)+t/2,top,sz*(D-inset)+t/2,.008); };
  switch (obj.class) {
    case 'bed': {
      const frame=Math.min(.28,h*.5), top=Math.max(h,frame+.12);
      part('frame',-W,.04,-D,W,frame,D,.03);                         // 받침
      leg('frame',.06,.06,.05);
      part('mattress',-W+.03,frame,-D+.05,W-.03,top,D-.03,.06);      // 매트리스
      part('frame',-W,0,-D-.07,W,Math.max(top+.4,.95),-D+.01,.03);   // 머리판(뒤쪽 벽)
      for (const sx of [-1,1]) part('pillow',sx*W*.5-W*.38,top-.01,-D+.1,sx*W*.5+W*.38,top+.11,-D+.42,.05);  // 베개
      part('duvet',-W+.01,top-.02,-D+.5,W-.01,top+.035,D+.005,.03);   // 이불
      break; }
    case 'couch': {
      const seat=Math.min(.43,h*.5), arm=Math.min(.2,w*.12), back=Math.min(.24,d*.32);
      part('sofa',-W,.06,-D,W,seat-.08,D,.04);                        // 바닥 몸체
      leg('metal',.07,.035,.06);
      part('sofa',-W,.06,-D,W,h,-D+back,.07);                         // 등받이
      for (const sx of [-1,1]) part('sofa',sx>0?W-arm:-W,.06,-D,sx>0?W:-W+arm,seat+.17,D,.07);  // 팔걸이
      const n=Math.max(1,Math.round((w-2*arm)/.65)), cw=(w-2*arm)/n;  // 방석 여러 칸
      for (let i=0;i<n;i++) part('cushion',-W+arm+i*cw+.01,seat-.08,-D+back,-W+arm+(i+1)*cw-.01,seat+.04,D-.02,.06);
      break; }
    case 'chair': {
      const seat=Math.min(.46,h*.52);
      leg('metal',seat-.05,.028,.05);
      part('seat',-W+.02,seat-.06,-D+.02,W-.02,seat,D-.02,.03);
      part('seat',-W+.04,seat-.02,-D+.02,W-.04,h,-D+.07,.03);         // 등받이
      break; }
    case 'dining table': {
      leg('wood',h-.04,.05,.07);
      part('wood',-W,h-.04,-D,W,h,D,.015);
      break; }
    case 'bench': {
      leg('wood',h-.05,.05,.06);
      part('wood',-W,h-.05,-D,W,h,D,.02);
      break; }
    case 'tv': {
      part('screen',-W,b.bottom,-D,W,b.bottom+h,D,.01);
      if (b.bottom>.25) part('metal',-.04,Math.max(0,b.bottom-.14),-.04,.04,b.bottom,.04,.01);
      break; }
    case 'refrigerator': {
      part('appliance',-W,0,-D,W,h,D,.03);
      part('metal',-W+.02,h*.62,D-.005,W-.02,h*.62+.008,D+.004,.002);   // 문 경계
      part('metal',W-.09,h*.67,D,W-.065,h*.9,D+.035,.01);               // 손잡이
      break; }
    case 'toilet': {
      part('ceramic',-W*.75,0,-D+.2,W*.75,h*.52,D,.08);                // 몸체
      part('ceramic',-W,h*.5,-D,W,h,-D+.2,.04);                        // 물탱크(뒤)
      break; }
    case 'oven': case 'microwave': case 'sink': {
      part('appliance',-W,0,-D,W,h,D,.02); break; }
    case 'potted plant': {
      part('pot',-W*.5,0,-D*.5,W*.5,h*.35,D*.5,.04);
      part('plant',-W,h*.35,-D,W,h,D,Math.min(W,D)*.8); break; }
    default: part('other',-W,0,-D,W,h,D,.03);
  }
  const [fx,fz]=b.front_xz??[0,1];
  g.rotation.y=Math.atan2(fx,fz);
  g.position.set(b.center[0],obj.class==='tv'?0:b.bottom,b.center[2]);
  g.userData.furniture=obj.class;
  return g;
}

export function structureModel(THREE, graph, height=2.4, plan=null, objects=null, theme='light') {
  if (plan?.walls?.length) {
    const group=planModel(THREE, plan, height, theme);
    for (const o of objects??[]) if (o.box3d?.display??o.box3d?.reliable) group.add(furnitureModel(THREE,o,theme));
    return group;
  }
  const group=new THREE.Group();
  const doors=graph.portals.filter(p=>p.status==='image_supported');
  const wallMat=new THREE.MeshStandardMaterial({color:0xf3f1ec,roughness:.9,side:THREE.DoubleSide,transparent:true,opacity:.92});
  const edgeMat=new THREE.LineBasicMaterial({color:0x5b6470});
  const verts=[], edges=[];
  const quad=(ax,az,bx,bz,y0,y1)=>{
    verts.push(ax,y0,az, bx,y0,bz, bx,y1,bz, ax,y0,az, bx,y1,bz, ax,y1,az);
    edges.push(ax,y1,az, bx,y1,bz);
  };
  const doorTop=Math.min(2.1,height-.1);
  for (const space of graph.spaces) {
    const poly=space.polygon_xz;
    const shape=new THREE.Shape(poly.map(([x,z])=>new THREE.Vector2(x,z)));
    const floor=new THREE.Mesh(new THREE.ShapeGeometry(shape).rotateX(Math.PI/2),
      new THREE.MeshStandardMaterial({color:FLOOR_COLOR[space.usage_estimate?.usage]??0xe4e4e4,roughness:1,side:THREE.DoubleSide}));
    floor.position.y=.005; group.add(floor);
    for (let i=0;i<poly.length;i++) {
      const [ax,az]=poly[i], [bx,bz]=poly[(i+1)%poly.length];
      const dx=bx-ax, dz=bz-az, len=Math.hypot(dx,dz); if (len<.05) continue;
      const gaps=[];
      for (const d of doors) {
        const [cx,cz]=d.center_xz, t=((cx-ax)*dx+(cz-az)*dz)/len;
        const off=Math.abs((cx-ax)*dz-(cz-az)*dx)/len;
        if (off<.45 && t>-.2 && t<len+.2) gaps.push([Math.max(0,t-d.width_m/2),Math.min(len,t+d.width_m/2)]);
      }
      gaps.sort((p,q)=>p[0]-q[0]);
      let s=0; const at=u=>[ax+dx*u/len, az+dz*u/len];
      for (const [g0,g1] of gaps) {
        if (g0>s) quad(...at(s),...at(g0),0,height);
        if (g1>g0) quad(...at(g0),...at(g1),doorTop,height);  // 문 위 인방
        s=Math.max(s,g1);
      }
      if (s<len) quad(...at(s),...at(len),0,height);
    }
  }
  const geometry=new THREE.BufferGeometry(); geometry.setAttribute('position',new THREE.Float32BufferAttribute(verts,3)); geometry.computeVertexNormals();
  group.add(new THREE.Mesh(geometry,wallMat));
  const lines=new THREE.BufferGeometry(); lines.setAttribute('position',new THREE.Float32BufferAttribute(edges,3));
  group.add(new THREE.LineSegments(lines,edgeMat));
  return group;
}
