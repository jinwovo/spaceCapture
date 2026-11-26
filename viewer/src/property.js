import { makeLabel } from './labels.js';

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
export function structureModel(THREE, graph, height=2.4) {
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
