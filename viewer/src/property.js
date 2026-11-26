import { makeLabel } from './labels.js';

const USAGE_NAME={living:'거실',bedroom:'침실',bathroom:'욕실',kitchen:'주방'};
// 용도 추정(사물 근거 우선, 없으면 공간 크기·모양·연결 모델). 확정이 아니라 후보다.
export function usageText(space) {
  const e=space.usage_estimate;
  if (!e||!e.usage||!USAGE_NAME[e.usage]) return '';
  const p=e.probabilities?.[e.usage];
  return `${USAGE_NAME[e.usage]} 추정`+(e.source==='objects'?'(사물 근거)':p!=null?`(형태 ${Math.round(p*100)}%)`:'');
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
  for (const [index,space] of graph.spaces.entries()) {
    const geometry=new THREE.BufferGeometry().setFromPoints(space.polygon_xz.map(([x,z])=>new THREE.Vector3(x,.025,z)));
    group.add(new THREE.LineLoop(geometry,new THREE.LineBasicMaterial({color:colors[index%colors.length]})));
    const usage=usageText(space);
    const label=makeLabel(usage?`${space.label} · ${usage}`:space.label,'dim');
    label.position.set(space.center_xz[0],.05,space.center_xz[1]);
    group.add(label);
  }
  for (const portal of graph.portals) {
    const [x,z]=portal.center_xz;
    const geometry=new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x,.02,z),new THREE.Vector3(x,1.1,z),
      new THREE.Vector3(x-.12,.03,z),new THREE.Vector3(x+.12,.03,z),
      new THREE.Vector3(x,.03,z-.12),new THREE.Vector3(x,.03,z+.12),
    ]);
    group.add(new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:portal.status==='image_supported'?0xdd8124:0x929292})));
    const label=makeLabel(portal.id+(portal.status==='image_supported'?'':' 미확인'),'dim'); label.position.set(x,.2,z); group.add(label);
  }
  return group;
}

