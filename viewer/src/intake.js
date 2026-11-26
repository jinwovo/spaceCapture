const $ = (id) => document.getElementById(id);
const statusLabels = { uploading: "전송 중", queued: "대기 중", processing: "분석 중", complete: "완료", failed: "분석 실패", needs_review: "확인 필요" };
const checkLabels = { geometric_evidence: "복원 근거", photo_connectivity: "사진 연결", indoor_outdoor_and_use: "실내·실외와 용도", layout_topology: "벽 구조", glass_and_mirrors: "유리·거울", metric_scale: "실제 크기", space_opening_connections:"공간·문 연결" };
const checkStates = { available: "근거 있음", limited: "근거 부족", connected: "연결됨", disconnected: "분리됨", not_checked: "미검사", partially_checked: "부분 검사", marker_based: "마커 기준", assumed: "가정" };

const node = (tag, text, className) => Object.assign(document.createElement(tag), { textContent: text ?? "", ...(className ? { className } : {}) });

export function renderAssessment(parent, assessment) {
  parent.replaceChildren();
  if (!assessment) { parent.append(node("p", "이 장면에는 검사 기록이 없어요.", "muted small")); return; }
  for (const [key, check] of Object.entries(assessment.checks ?? {})) {
    const row = node("div", "", "evidence-row");
    row.append(node("b", checkLabels[key] ?? key), node("span", checkStates[check.status] ?? check.status));
    row.append(node("p", check.message, "muted small"));
    parent.append(row);
  }
}

async function request(url, options) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { throw new Error("로컬 분석 서버에 연결하지 못했어요."); }
  if (!response.ok) throw new Error(data.error ?? "요청을 처리하지 못했어요.");
  return data;
}

export function initIntake(onOpen) {
  let selected = null, busy = false, jobs = [], lastJobs = "", lastResult = "", lastProcessing = null;
  const status = $("capture-status");
  $("capture-kind").onchange = () => {
    const kind = $("capture-kind").value, photoExt = ".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff", videoExt = ".mp4,.mov,.m4v,.avi,.webm";
    $("capture-files").multiple = kind !== "video";
    // 영상 + 보충 사진: 집을 걸으며 찍은 영상 1개와, 덜 찍힌 방·구석을 찍은 사진을 한 번에 고른다.
    $("capture-files").accept = kind === "photos" ? photoExt : kind === "video" ? videoExt : `${videoExt},${photoExt}`;
    $("capture-files").value = "";
  };
  function render() {
    const key = JSON.stringify([selected, jobs.map(j => [j.id, j.status, j.progress, j.message])]);
    if (key !== lastJobs) {
      lastJobs = key;
      $("capture-jobs").replaceChildren(...jobs.slice(0, 5).map(job => {
        const button = node("button", `${job.title} · ${statusLabels[job.status] ?? job.status}`, "btn job-choice");
        button.classList.toggle("on", job.id === selected);
        button.onclick = () => { selected = job.id; render(); };
        return button;
      }));
    }
    const job = jobs.find(j => j.id === selected);
    const resultKey = JSON.stringify(job);
    if (resultKey === lastResult) return;
    lastResult = resultKey;
    const output = $("capture-result");
    output.replaceChildren();
    if (!job) return;
    output.append(node("p", job.message, "job-message"));
    if (["queued", "processing"].includes(job.status)) {
      const progress = document.createElement("progress");
      progress.max = 100; progress.value = job.progress;
      progress.setAttribute("aria-label", "분석 단계 진행률");
      output.append(progress);
    }
    if (!job.result) return;
    const result = job.result;
    output.append(node("p", `사진·프레임 ${result.registered}/${result.input}장 위치 복원 · ${result.splat ? "실사 3D 있음" : "점 기반 3D"}`, "small"));
    const open = node("button", "3D 결과 열기", "btn primary");
    open.onclick = () => onOpen(result.scene, job.title).catch(error => { status.textContent = error.message; status.dataset.sticky = "true"; });
    const base = `/scenes/${encodeURIComponent(result.scene)}/`;
    const hasPlan = result.plan && !result.plan.status;
    const plan = node("a", hasPlan ? "평면도 PNG" : (result.property_graph ? "공간 연결 도면 SVG" : "2D 구조 후보 SVG"), "btn");
    plan.href = base + (hasPlan ? "plan.png" : (result.property_graph ? "property.svg" : "floorplan.svg"));
    plan.download = `${job.title}-${hasPlan ? "평면도.png" : "구조후보.svg"}`;
    const actions = node("div", "", "row"); actions.append(open, plan);
    if (hasPlan) {
      const vector = node("a", "평면도 SVG", "btn"); vector.href = base + "plan.svg"; vector.download = `${job.title}-평면도.svg`;
      actions.append(vector);
      const preview = document.createElement("img");
      preview.src = base + "plan.png"; preview.alt = "사진으로 추정한 평면도"; preview.className = "plan-preview";
      preview.loading = "lazy"; output.append(actions, preview);
    } else output.append(actions);
    const details = node("details", "", "evidence");
    details.append(node("summary", "분석 근거와 미확인 항목"));
    const checks = node("div"); renderAssessment(checks, result.assessment); details.append(checks); output.append(details);
    if (result.connectivity?.groups.length > 1) {
      const groups = node("details", "", "evidence");
      groups.append(node("summary", `서로 연결되지 않는 사진 묶음 ${result.connectivity.groups.length}개`));
      for (const group of result.connectivity.groups) {
        groups.append(node("p", `묶음 ${group.id} · ${group.count}장: ${(group.source_photos ?? group.images).join(", ")}`, "small"));
      }
      output.append(groups);
    }
  }
  async function refresh() {
    try {
      jobs = await request("/api/jobs");
      const processing = jobs.some(job => job.status === "processing");
      if (processing !== lastProcessing) {
        lastProcessing = processing;
        window.dispatchEvent(new CustomEvent("spacecapture-processing", { detail: processing }));
      }
      if (!selected && jobs.length) selected = jobs[0].id;
      if (!busy && status.dataset.sticky !== "true") status.textContent = "로컬 분석 서버 연결됨";
      render();
    } catch {
      if (!busy && status.dataset.sticky !== "true") status.textContent = "분석 서버 연결 전 · README의 로컬 앱 실행 방법을 확인해 주세요.";
    }
  }
  $("capture-form").onsubmit = async event => {
    event.preventDefault();
    if (busy) return;
    const files = Array.from($("capture-files").files), kind = $("capture-kind").value;
    const videos = files.filter(f => /\.(mp4|mov|m4v|avi|webm)$/i.test(f.name)).length;
    const invalid = kind === "photos" ? files.length < 3 || files.length > 150
      : kind === "video" ? files.length !== 1 : videos !== 1 || files.length - 1 < 3 || files.length - 1 > 150;
    if (invalid) {
      status.textContent = kind === "mixed" ? "영상 1개와 보충 사진 3~150장을 함께 선택해 주세요." : "사진 3~150장 또는 영상 1개를 선택해 주세요.";
      status.dataset.sticky = "true"; return;
    }
    busy = true; status.dataset.sticky = "false"; $("capture-submit").disabled = true;
    try {
      const job = await request("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        kind, title: $("capture-title").value || "새 공간", space_type: $("capture-space").value,
        splat: $("capture-splat").checked, files: files.map(f => ({ name: f.name, size: f.size })),
      }) });
      selected = job.id;
      for (let i = 0; i < files.length; i++) {
        status.textContent = `이 PC로 파일 전송 ${i + 1}/${files.length}`;
        await request(`/api/jobs/${job.id}/files/${i}`, { method: "PUT", body: files[i] });
      }
      await request(`/api/jobs/${job.id}/start`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      status.textContent = "자료 전송을 마쳤어요.";
    } catch (error) { status.textContent = error.message; status.dataset.sticky = "true"; }
    finally { busy = false; $("capture-submit").disabled = false; await refresh(); }
  };
  refresh();
  setInterval(refresh, 2000);
}
