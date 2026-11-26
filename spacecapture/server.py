"""Local upload API, sequential processing queue, and built viewer hosting.

Only loopback clients are accepted. Source photos stay on this PC. The server
does not fetch arbitrary URLs or expose uploads/work directories as static files.
"""
from __future__ import annotations

import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
SPACE_TYPES = {"unknown", "residential", "office", "meeting_room", "outdoor"}


class JobStore:
    def __init__(self, root: Path = ROOT, start_worker: bool = True):
        self.root = Path(root).resolve()
        self.directory = self.root / "artifacts/jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.queue = queue.Queue()
        for path in self.directory.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["status"] in {"processing", "queued"}:
                    job.update(status="failed", message="서버가 재시작되어 작업이 중단됐어요. 파일을 다시 선택해 실행해 주세요.")
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                continue
        if start_worker:
            threading.Thread(target=self._work, daemon=True, name="spacecapture-jobs").start()

    def _save(self, job):
        folder = self.directory / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder / "job.tmp"
        temp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        temp.replace(folder / "job.json")

    def update(self, job_id, **changes):
        with self.lock:
            self.jobs[job_id].update(changes)
            self._save(self.jobs[job_id])

    def create(self, data):
        kind, files = data.get("kind"), data.get("files")
        if kind not in {"photos", "video", "mixed"} or not isinstance(files, list):
            raise ValueError("영상 1개, 사진 묶음, 또는 영상 1개와 사진을 함께 선택해 주세요.")
        is_video = lambda e: isinstance(e, dict) and isinstance(e.get("name"), str) and Path(e["name"]).suffix.lower() in VIDEO_EXT
        videos = sum(is_video(e) for e in files)
        if kind == "mixed":  # 영상으로 집을 훑고, 덜 찍힌 곳을 사진으로 보충
            if videos != 1 or not 3 <= len(files)-1 <= 150:
                raise ValueError("영상 1개와 보충 사진 3~150장을 함께 선택해 주세요.")
        elif not (3 <= len(files) <= 150 if kind == "photos" else len(files) == 1):
            raise ValueError("사진은 3~150장, 영상은 1개를 선택해 주세요.")
        if data.get("space_type", "unknown") not in SPACE_TYPES:
            raise ValueError("공간 종류를 확인해 주세요.")
        normalized, total = [], 0
        for i, entry in enumerate(files):
            if not isinstance(entry, dict):
                raise ValueError("선택한 파일 정보를 확인해 주세요.")
            name, size = entry.get("name"), entry.get("size")
            if not isinstance(name, str) or not name or len(name) > 180 or any(c in name for c in '/\\\x00:') or name in {".", ".."}:
                raise ValueError("파일 이름에 사용할 수 없는 문자가 있어요.")
            suffix = Path(name).suffix.lower()
            allowed = PHOTO_EXT | VIDEO_EXT if kind == "mixed" else PHOTO_EXT if kind == "photos" else VIDEO_EXT
            if suffix not in allowed:
                raise ValueError("지원하지 않는 사진 또는 영상 형식이에요.")
            limit = 2 * 1024**3 if suffix in VIDEO_EXT else 30 * 1024**2
            if type(size) is not int or not 0 < size <= limit:
                raise ValueError("사진은 장당 30MB, 영상은 2GB 이하로 넣어 주세요.")
            total += size
            normalized.append({"name": name, "size": size, "stored": f"{i:03d}_{name}", "uploaded": False})
        if total > 2 * 1024**3:
            raise ValueError("한 작업의 총 파일 크기는 2GB 이하로 넣어 주세요.")
        job_id = uuid.uuid4().hex
        title = str(data.get("title", "새 공간")).strip()[:80] or "새 공간"
        job = {"id": job_id, "title": title, "kind": kind, "files": normalized,
               "space_type": data.get("space_type", "unknown"), "splat": bool(data.get("splat", True)),
               "status": "uploading", "progress": 0, "message": "파일을 이 PC로 전송하는 중이에요.",
               "created": time.time(), "messages": [], "scene": f"capture-{job_id[:12]}"}
        with self.lock:
            if sum(j["status"] in {"queued", "processing"} for j in self.jobs.values()) >= 8:
                raise ValueError("대기 중인 작업이 많아요. 현재 작업이 끝난 뒤 다시 넣어 주세요.")
            self.jobs[job_id] = job
            (self.directory / job_id / "input").mkdir(parents=True)
            self._save(job)
        return job_id

    def upload(self, job_id, index, stream, length):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] != "uploading" or not 0 <= index < len(job["files"]):
                raise ValueError("현재 작업에는 파일을 추가할 수 없어요.")
            entry = dict(job["files"][index])
        if length != entry["size"]:
            raise ValueError("선택한 파일 크기와 전송 크기가 달라요.")
        target = self.directory / job_id / "input" / entry["stored"]
        temp = target.with_name(target.name + ".upload")
        left = length
        with temp.open("wb") as output:
            while left:
                chunk = stream.read(min(left, 1024**2))
                if not chunk:
                    raise ValueError("파일 전송이 끊겼어요. 다시 선택해 주세요.")
                output.write(chunk)
                left -= len(chunk)
        temp.replace(target)
        with self.lock:
            self.jobs[job_id]["files"][index]["uploaded"] = True
            self._save(self.jobs[job_id])

    def start(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] != "uploading" or not all(f["uploaded"] for f in job["files"]):
                raise ValueError("파일 전송이 모두 끝난 뒤 실행해 주세요.")
            if job["space_type"] == "outdoor":
                self.update(job_id, status="needs_review", message="실외로 지정한 자료는 실내 도면 생성에서 제외했어요. 원본은 이 PC에 보관돼요.")
                return
            self.update(job_id, status="queued", message="분석 순서를 기다리는 중이에요.")
            self.queue.put(job_id)

    def public(self, job_id):
        with self.lock:
            j = self.jobs[job_id]
            return {k: j[k] for k in ("id", "title", "kind", "space_type", "status", "progress", "message", "created", "scene")} | {
                "files": [{"name": f["name"], "uploaded": f["uploaded"]} for f in j["files"]],
                "messages": j["messages"][-12:], "result": j.get("result")}

    def _work(self):
        while True:
            job_id = self.queue.get()
            try:
                self._process(job_id)
            except Exception as error:
                self.update(job_id, status="failed", message=f"분석을 마치지 못했어요: {error}")
            finally:
                self.queue.task_done()

    def _process(self, job_id):
        job = self.jobs[job_id]
        self.update(job_id, status="processing", progress=5, message="입력 자료를 확인하고 있어요.")
        folder = self.directory / job_id
        source = folder / "input"
        extra = []
        if job["kind"] == "video":
            source = source / job["files"][0]["stored"]
        elif job["kind"] == "mixed":
            photo_dir = folder / "input_photos"
            photo_dir.mkdir(exist_ok=True)
            video = next(f for f in job["files"] if Path(f["name"]).suffix.lower() in VIDEO_EXT)
            for f in job["files"]:
                if f is not video and not (photo_dir / f["stored"]).exists():
                    shutil.copyfile(source / f["stored"], photo_dir / f["stored"])
            source = source / video["stored"]
            extra = ["--photos", str(photo_dir)]
        if job["kind"] != "photos":
            extra += ["--max-frames", "800"]  # 집 전체 걷기 영상(2~3분, 초당 5장)
        cmd = [sys.executable, "-u", "-m", "spacecapture", "photos" if job["kind"] == "photos" else "process",
               str(source), "--name", job["scene"], "--steps", "10000", "--max-resolution", "960", "--max-splats", "300000"] + extra
        if not job["splat"]:
            cmd.append("--no-splat")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1")
        child = subprocess.Popen(cmd, cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace",
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        assert child.stdout is not None
        with (folder / "process.log").open("w", encoding="utf-8") as log, child.stdout:
            for line in child.stdout:
                log.write(line)
                log.flush()
                line = line.strip()
                if not line:
                    continue
                with self.lock:
                    job["messages"] = (job["messages"] + [line])[-80:]
                stage = re.search(r"\[(\d)/5\]\s*(.*)", line)
                if stage:
                    self.update(job_id, progress=int(stage[1]) * 17, message=stage[2])
                elif "학습 중" in line:
                    self.update(job_id, message=line)
        code = child.wait()
        scene_dir = self.root / "scenes" / job["scene"]
        if code != 0 or not (scene_dir / "scene.json").is_file():
            message = next((s for s in reversed(job["messages"]) if s.startswith("처리 실패:")),
                           "사진의 겹침과 특징이 부족해 복원하지 못했어요. 같은 공간을 다른 위치에서 찍은 자료를 넣어 주세요.")
            self.update(job_id, status="failed", message=message)
            return
        from .pipeline import load_scene, _write_scene
        scene = load_scene(scene_dir)
        scene["source"]["declared_space_type"] = job["space_type"]
        scene["source"]["title"] = job["title"]
        if job["splat"] and not scene["files"].get("splat"):
            scene["quality"]["warnings"].append("기하 모델은 만들었지만 실사 3D 학습은 완료하지 못했어요.")
        _write_scene(scene_dir, scene)
        result = {"scene": job["scene"], "assessment": scene["assessment"],
                  "layout": scene.get("layout"), "connectivity": scene.get("connectivity"), "property_graph":scene.get("property_graph"),"plan":scene.get("plan"),
                  "registered": scene["quality"]["registered"], "input": scene["quality"]["frames"],
                  "splat": bool(scene["files"].get("splat")), "warnings": scene["quality"]["warnings"]}
        self.update(job_id, status="complete", progress=100, message="분석을 마쳤어요. 검사 결과와 구조 후보를 확인해 주세요.", result=result)


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store):
        self.store = store
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _check_local(self):
        host = urlparse("http://" + self.headers.get("Host", "")).hostname
        origin = self.headers.get("Origin")
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise PermissionError("이 서버는 이 PC에서만 사용할 수 있어요.")
        if origin and urlparse(origin).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise PermissionError("외부 사이트에서는 작업을 요청할 수 없어요.")

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= 256 * 1024:
            raise ValueError("요청 정보가 너무 커요.")
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            raise ValueError("요청 정보의 형식을 확인해 주세요.") from error
        if not isinstance(data, dict):
            raise ValueError("요청 형식을 확인해 주세요.")
        return data

    def _handle(self):
        try:
            self._check_local()
            url = unquote(urlparse(self.path).path)
            parts = url.strip("/").split("/")
            store = self.server.store
            if url == "/api/health" and self.command == "GET":
                return self._json({"status": "ok", "message": "로컬 분석 서버 연결됨"})
            if url == "/scenes/" and self.command == "GET":
                items = []
                for file in (store.root / "scenes").glob("*/scene.json"):
                    try:
                        scene = json.loads(file.read_text(encoding="utf-8"))
                        items.append({"name": file.parent.name, "room": scene["room"],
                                      "splat": bool(scene["files"].get("splat")), "created": scene["created"]})
                    except (OSError, ValueError, KeyError):
                        continue
                return self._json(sorted(items, key=lambda item: item["created"], reverse=True))
            if url == "/api/jobs":
                if self.command == "POST":
                    return self._json(store.public(store.create(self._read_json())), 201)
                if self.command == "GET":
                    with store.lock:
                        ordered = sorted(store.jobs, key=lambda k: store.jobs[k]["created"], reverse=True)[:30]
                        return self._json([store.public(key) for key in ordered])
            if len(parts) >= 3 and parts[:2] == ["api", "jobs"]:
                job_id = parts[2]
                if not re.fullmatch(r"[0-9a-f]{32}", job_id) or job_id not in store.jobs:
                    return self._json({"error": "작업을 찾을 수 없어요."}, 404)
                if len(parts) == 3 and self.command == "GET":
                    return self._json(store.public(job_id))
                if len(parts) == 4 and parts[3] == "start" and self.command == "POST":
                    self._read_json()
                    store.start(job_id)
                    return self._json(store.public(job_id), 202)
                if len(parts) == 5 and parts[3] == "files" and parts[4].isdigit() and self.command == "PUT":
                    if "Content-Length" not in self.headers:
                        return self._json({"error": "파일 크기 정보가 필요해요."}, 411)
                    store.upload(job_id, int(parts[4]), self.rfile, int(self.headers["Content-Length"]))
                    return self._json(store.public(job_id))
            if self.command != "GET":
                return self._json({"error": "지원하지 않는 요청이에요."}, 404)
            if parts[0] == "scenes":
                if len(parts) != 3 or parts[1] in {"", ".", ".."}:
                    return self._json({"error": "장면 파일을 찾을 수 없어요."}, 404)
                allowed = {".json", ".ply", ".spz", ".svg"}
                if parts[2].startswith("thumb_"):
                    allowed.add(".jpg")
                if Path(parts[2]).suffix not in allowed:
                    return self._json({"error": "공개된 장면 파일이 아니에요."}, 404)
                base = (store.root / "scenes").resolve()
                target = (base / parts[1] / parts[2]).resolve()
            else:
                base = (store.root / "viewer/dist").resolve()
                target = (base / (url.lstrip("/") or "index.html")).resolve()
            if not target.is_relative_to(base) or not target.is_file():
                return self._json({"error": "파일을 찾을 수 없어요. 뷰어는 먼저 npm run build로 빌드해 주세요."}, 404)
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(target.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with target.open("rb") as source:
                while chunk := source.read(1024**2):
                    self.wfile.write(chunk)
        except PermissionError as error:
            self._json({"error": str(error)}, 403)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            self._json({"error": str(error) if isinstance(error, ValueError) else "요청 정보를 확인해 주세요."}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError:
            self._json({"error": "파일을 처리하지 못했어요. 남은 공간과 전송 상태를 확인해 주세요."}, 500)

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle


def serve(port: int = 8765):
    server = LocalServer(("127.0.0.1", port), JobStore())
    print(f"공간 분석 앱: http://127.0.0.1:{port} (이 PC에서만 사용)", flush=True)
    server.serve_forever()
