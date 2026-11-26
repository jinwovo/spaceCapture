"""Photo-real 3D (Gaussian splatting) with Brush, run as an external program.

Brush (Apache-2.0) trains on WebGPU, so it also runs on AMD / Intel GPUs.
Download: https://github.com/ArthurBrussee/brush/releases  ->  tools/brush-*/brush_app.exe
or point SPACECAPTURE_BRUSH at the executable.
"""

from __future__ import annotations

import os
import json
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pycolmap

REPO = Path(__file__).resolve().parent.parent


def find_brush() -> Path | None:
    env = os.environ.get("SPACECAPTURE_BRUSH")
    candidates = [Path(env)] if env else []
    tools = REPO / "tools"
    if tools.is_dir():
        candidates += sorted(tools.glob("brush*/brush_app*"), reverse=True)
    for c in candidates:
        if c.is_file():
            return c
    found = shutil.which("brush_app") or shutil.which("brush")
    return Path(found) if found else None


def prepare_dataset(sparse_dir: Path, image_dir: Path, out_dir: Path) -> Path:
    """Undistort images to pinhole cameras (what splatting expects), COLMAP layout."""
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    pycolmap.undistort_images(out_dir, sparse_dir, image_dir)
    return out_dir


def train(dataset_dir: Path, export_dir: Path, steps: int = 15000, log=print,
          max_resolution: int = 1280, max_splats: int = 500000) -> Path:
    if min(steps, max_resolution, max_splats) <= 0:
        raise ValueError("학습 스텝, 해상도, 최대 스플랫 수는 양수여야 해요.")
    exe = find_brush()
    if exe is None:
        raise RuntimeError(
            "Brush를 찾지 못했어요. https://github.com/ArthurBrussee/brush/releases 에서 받아 "
            "tools/brush-<버전>/ 폴더에 풀거나, SPACECAPTURE_BRUSH 환경변수로 경로를 지정해 주세요."
        )
    shutil.rmtree(export_dir, ignore_errors=True)
    export_dir.mkdir(parents=True)
    every = min(steps, max(100, steps // 5))
    cmd = [str(exe), str(dataset_dir.resolve()),
           "--total-steps", str(steps),
           "--export-every", str(every),
           "--export-path", str(export_dir.resolve()),
           "--export-name", "splat_{iter}.ply",
           "--max-resolution", str(max_resolution),
           "--max-splats", str(max_splats)]
    log(f"  실행: {' '.join(cmd)}")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    last_report, tail = t0, []
    assert proc.stdout is not None
    lines = queue.Queue()

    def read_output():
        with proc.stdout:
            for line in proc.stdout:
                lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    # Brush may emit no lines when stdout is a pipe. Poll independently so the
    # Korean progress messages and checkpoint timings still reach the caller.
    checkpoints = {}
    peak_memory = 0
    try:
        with (export_dir / "brush.log").open("w", encoding="utf-8") as raw:
            ended = False
            while not ended:
                try:
                    line = lines.get(timeout=1)
                    if line is None:
                        ended = True
                    else:
                        raw.write(line)
                        raw.flush()
                        if line.strip():
                            tail = (tail + [line.strip()])[-15:]
                except queue.Empty:
                    pass
                now = time.monotonic()
                done = sorted(export_dir.glob("splat_*.ply"), key=lambda p: p.stat().st_mtime)
                for p in done:
                    checkpoints.setdefault(p.name, round(now - t0, 1))
                if os.name == "nt" and proc.poll() is None:
                    import ctypes
                    from ctypes import wintypes

                    class MemoryCounters(ctypes.Structure):
                        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                            (name, ctypes.c_size_t) for name in
                            ("peak_working_set", "working_set", "peak_paged", "paged",
                             "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]

                    counters = MemoryCounters()
                    counters.cb = ctypes.sizeof(counters)
                    if ctypes.windll.psapi.GetProcessMemoryInfo(
                            wintypes.HANDLE(int(proc._handle)), ctypes.byref(counters), counters.cb):
                        peak_memory = max(peak_memory, counters.peak_working_set)
                if now - last_report >= 30:
                    log(f"  학습 중... {now - t0:.0f}초 경과" +
                        (f", 최근 저장: {done[-1].name}" if done else ", 첫 저장 대기"))
                    last_report = now
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    elapsed = round(time.monotonic() - t0, 1)
    (export_dir / "training.json").write_text(json.dumps({
        "steps": steps, "max_resolution": max_resolution, "max_splats": max_splats,
        "elapsed_s": elapsed, "peak_working_set_mb": round(peak_memory / 1048576, 1),
        "checkpoints_s": checkpoints, "returncode": rc, "command": cmd,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    plys = [p for p in export_dir.glob("splat_*.ply")
            if p.stem.rsplit("_", 1)[-1].isdigit() and int(p.stem.rsplit("_", 1)[-1]) == steps]
    if rc != 0 or not plys:
        raise RuntimeError(f"Brush 학습이 실패했거나 최종 파일이 없어요 (종료 코드 {rc}). "
                           f"진단 기록: {export_dir / 'brush.log'}\n    " + "\n    ".join(tail))
    log(f"  학습 완료 ({elapsed:.1f}초 / {elapsed / 60:.1f}분)")
    return plys[-1]
