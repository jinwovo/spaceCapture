"""천장 모델의 학습 실행 폴더·검증 기록·활성화를 다루는 도구용 모듈.

학습 결과는 항상 별도 실행 폴더(run)에 저장하고, 검증을 모두 통과한 경우에만
models/에 설치한다. 학습 보고서·사진 회귀 보고서·활성화 기록은 같은
추론 설정 지문(가중치 SHA·임계값·특징 버전/순서)으로 묶이며, 시험 자료와
입력 장면의 해시도 함께 기록·재확인한다.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .ceiling_model import (DEFAULT_MODEL, FEATURES, VERSION, audit_dataset, inference_fingerprint,
                            load_model, predict)

ROOT = Path(__file__).resolve().parents[1]
LEARNING = ROOT/"artifacts/ceiling-learning"
RUNS = LEARNING/"runs"
MODEL_NAME = "ceiling-mlp-v1"
SPLITS = ("train", "validation", "test")
DATA_FILES = ("dataset.json",)+tuple(f"{s}{ext}" for s in SPLITS for ext in (".npz", "-clouds.json"))


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def run_model_path(run) -> Path:
    return Path(run)/"model"/f"{MODEL_NAME}.npz"


def new_run_dir(base=RUNS) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(base)/stamp
    index = 1
    while path.exists():
        index += 1
        path = Path(base)/f"{stamp}-{index}"
    path.mkdir(parents=True)
    return path


def ensure_not_installed_location(path):
    """학습 산출물이 현재 설치 모델 폴더에 직접 써지는 것을 막는다."""
    if Path(path).resolve().parent == DEFAULT_MODEL.resolve().parent:
        raise RuntimeError("학습 결과를 설치 모델 폴더(models/)에 직접 저장할 수 없어요. 실행 폴더를 사용해 주세요.")


def data_hashes(run) -> dict:
    run = Path(run)
    missing = [name for name in DATA_FILES if not (run/name).exists()]
    if missing:
        raise RuntimeError(f"학습 실행 폴더에 자료가 없어요: {', '.join(missing)}")
    return {name: file_sha256(run/name) for name in DATA_FILES}


def load_split(run, split) -> dict:
    run = Path(run)
    with np.load(run/f"{split}.npz", allow_pickle=False) as arrays:
        data = {k: arrays[k].copy() for k in arrays.files}
    data["descriptors"] = json.loads((run/f"{split}-clouds.json").read_text(encoding="utf-8"))
    return data


def metrics(data, scores, threshold, legacy=False):
    accepted = np.asarray(scores) >= threshold
    if legacy:
        tops = np.array([d["camera_top"] for d in data["descriptors"]])
        accepted &= data["height"] <= tops[data["cloud"]]+2.5
    y = data["y"] > 0
    tp = int(np.sum(y & accepted)); fp = int(np.sum(~y & accepted)); fn = int(np.sum(y & ~accepted))
    errors = []; correct = wrong = abstained = 0
    for i, descriptor in enumerate(data["descriptors"]):
        keep = np.flatnonzero((data["cloud"] == i) & accepted)
        if not len(keep):
            abstained += 1; continue
        j = keep[np.argmax(data["height"][keep])]
        if y[j]:
            correct += 1; errors.append(abs(float(data["height"][j]-descriptor["truth_height"]))*100)
        else:
            wrong += 1
    total = len(data["descriptors"])
    return {"candidate_precision": round(tp/max(1, tp+fp), 4), "candidate_recall": round(tp/max(1, tp+fn), 4),
            "clouds": total, "correct_detections": correct, "wrong_detections": wrong, "abstained": abstained,
            "wrong_detection_rate": round(wrong/max(1, total), 4),
            "detectable_clouds": sum(d["detectable"] for d in data["descriptors"]),
            "correct_height_mean_error_cm": round(float(np.mean(errors)), 3) if errors else None}


def model_metadata(model_path, threshold, manifest, report_path, extra=None) -> dict:
    sha = file_sha256(model_path)
    meta = {"id": MODEL_NAME, "license": "Apache-2.0", "feature_version": VERSION, "features": FEATURES,
            "sha256": sha, "threshold": float(threshold),
            "inference_fingerprint": inference_fingerprint(MODEL_NAME, sha, float(threshold)),
            "enabled_by_default": False, "neural_training_performed": True, "real_photo_validated": False,
            "training_report": rel(report_path),
            "source_commits": sorted(set(s["commit"] for s in manifest["sources"].values())),
            "note": "천장 후보의 기하 특징 신경망. 합성 점군 학습이며 사진 의미 분류·집 전체 생성 모델이 아닙니다."}
    meta.update(extra or {})
    return meta


def scene_inputs(name) -> dict:
    """사진 회귀 추론에 실제로 들어간 입력 파일의 해시."""
    folder = ROOT/"scenes"/name
    scene = json.loads((folder/"scene.json").read_text(encoding="utf-8"))
    files = {"points.ply": file_sha256(folder/"points.ply")}
    if scene.get("files", {}).get("splat"):
        files["splat.ply"] = file_sha256(folder/scene["files"]["splat"])
    room = scene["room"]
    return {"files": files, "camera_top": max(c["position"][1] for c in scene["cameras"]),
            "box": [room["min"][0], room["max"][0], room["min"][2], room["max"][2]]}


def refresh_training_report(run) -> dict:
    """고정 모델로 점수를 재계산해 보고서를 보강한다. 학습 당시 모델 해시는 덮어쓰지 않는다."""
    run = Path(run)
    report_path = run/"result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model = run_model_path(run)
    weights, meta = load_model(model)
    actual_sha = file_sha256(model)
    if report.get("model_sha256") != actual_sha:
        raise RuntimeError("학습 보고서의 모델 해시와 실행 폴더의 가중치가 달라요. 보고서를 덮어쓰지 않아요.")
    if float(report.get("threshold", -1)) != float(meta["threshold"]):
        raise RuntimeError("학습 보고서의 임계값과 모델 기록의 임계값이 달라요.")
    fingerprint = inference_fingerprint(MODEL_NAME, report["model_sha256"], report["threshold"])
    if report.get("inference_fingerprint", fingerprint) != fingerprint or meta["inference_fingerprint"] != fingerprint:
        raise RuntimeError("학습 보고서와 모델 기록의 추론 설정 지문이 달라요.")
    report["inference_fingerprint"] = fingerprint
    report.setdefault("feature_version", VERSION)
    report.setdefault("features", FEATURES)
    hashes = data_hashes(run)
    if "data_files" in report and report["data_files"] != hashes:
        raise RuntimeError("학습·검증·시험 자료의 해시가 학습 보고서와 달라요.")
    if "data_files" not in report:
        report["data_files"] = hashes
        report["data_files_recorded_by"] = "report-backfill"
    report["split_audit"] = audit_dataset(json.loads((run/"dataset.json").read_text(encoding="utf-8")))
    report["model_bytes"] = model.stat().st_size
    report["trainable_parameters"] = int(sum(v.size for k, v in weights.items() if k.startswith(("w", "b"))))
    for split in ("validation", "test"):
        data = load_split(run, split)
        if metrics(data, predict(data["X"], weights), meta["threshold"]) != report[split]:
            raise RuntimeError("저장된 학습 보고서와 고정 모델의 점수가 달라요.")
        X = data["X"]
        rule = (X[:, 7] > .75) & (X[:, 11] > .75) & (X[:, 17] < .08)
        report[f"normal_rule_{split}"] = metrics(data, rule.astype(float), .5)
    report["normal_rule_definition"] = {"normal_y_mean_12_min": .75, "normal_y_mean_24_min": .75,
        "points_above_fraction_max": .08,
        "note": "시험에 맞춰 최적화하지 않은 참고 규칙입니다. 최적 기하 알고리즘의 성능을 뜻하지 않습니다."}
    report["benchmark_scope"] = "시험 집에서 만든 점군 변형. 같은 집의 변형은 독립 실제 주택 표본이 아닙니다."
    report["legacy_baseline_scope"] = "동일 후보 집합에 기존 카메라 위 2.5m 상한을 적용한 모사 비교"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def check_activation(run) -> dict:
    """활성화 전 모든 기록이 같은 추론 설정·자료·입력 장면을 가리키는지 검사한다."""
    run = Path(run)
    model = run_model_path(run)
    try:
        weights, meta = load_model(model)
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"실행 폴더의 모델을 확인할 수 없어요: {error}") from error
    fingerprint = meta["inference_fingerprint"]
    report = json.loads((run/"result.json").read_text(encoding="utf-8"))
    if report.get("inference_fingerprint") != fingerprint or report.get("model_sha256") != meta["sha256"] \
            or float(report.get("threshold", -1)) != float(meta["threshold"]):
        raise RuntimeError("학습 보고서와 현재 모델의 추론 설정(가중치·임계값·특징)이 달라요.")
    if report.get("features", FEATURES) != FEATURES or report.get("feature_version", VERSION) != VERSION:
        raise RuntimeError("학습 보고서의 특징 버전·순서가 현재 코드와 달라요.")
    hashes = data_hashes(run)
    if report.get("data_files") != hashes:
        raise RuntimeError("학습·검증·시험 자료의 해시가 학습 보고서와 달라요.")
    audit = audit_dataset(json.loads((run/"dataset.json").read_text(encoding="utf-8")))
    recomputed = {}
    for split in ("validation", "test"):
        data = load_split(run, split)
        recomputed[split] = metrics(data, predict(data["X"], weights), meta["threshold"])
        if recomputed[split] != report[split]:
            raise RuntimeError(f"현재 추론 설정으로 다시 계산한 {split} 점수가 학습 보고서와 달라요.")
    test, base = recomputed["test"], report["uncapped_geometry_test"]
    if test["wrong_detections"] > base["wrong_detections"] or test["correct_detections"] < base["correct_detections"]:
        raise RuntimeError("별도 시험에서 기하 기준보다 나빠 모델을 활성화하지 않아요.")
    if test["wrong_detections"] > 0:
        raise RuntimeError("별도 시험 점군에서 잘못된 최종 높이 검출이 있어 활성화하지 않아요.")
    regression_path = run/"sfm-regression.json"
    if not regression_path.exists():
        raise RuntimeError("이 실행 폴더의 사진 복원 회귀 검사가 없어요. validate-ceiling-model.py를 먼저 실행해 주세요.")
    regression = json.loads(regression_path.read_text(encoding="utf-8"))
    if regression.get("inference_fingerprint") != fingerprint:
        raise RuntimeError("사진 회귀 결과의 추론 설정 지문이 현재 모델과 달라요. 현재 설정으로 다시 검사해 주세요.")
    cases = regression.get("cases", [])
    checked = set()
    for case in cases:
        if case.get("analysis", {}).get("inference_fingerprint") != fingerprint:
            raise RuntimeError("사진 회귀 사례 중 다른 추론 설정으로 계산한 결과가 있어요.")
        name = case["scene"]
        if name not in checked:
            current = scene_inputs(name)
            recorded = regression.get("inputs", {}).get(name)
            if recorded != current:
                raise RuntimeError(f"사진 회귀 입력 장면 {name}의 해시·상자가 기록과 달라요. 다시 검사해 주세요.")
            checked.add(name)
        if name == "synth" and case["model_height_m"] is not None:
            raise RuntimeError("천장 관측이 부족한 기존 장면에서 높이를 확정해 활성화를 보류해요.")
        if case["model_height_m"] is not None and abs(case["error_cm"]) > 5:
            raise RuntimeError("사진 복원 회귀에서 잘못된 높이를 확정해 활성화를 보류해요.")
    tests = [c for c in cases if c["scene"].startswith("procthor-test-")]
    if not tests:
        raise RuntimeError("시험 구조의 실제 사진 복원 검사가 없어요.")
    for name, truth in regression.get("truth_files", {}).items():
        if file_sha256(ROOT/truth["path"]) != truth["sha256"]:
            raise RuntimeError(f"{name}의 평가용 정답 파일 해시가 기록과 달라요.")
    if not any(c["scene"] in regression.get("truth_files", {}) for c in tests):
        raise RuntimeError("시험 사진 복원 사례의 정답 파일 해시가 기록되지 않았어요.")
    return {"inference_fingerprint": fingerprint, "model_sha256": meta["sha256"], "threshold": meta["threshold"],
            "feature_version": VERSION, "features": FEATURES, "run": rel(run),
            "training_report": {"path": rel(run/"result.json"), "sha256": file_sha256(run/"result.json")},
            "sfm_regression": {"path": rel(regression_path), "sha256": file_sha256(regression_path)},
            "data_files": hashes, "scene_inputs": regression["inputs"], "truth_files": regression.get("truth_files", {}),
            "split_audit": audit, "test": test, "validation": recomputed["validation"],
            "actual_sfm_cases": len(checked), "real_photo_validated": False, "requires_observed_height": True}


def _atomic_write(path, data: bytes):
    temp = Path(str(path)+".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def activate(run, installed=DEFAULT_MODEL, pointer=None, backup_root=None) -> dict:
    """검증을 모두 통과한 실행 폴더의 모델만 설치한다. 실패하면 현재 설치 모델은 그대로다."""
    run = Path(run)
    record = check_activation(run)
    installed = Path(installed)
    default_install = installed.resolve() == DEFAULT_MODEL.resolve()
    pointer = pointer if pointer is not None else (LEARNING/"activation.json" if default_install else None)
    backup_root = Path(backup_root) if backup_root is not None else LEARNING/"previous-models"
    source = run_model_path(run)
    meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    record["activated_at"] = now
    if installed.exists():
        backup = backup_root/now.replace(":", "")
        backup.mkdir(parents=True, exist_ok=True)
        for path in (installed, installed.with_suffix(".json")):
            if path.exists():
                shutil.copy2(path, backup/path.name)
        record["previous_model_backup"] = rel(backup)
    meta.update(enabled_by_default=True, deployment_scope="experimental_ceiling_candidate_filter",
                activation_note="합성 점군 시험과 사진 복원 회귀를 통과한 관측 높이 후보 판별. 실사 일반화는 미검증.",
                activation={"inference_fingerprint": record["inference_fingerprint"], "activated_at": now,
                            "record": rel(run/"activation.json")})
    installed.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(installed, source.read_bytes())
    _atomic_write(installed.with_suffix(".json"), json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
    license_file = source.parent/"LICENSE.procthor"
    if license_file.exists():
        shutil.copyfile(license_file, installed.parent/"LICENSE.procthor")
    load_model(installed, require_active=True)
    record["enabled"] = True
    record["installed_model"] = rel(installed)
    text = json.dumps(record, ensure_ascii=False, indent=2)
    (run/"activation.json").write_text(text, encoding="utf-8")
    if pointer is not None:
        Path(pointer).write_text(text, encoding="utf-8")
    return record
