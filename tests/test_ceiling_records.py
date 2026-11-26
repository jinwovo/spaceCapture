"""천장 모델의 설치 상태·추론 설정 지문·활성화 검사 회귀 시험."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spacecapture import ceiling_model
from spacecapture.ceiling_model import DEFAULT_MODEL, inference_fingerprint, load_model, predict
from spacecapture.ceiling_records import (DATA_FILES, LEARNING, activate, check_activation, ensure_not_installed_location,
                                          load_split, metrics, refresh_training_report, run_model_path)
from spacecapture.room import detect_room

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = DATA_FILES+("result.json", "sfm-regression.json")
HAS_RUN = run_model_path(LEARNING).exists() and all((LEARNING/n).exists() for n in RUN_FILES)
CAMERAS = np.array([[2, 1.45, 2], [3, 1.45, 2], [2, 1.45, 3]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def high_ceiling_points():
    return np.random.default_rng(3).uniform([0, 0, 0], [6, 4.3, 5], (1000, 3))


@unittest.skipUnless(HAS_RUN, "현재 v1 실행 폴더 기록이 필요해요.")
class ActivationRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)/"run"
        self.run.mkdir()
        for name in RUN_FILES:
            shutil.copy2(LEARNING/name, self.run/name)
        shutil.copytree(LEARNING/"model", self.run/"model")
        self.meta_path = run_model_path(self.run).with_suffix(".json")

    def tearDown(self):
        self.temp.cleanup()

    def set_threshold(self, value, report_too=False):
        meta = read(self.meta_path)
        meta["threshold"] = value
        meta["inference_fingerprint"] = inference_fingerprint(meta["id"], meta["sha256"], value)
        write(self.meta_path, meta)
        if report_too:
            report = read(self.run/"result.json")
            report["threshold"] = value
            report["inference_fingerprint"] = meta["inference_fingerprint"]
            write(self.run/"result.json", report)

    def test_unchanged_run_passes(self):
        record = check_activation(self.run)
        self.assertEqual(record["inference_fingerprint"], read(self.meta_path)["inference_fingerprint"])
        self.assertEqual(record["test"]["wrong_detections"], 0)
        self.assertTrue(any(name.startswith("procthor-test-") for name in record["truth_files"]))
        self.assertIn("points.ply", record["scene_inputs"]["synth"]["files"])

    def test_threshold_zero_with_same_weights_is_rejected(self):
        # 같은 가중치에서 임계값만 0으로 바꾸면 시험 점군의 최종 오검출이 크게 늘어난다.
        weights, _ = load_model(run_model_path(self.run))
        data = load_split(self.run, "test")
        self.assertGreater(metrics(data, predict(data["X"], weights), 0.0)["wrong_detections"], 0)
        self.set_threshold(0.0)
        with self.assertRaisesRegex(RuntimeError, "추론 설정"):
            check_activation(self.run)

    def test_threshold_edit_in_report_too_fails_on_recomputed_scores(self):
        self.set_threshold(0.0, report_too=True)
        with self.assertRaisesRegex(RuntimeError, "점수"):
            check_activation(self.run)

    def test_regression_from_other_configuration_is_rejected(self):
        regression = read(self.run/"sfm-regression.json")
        regression["inference_fingerprint"] = "0"*64
        write(self.run/"sfm-regression.json", regression)
        with self.assertRaisesRegex(RuntimeError, "사진 회귀"):
            check_activation(self.run)

    def test_changed_test_data_is_rejected(self):
        clouds = read(self.run/"test-clouds.json")
        clouds[0]["truth_height"] += .5
        write(self.run/"test-clouds.json", clouds)
        with self.assertRaisesRegex(RuntimeError, "자료의 해시"):
            check_activation(self.run)

    def test_changed_input_scene_or_truth_is_rejected(self):
        regression = read(self.run/"sfm-regression.json")
        original = json.loads(json.dumps(regression))
        regression["inputs"]["synth"]["files"]["points.ply"] = "0"*64
        write(self.run/"sfm-regression.json", regression)
        with self.assertRaisesRegex(RuntimeError, "입력 장면"):
            check_activation(self.run)
        name = next(iter(original["truth_files"]))
        original["truth_files"][name]["sha256"] = "0"*64
        write(self.run/"sfm-regression.json", original)
        with self.assertRaisesRegex(RuntimeError, "정답 파일"):
            check_activation(self.run)

    def test_activation_installs_only_after_checks(self):
        installed = Path(self.temp.name)/"models/ceiling-mlp-v1.npz"
        installed.parent.mkdir()
        pointer = Path(self.temp.name)/"activation.json"
        backups = Path(self.temp.name)/"backup"
        record = activate(self.run, installed, pointer, backups)
        _, meta = load_model(installed, require_active=True)
        self.assertEqual(meta["inference_fingerprint"], record["inference_fingerprint"])
        before = {p.name: sha(p) for p in installed.parent.iterdir()}
        self.set_threshold(0.0)
        with self.assertRaises(RuntimeError):
            activate(self.run, installed, pointer, backups)
        self.assertEqual(before, {p.name: sha(p) for p in installed.parent.iterdir()})

    def test_report_refresh_does_not_overwrite_model_hash(self):
        report = read(self.run/"result.json")
        report["model_sha256"] = "1"*64
        write(self.run/"result.json", report)
        with self.assertRaisesRegex(RuntimeError, "모델 해시"):
            refresh_training_report(self.run)
        self.assertEqual(read(self.run/"result.json")["model_sha256"], "1"*64)
        shutil.copy2(LEARNING/"result.json", self.run/"result.json")
        refreshed = refresh_training_report(self.run)
        self.assertEqual(refreshed["model_sha256"], read(LEARNING/"result.json")["model_sha256"])

    def test_retraining_writes_new_run_without_touching_installed_model(self):
        installed = {p.name: sha(p) for p in DEFAULT_MODEL.parent.iterdir() if p.is_file()}
        out = Path(self.temp.name)/"new-run"
        result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT/"tools/train-ceiling-model.py"),
                                 "--reuse-data", str(self.run), "--epochs", "2", "--run-dir", str(out)],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(installed, {p.name: sha(p) for p in DEFAULT_MODEL.parent.iterdir() if p.is_file()})
        meta = read(run_model_path(out).with_suffix(".json"))
        self.assertIs(meta["enabled_by_default"], False)
        self.assertEqual(read(out/"result.json")["inference_fingerprint"], meta["inference_fingerprint"])
        # 짧게 학습한 새 실행은 검사 없이 설치되지 않는다 (점수 또는 사진 회귀 부재로 거절).
        with self.assertRaises(RuntimeError):
            check_activation(out)


@unittest.skipUnless(DEFAULT_MODEL.exists(), "설치된 천장 모델이 필요해요.")
class InstalledModelStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model = Path(self.temp.name)/"ceiling-mlp-v1.npz"
        shutil.copy2(DEFAULT_MODEL, self.model)
        self.meta = self.model.with_suffix(".json")
        shutil.copy2(DEFAULT_MODEL.with_suffix(".json"), self.meta)
        self.patch = mock.patch.object(ceiling_model, "DEFAULT_MODEL", self.model)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def room(self):
        return detect_room(high_ceiling_points(), CAMERAS)

    def assert_assumed(self):
        room = self.room()
        self.assertEqual(room["height_source"], "assumed")
        self.assertEqual(room["ceiling_analysis"]["status"], "model_unavailable")

    def test_installed_model_is_used(self):
        room = self.room()
        self.assertEqual(room["ceiling_analysis"]["inference_fingerprint"], read(self.meta)["inference_fingerprint"])

    def test_missing_metadata_with_weights_assumes_height(self):
        self.meta.unlink()
        self.assertEqual(ceiling_model.active_model(), self.model)
        self.assert_assumed()

    def test_damaged_metadata_assumes_height(self):
        for text in ("{깨짐", "[1,2]", json.dumps({"enabled_by_default": "true"})):
            self.meta.write_text(text, encoding="utf-8")
            self.assert_assumed()

    def test_threshold_edit_in_installed_metadata_assumes_height(self):
        meta = read(self.meta)
        meta["threshold"] = 0.0
        write(self.meta, meta)
        self.assert_assumed()
        # 지문까지 다시 계산해도 활성화 기록과 다르면 사용하지 않는다.
        meta["inference_fingerprint"] = inference_fingerprint(meta["id"], meta["sha256"], 0.0)
        write(self.meta, meta)
        self.assert_assumed()

    def test_unactivated_metadata_assumes_height(self):
        meta = read(self.meta)
        meta.pop("activation")
        write(self.meta, meta)
        self.assert_assumed()

    def test_intentionally_disabled_model_uses_geometry_path(self):
        meta = read(self.meta)
        meta["enabled_by_default"] = False
        write(self.meta, meta)
        self.assertIsNone(ceiling_model.active_model())
        self.assertIsNone(self.room()["ceiling_analysis"])

    def test_no_model_files_uses_geometry_path(self):
        self.meta.unlink(); self.model.unlink()
        self.assertIsNone(ceiling_model.active_model())


class TrainingLocationTests(unittest.TestCase):
    def test_training_output_cannot_target_installed_folder(self):
        with self.assertRaisesRegex(RuntimeError, "설치 모델 폴더"):
            ensure_not_installed_location(DEFAULT_MODEL)


if __name__ == "__main__":
    unittest.main()
