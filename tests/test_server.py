import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from spacecapture.server import JobStore, LocalServer


class UploadApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = JobStore(self.root, start_worker=False)
        self.server = LocalServer(("127.0.0.1", 0), self.store)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, url, data=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        try:
            conn.request(method, url, data, headers or {})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def create(self, space="office"):
        data = {"title": "회의실", "kind": "photos", "space_type": space, "splat": False,
                "files": [{"name": f"사진{i}.png", "size": 3} for i in range(3)]}
        code, job = self.request("POST", "/api/jobs", json.dumps(data))
        self.assertEqual(code, 201)
        return job

    def test_upload_then_queue_and_persist_original_files(self):
        job = self.create()
        job_id = job["id"]
        self.assertEqual(self.request("POST", f"/api/jobs/{job_id}/start", "{}")[0], 400)
        for i in range(3):
            self.assertEqual(self.request("PUT", f"/api/jobs/{job_id}/files/{i}", b"abc")[0], 200)
        code, started = self.request("POST", f"/api/jobs/{job_id}/start", "{}")
        self.assertEqual(code, 202)
        self.assertEqual(started["status"], "queued")
        self.assertEqual(started["space_type"], "office")
        files = list((self.store.directory / job_id / "input").glob("*.png"))
        self.assertEqual(len(files), 3)
        self.assertTrue(all(file.read_bytes() == b"abc" for file in files))
        recovered = JobStore(self.root, start_worker=False)
        self.assertEqual(recovered.public(job_id)["status"], "failed")

    def test_mixed_video_and_photos_job(self):
        files = [{"name": "집안걷기.mp4", "size": 10}] + [{"name": f"보충{i}.jpg", "size": 3} for i in range(3)]
        data = {"title": "집", "kind": "mixed", "space_type": "residential", "splat": False, "files": files}
        code, job = self.request("POST", "/api/jobs", json.dumps(data))
        self.assertEqual(code, 201)
        self.assertEqual(job["kind"], "mixed")
        for bad in (files[1:], files + [{"name": "둘째.mov", "size": 10}], files[:3]):  # 영상 없음·영상 2개·사진 2장
            code, _ = self.request("POST", "/api/jobs", json.dumps(data | {"files": bad}))
            self.assertEqual(code, 400)

    def test_declared_outdoor_is_not_queued_as_indoor(self):
        job = self.create("outdoor")
        for i in range(3):
            self.request("PUT", f"/api/jobs/{job['id']}/files/{i}", b"abc")
        _, started = self.request("POST", f"/api/jobs/{job['id']}/start", "{}")
        self.assertEqual(started["status"], "needs_review")
        self.assertTrue(self.store.queue.empty())

    def test_cross_site_request_cannot_create_jobs(self):
        code, _ = self.request("POST", "/api/jobs", "{}", {"Origin": "https://outside.example"})
        self.assertEqual(code, 403)
        self.assertEqual(len(self.store.jobs), 0)

    def test_source_work_and_traversal_are_not_served(self):
        for path in ["/scenes/x/work/database.db", "/scenes/%2e%2e/secret.json", "/../../secret.json"]:
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_wrong_length_and_invalid_filenames_are_rejected(self):
        job = self.create()
        self.assertEqual(self.request("PUT", f"/api/jobs/{job['id']}/files/0", b"ab")[0], 400)
        payload = {"kind": "video", "files": [{"name": "../house.mp4", "size": 5}]}
        self.assertEqual(self.request("POST", "/api/jobs", json.dumps(payload))[0], 400)

    def test_static_app_and_scene_listing(self):
        assets = self.root / "viewer/dist"
        assets.mkdir(parents=True)
        (assets / "index.html").write_text('{"local":true}', encoding="utf-8")
        self.assertEqual(self.request("GET", "/")[1], {"local": True})
        self.assertEqual(self.request("GET", "/scenes/")[1], [])


if __name__ == "__main__":
    unittest.main()
