"""Read-only network checks. Never changes sandbox, account, proxy, or firewall."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://huggingface.co/datasets/prs-eth/PanoInfinigen/resolve/main/README.md"


def probe(url, limit=8192):
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            data = response.read(limit)
            return {"status": "ok", "http_status": response.status, "bytes": len(data),
                    "sha256_of_received_bytes": hashlib.sha256(data).hexdigest(),
                    "elapsed_seconds": round(time.perf_counter()-started, 3)}
    except (OSError, urllib.error.URLError) as error:
        cause = getattr(error, "reason", error)
        return {"status": "failed", "winerror": getattr(cause, "winerror", None),
                "error_type": type(cause).__name__, "message": str(cause),
                "elapsed_seconds": round(time.perf_counter()-started, 3)}


def main():
    identity = subprocess.run(["whoami"], capture_output=True, text=True, timeout=5)
    account = identity.stdout.strip().split("\\")[-1]
    disabled = os.environ.get("CODEX_SANDBOX_NETWORK_DISABLED") == "1"
    profile = os.environ.get("CODEX_PERMISSION_PROFILE")
    try:
        answers = socket.getaddrinfo("huggingface.co", 443, type=socket.SOCK_STREAM)
        dns = {"status": "ok", "address_count": len({item[4][0] for item in answers})}
    except OSError as error:
        dns = {"status": "failed", "message": str(error), "winerror": getattr(error, "winerror", None)}
    external = probe(ENDPOINT)
    local = probe("http://127.0.0.1:8765/api/health")
    if disabled and account.casefold() == "codexsandboxoffline" and external.get("winerror") == 10013:
        finding = "codex_offline_sandbox_confirmed"
        message = "Codex의 오프라인 실행 계정·네트워크 차단 설정·외부 연결 권한 거부가 함께 확인됐습니다."
    elif external["status"] == "ok":
        finding = "https_connection_succeeded"
        message = "이 프로세스에서 공식 데이터 출처에 HTTPS로 연결해 소량 내용을 받았습니다."
    else:
        finding = "needs_further_diagnosis"
        message = "외부 연결이 실패했지만 기존 오프라인 샌드박스 조건만으로 원인을 확정할 수 없습니다."
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "account": account,
              "sandbox_network_disabled": disabled, "permission_profile": profile,
              "endpoint": ENDPOINT, "dns": dns, "https_probe": external,
              "local_app_probe": local, "finding": finding, "message": message,
              "settings_changed": False, "security_controls_changed": False,
              "resolution_verified": external["status"] == "ok" and not disabled,
              "note": "네트워크 허용 여부는 이 스크립트 바깥의 실행 환경에서 설정합니다. 환경 변수·계정·방화벽·프록시를 변경하지 않습니다."}
    output = ROOT/"artifacts/network-diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    (output/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"실행 계정: {account}")
    print(f"샌드박스 외부 네트워크 차단: {'켜짐' if disabled else '표시 없음'}")
    print(f"호스트 이름 조회: {'성공' if dns['status']=='ok' else '실패'}")
    print(f"로컬 앱 연결: {'성공' if local['status']=='ok' else '실패 또는 서버 미실행'}")
    print(f"외부 HTTPS 연결: {'성공' if external['status']=='ok' else '실패'} / Windows 오류: {external.get('winerror', '없음')}")
    print(message)
    print(f"연결 복구 검증: {'완료' if report['resolution_verified'] else '미완료'}")
    print("진단 결과 저장: artifacts/network-diagnostics/result.json")
    return 0 if report["resolution_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
