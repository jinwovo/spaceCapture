# 다운로드 오류 10013의 확인된 원인

**복구 검증 완료 (2026-11-09 20:47 KST)**: CLI 실행 환경 변경 후 현재 계정은 `codexsandboxonline`, 외부 차단 표시는 없다. 공식 데이터 출처에 HTTPS **200 / 6,873바이트 / 0.260초**, `resolution_verified=true`를 확인했다. 이어 공식 ProcTHOR 검증 구조 **1,000개 / 5,094,681바이트 / 3.576초** 다운로드와 Git LFS SHA-256 검증도 성공했다. `samples/external/procthor-10k/download.json` 참고. 이 코드에서 방화벽·전역 설정을 변경하지 않았다.

로컬 8765의 10061은 서버 미실행 상태의 연결 거부다. 외부 HTTPS 복구와 별개다. 아래는 복구 전 진단과 재실행 방법의 기록이다.

확인: 2026-11-09 20:05 KST. 진단 결과: `artifacts/network-diagnostics/result.json`.

사용자가 실행 화면을 **Codex CLI**로 확인했다. 설치된 CLI는 **0.164.0**이며 `codex --help`, `codex resume --help`에서 아래 옵션을 확인했다. 현재 사용자 설정은 Windows 샌드박스 `elevated`이며 `sandbox_mode`, `default_permissions`, `sandbox_workspace_write.network_access`는 명시하지 않았다.

현재 명령은 Windows의 `codexsandboxoffline` 계정에서 실행되고 있다. `CODEX_SANDBOX_NETWORK_DISABLED=1`, 적용 프로필 `:workspace`도 확인했다. 같은 Python 프로세스에서 이름 조회와 로컬 앱 연결은 성공하지만 외부 HTTPS 연결은 소켓 접근 권한 오류 10013으로 거절된다. **현재 Codex 실행 환경의 오프라인 정책이 원인으로 확인됐다.**

| 확인 항목 | 결과 |
|---|---|
| 실행 계정 | codexsandboxoffline |
| 샌드박스 네트워크 차단 표시 | 켜짐 |
| 데이터 사이트 이름 조회 | 성공 |
| 로컬 앱 HTTP 연결 | 200, 61바이트, 0.015초 |
| 공식 데이터 출처 HTTPS 연결 | PermissionError / WinError 10013, 0.042초 |
| WinHTTP 프록시 | 직접 연결 |
| 방화벽 프로필·WMI 프로세스 조회 | 현재 계정에서 접근 거부 |
| 설정 변경·연결 복구 | 수행하지 않음·아직 미완료 |

OpenAI Docs는 Windows 샌드박스가 적용된 파일/네트워크 권한을 자식 프로세스에도 적용하며, 일부 대화는 의도적으로 외부 접속을 차단한다고 설명한다. 현재 결과는 이 정책과 일치한다. GPU·메모리·3D 파이프라인 수정으로 해결할 문제가 아니며, 특정 Windows 방화벽 규칙 자체를 읽어 검증한 것은 아니다. [Windows 샌드박스 공식 문서](https://developers.openai.com/codex/windows/)

## 필요한 변경

이 작업을 시작한 **호스트 앱 또는 Codex 실행 환경의 소유자**가 네트워크 접근을 허용한 세션을 제공해야 한다. 현재 대화 안에서 실행되는 프로그램은 자신에게 적용된 네트워크 정책을 변경할 권한이 없다. 사용자 전역 설정 파일은 프로젝트의 쓰기 허용 범위 밖이며, 이 세션의 권한 확장 요청도 허용되지 않는다.

### 확인된 Codex CLI에서 재실행

현재 CLI를 종료한 뒤 사용자의 PowerShell에서 다음 명령으로 이 프로젝트의 최근 대화를 이어받는다. 이 명령은 문서로만 준비했으며 현재 제한된 세션에서 새 실행 환경을 띄우는 방식으로 실행하지 않았다.

```powershell
codex resume --last -C C:\workspace\spaceCapture -s workspace-write -c sandbox_workspace_write.network_access=true
```

`-s workspace-write`는 작업 폴더 쓰기 범위를 유지하고, `-c sandbox_workspace_write.network_access=true`는 이번 실행의 외부 네트워크 접근을 허용한다. `--last`는 해당 작업 폴더의 최근 대화를 이어받는다. 사용자 전역 설정 파일을 변경하지 않는다. [CLI 옵션](https://learn.chatgpt.com/docs/cli/reference), [네트워크 설정](https://learn.chatgpt.com/docs/agent-approvals-security)

관리자가 허용 가능한 프로필을 제한한 경우에는 이 명령도 상위 정책을 덮어쓰지 않는다. 새 실행 환경에서 아래 진단이 성공하는지 확인하기 전까지 연결 복구 완료라고 표시하지 않는다.

관리형 앱/회사 정책이 적용된다면 그 권한 설정에서 필요한 데이터 출처의 HTTPS 접근만 허용하고 새 실행 환경에 적용해야 한다. 사용자 설정 파일을 편집해도 상위 관리 정책은 덮어쓸 수 없다. 설정을 제공하는 앱의 종류와 실제 UI 경로는 이 진단으로 확정하지 않았다.

**ChatGPT Work 화면을 사용하는 경우**, 공식 문서가 안내하는 경로는 **설정 → 데이터 제어 → Work 네트워크 접근 → 공개 인터넷 접근 허용**(`Settings → Data controls → Work network access → Allow public internet access`)이다. 해당 설정이 제공되는 계정/환경에만 표시되며 관리자 제한을 덮어쓰지 않는다. 변경은 현재 코드/명령 실행이 끝나고 실행 환경이 새로 적용된 뒤 반영된다. [Work 네트워크 공식 안내](https://learn.chatgpt.com/docs/enterprise/chatgpt-work-overview#network-access-and-external-destinations)

**Codex CLI를 직접 사용하는 경우**, 관리 정책이 허용하는 환경에서 실행 환경 소유자가 사용하는 설정은 다음과 같다. 이는 검토용 예시이며 프로젝트나 사용자 설정에 자동으로 적용하지 않았다. [설정 항목 공식 문서](https://developers.openai.com/codex/config-reference/)

```toml
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = true
```

이 설정은 작업 폴더 범위를 유지하면서 외부 연결 권한을 지정하는 예시다. 현재 호스트가 주입한 `:workspace` 프로필에 적용됐다고 가정해서는 안 된다. 권한을 설정한 뒤 새 실행 환경에서 유효 설정을 확인해야 한다. 환경 변수만 바꾸거나 Windows 방화벽을 끄는 방식으로 처리하지 않는다.

이 예시는 기존 `sandbox_mode` 형식을 쓰는 CLI용이다. `default_permissions`/`permissions` 프로필을 쓰는 설정과 혼합하지 않는다. 관리형 앱은 앱의 권한 정책을 우선 확인한다. 승인 요청을 자동 검토하는 모드를 선택하는 것만으로 네트워크가 활성화되는 것도 아니다. [권한 프로필 공식 설명](https://learn.chatgpt.com/docs/permissions)

## 복구 검증

권한이 허용된 환경에서 다음 진단을 다시 실행하면, 공식 데이터 README에 실제 HTTPS 연결을 하고 최대 8KiB만 읽어 결과를 기록한다. 외부 AI API·모델·대용량 데이터는 사용하지 않는다.

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/diagnose-network.py
```

`resolution_verified=true`, `https_probe.http_status=200`이고 실제 받은 바이트 수가 기록되어야 복구 완료다. 복구 전 결과는 false였으며 현재 재검증 결과는 true다. 다른 환경에서 다시 실패하면 새 오류를 기준으로 프록시·TLS·목적지 허용 목록 등을 구분한다.
