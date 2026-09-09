# Local STT Batch Server

멀티채널 오디오를 받아 **로컬 GPU에서 faster-whisper로 STT**를 돌려주는 잡 큐 서버입니다.
Gemini STT 같은 종량제 API 대신, 평일 밤·주말처럼 컴퓨터가 노는 시간에만 배치로 처리하는 것이 목적입니다.

- **채널별 개별 STT** — 상담 녹취처럼 채널이 곧 화자인 오디오에서, 채널을 분리해 각각 인식한 뒤
  타임스탬프 기준으로 병합합니다. diarization 없이 화자 구분이 정확합니다.
- **오프피크 스케줄링** — 지정한 시간창 안에서, 그리고 GPU가 실제로 놀고 있을 때만 큐를 소진합니다.
- **비동기 잡 큐** — 업로드 즉시 `job_id`를 돌려주고, 클라이언트는 폴링하거나 웹훅 콜백을 받습니다.
- 외부 ffmpeg 바이너리 불필요(PyAV 내장), CUDA 툴킷·cuDNN 별도 설치 불필요(CTranslate2 내장).

## 검증된 환경

RTX 5060 Ti 16GB (Blackwell) / Windows 11 / Python 3.11 / CTranslate2 4.8.1 에서 동작 확인했습니다.

| 항목 | 실측값 |
|---|---|
| large-v3 float16 VRAM | 약 4.4 GB |
| 모델 최초 로드 | 약 54초 (이후 상주) |
| 2채널 오디오 처리 속도 | 실시간 대비 **3.8배** (채널당 약 7.5배) |

1시간짜리 2채널 통화 1건이 약 16분입니다. 22:00~08:00 창이면 하룻밤에 대략 **35시간 분량**을 처리합니다.

## 대시보드

서버를 띄우면 `http://<서버IP>:8760/` 이 현황 대시보드입니다. 게이트 상태와 이유, GPU 사용률·VRAM,
모델 상주 여부, 큐, 잡별 진행률을 보여주고 브라우저에서 바로 파일을 끌어다 넣을 수 있습니다.

- **오늘의 오프피크 창** 막대가 지금이 창 안인지 밖인지를 한눈에 보여줍니다(빨간 선이 현재 시각).
- **지금 실행 (60분)** 은 스케줄을 무시하고 즉시 처리합니다. 테스트가 끝나면 **스케줄 복귀**를 누르세요.
- 완료된 잡의 **결과** 버튼을 누르면 화자별 대화록이 뜹니다.
- **설정**에서 스케줄 기준과 GPU 유휴 기준을 각각 켜고 끄고, 임계값과 오디오 보관 기간을 바꿔
  저장할 수 있습니다. 저장하면 즉시 적용되고 `.env`에도 기록되어 재시작 후에도 유지됩니다.

자동 새로고침은 처리 중일 때 1.5초, 조용할 때 5초 간격이고 내용이 바뀐 경우에만 다시 그립니다.

## 호출하는 쪽 연동

호출하는 서버의 코딩 에이전트에게는 [docs/INTEGRATION.md](docs/INTEGRATION.md)를 통째로 넘기세요.
이 저장소를 몰라도 그것만 읽고 클라이언트를 붙일 수 있게 쓴 자립형 문서입니다.
(그 안에 클라이언트 코드 사본이 들어 있습니다. API를 바꾸면 `scripts/stt_client.py`와 함께 갱신하세요.)

## 설치

```bash
uv venv --python 3.11
```

```bash
uv sync
```

`.env.example`을 `.env`로 복사해 필요한 값만 고칩니다. 모든 설정은 `STT_` 접두사 환경변수로도 줄 수 있습니다.

## 실행

```bash
.venv/Scripts/python.exe -m app.serve
```

`app.serve`는 UTF-8 로그를 `data/server.log`에 20MB 단위로 로테이션해 남깁니다(3세대 보관).
대시보드 폴링이 로그를 뒤덮지 않도록 접근 로그는 끄고, 잡 제출·오버라이드·설정 변경만 기록합니다.

로그온할 때 자동으로 띄우려면 작업 스케줄러에 등록하세요. 관리자 권한은 필요 없습니다.

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_task.ps1
```

해제는 `scripts\uninstall_task.ps1` 입니다.

`http://<서버IP>:8760/docs` 에서 OpenAPI 문서를 볼 수 있습니다.
모델은 첫 잡이 들어올 때 내려받아 로드하고, 큐가 비면 `STT_MODEL_IDLE_UNLOAD_SEC` 후 VRAM을 반납합니다.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/v1/jobs` | 오디오 업로드 → `202` + `job_id` |
| `GET` | `/v1/jobs` | 잡 목록, 상태별 집계 |
| `GET` | `/v1/jobs/{id}` | 상태·진행률·단계 |
| `GET` | `/v1/jobs/{id}/result` | 완료된 전사 JSON |
| `DELETE` | `/v1/jobs/{id}` | 취소 (대기 중이면 즉시, 처리 중이면 다음 세그먼트에서 중단) |
| `GET` | `/v1/health` | GPU·모델·큐·스케줄 게이트 상태 |
| `POST` | `/v1/control` | 워커 일시정지, 또는 스케줄 무시(`override_minutes`) |
| `GET` | `/v1/settings` | 무중단으로 바꿀 수 있는 설정 조회 |
| `PUT` | `/v1/settings` | 설정 부분 변경 → 즉시 적용 + `.env` 저장 |
| `POST` | `/v1/maintenance/cleanup` | 오디오 정리를 지금 실행 |

`STT_API_KEY`를 설정하면 모든 엔드포인트가 `X-API-Key` 또는 `Authorization: Bearer` 를 요구합니다.

### 제출

```bash
curl -X POST http://127.0.0.1:8760/v1/jobs -F "file=@call.wav" -F 'options={"language":"ko","channel_names":["agent","customer"]}' -F "callback_url=http://client:9000/stt-done" -F "client_ref=ticket-42"
```

동봉된 레퍼런스 클라이언트가 업로드·폴링·저장까지 한 번에 해줍니다.

```bash
.venv/Scripts/python.exe scripts/submit.py call.wav --server http://127.0.0.1:8760 --channel-names agent,customer --out call.json
```

`options`에 넣을 수 있는 값은 `language`, `beam_size`, `vad_filter`, `word_timestamps`,
`condition_on_previous_text`, `initial_prompt`, `temperature`, `channel_names` 입니다.
생략하면 서버 기본값을 씁니다.

### 콜백

`callback_url`을 주면 완료·실패·취소 시 아래 JSON을 POST합니다 (실패하면 최대 3회 재시도).
본문에 전사 결과는 없으니, 받은 뒤 `result_url`을 GET 하세요.

```json
{
  "job_id": "cb1e8d34…",
  "status": "done",
  "client_ref": "ticket-42",
  "filename": "call.wav",
  "error": null,
  "result_url": "/v1/jobs/cb1e8d34…/result"
}
```

### 결과 형식

```json
{
  "job_id": "cb1e8d34…",
  "language": "ko",
  "audio": { "channels_total": 2, "duration_sec": 14.572, "codec": "pcm_s16le" },
  "channels": [
    { "index": 0, "name": "agent", "segments": [ { "start": 0.0, "end": 5.6, "text": "…" } ] },
    { "index": 1, "name": "customer", "segments": [ … ] }
  ],
  "segments": [
    { "channel": 0, "speaker": "agent", "start": 0.0, "end": 5.6, "text": "안녕하세요 고객님…" },
    { "channel": 1, "speaker": "customer", "start": 7.0, "end": 13.9, "text": "네 안녕하세요…" }
  ],
  "text": "[00:00:00] agent: 안녕하세요 고객님…\n[00:00:07] customer: 네 안녕하세요…"
}
```

`segments`는 모든 채널을 시작 시각순으로 병합한 것이고, `channels`는 채널별 원본입니다.
무음 채널은 `"skipped": "silent"`로 표시되고 STT를 돌리지 않습니다.

## 스케줄링

`STT_WINDOWS`는 `<요일> <HH:MM>-<HH:MM>` 항목을 `;`로 이어 씁니다.
끝 시각이 시작 시각보다 작거나 같으면 자정을 넘긴 것으로 봅니다.

```
STT_WINDOWS=Mon-Fri 22:00-08:00; Sat 00:00-24:00; Sun 00:00-24:00
```

요일은 `Mon-Fri`, `Sat,Sun`, `*`를 받고, 요일을 생략하면 매일입니다. `24:00`은 그날의 끝입니다.

### 두 개의 독립된 기준

잡을 꺼낼지는 **스케줄 기준**과 **GPU 유휴 기준** 두 개가 각각 결정합니다. 대시보드의 **설정**에서
켜고 끄거나 `PUT /v1/settings`로 바꿉니다.

| 스케줄 | GPU | 동작 |
|---|---|---|
| ON | ON | 시간창 안이면서 GPU도 한가할 때만 (기본값) |
| OFF | ON | **시간과 무관하게 GPU가 놀 때마다.** 낮에도 남는 자원을 씁니다 |
| ON | OFF | 시간창 안이면 GPU가 바빠도 |
| OFF | OFF | 들어오는 즉시 무조건 |

GPU 기준은 `STT_GPU_MAX_UTIL`(사용률 상한)과 `STT_GPU_MIN_FREE_MB`(여유 VRAM 하한)를 봅니다.
`STT_GPU_PROBE_SAMPLES`회 연속으로 한가해야 시작하므로 순간적인 부하에 끌려가지 않습니다.
모델이 이미 올라가 있으면 그 VRAM은 우리 것이므로 여유 VRAM 조건은 건너뜁니다.

게이트를 통과하지 못한 이유는 `/v1/health`의 `gate.reason`에 문장으로 나옵니다.

> 자기 자신이 전사 중일 때는 GPU 사용률이 90%를 넘으므로 **처리 중에는 게이트가 "GPU busy"로 보입니다.**
> 한 번에 한 잡만 처리하는 설계라 정상입니다.

창이 닫혀도 이미 시작한 잡은 끝까지 갑니다(`STT_FINISH_JOB_OUTSIDE_WINDOW=true`).
`false`로 두면 다음 채널 경계에서 잡을 큐로 되돌립니다.

급하게 한 건 돌려야 하면 스케줄을 잠시 무시할 수 있습니다.

```bash
curl -X POST http://127.0.0.1:8760/v1/control -H "content-type: application/json" -d "{\"override_minutes\": 60}"
```

`override_minutes: 0`이면 무시 상태가 풀리고 원래 스케줄로 돌아갑니다.

## 동작 방식

워커 스레드는 **한 번에 잡 하나만** 처리합니다. GPU를 나눠 쓰면 오히려 느려지고 VRAM만 위험해지기 때문입니다.

```
업로드 → data/uploads/  →  [게이트: 시간창 ∧ GPU 유휴]  →  워커
                                                            ├─ PyAV 디코드 → 채널별 16kHz float32
                                                            ├─ 채널마다 faster-whisper
                                                            ├─ 타임스탬프 병합
                                                            ├─ data/results/{job_id}.json → 콜백
                                                            └─ 업로드 원본 삭제
```

잡 상태는 `data/jobs.db`(SQLite, WAL)에 남습니다. 서버가 죽었다 살아나면 `running`이던 잡은
자동으로 `queued`로 되돌아갑니다. 실패한 잡은 `STT_MAX_ATTEMPTS`까지 재시도합니다.

## 오디오 보관 정책

**전사가 끝난 오디오는 바로 지웁니다.** 원본 녹취는 용량의 대부분이자 민감한 쪽이고, 남길 가치가 있는 건
전사 결과입니다. `data/results/*.json`은 건드리지 않습니다.

전사되지 않은 잡 — 대기 중이거나, 실패했거나, 취소된 — 의 오디오는 **3일간 보관한 뒤** 지웁니다.
기간이 지난 잡이 아직 대기 중이었다면 실패로 정리합니다. 오디오가 없어진 잡을 워커가 집어가서
매번 파일 없음으로 넘어지는 것보다 이유가 남는 편이 낫습니다.

| 잡 상태 | 오디오 |
|---|---|
| `done` | 결과 JSON을 쓰는 즉시 삭제 |
| `running` / `canceling` | 절대 건드리지 않음 (워커가 파일을 잡고 있음) |
| `queued` / `failed` / `canceled` | `STT_AUDIO_RETENTION_DAYS` 경과 후 삭제 |
| 잡 없는 `data/uploads/` 잔여 파일 | 같은 기간 경과 후 삭제 |

```
STT_DELETE_AUDIO_ON_SUCCESS=true   # 완료 즉시 원본 삭제
STT_AUDIO_RETENTION_DAYS=3         # 미완료 잡 오디오 보관 기간 (0 = 삭제 안 함)
STT_CLEANUP_INTERVAL_SEC=3600      # 정리 주기 (0 = 주기 정리 끔)
```

워커가 이 주기로 한 번씩 쓸어내고, 서버가 뜬 직후에도 한 번 돕니다. 대시보드 **설정 → 오디오 보관**에서
바꾸거나 **지금 정리**로 즉시 돌릴 수 있고, `POST /v1/maintenance/cleanup`도 같은 일을 합니다.
잠겨서 지우지 못한 파일은 다음 정리 때 다시 시도합니다.

지워진 잡은 `GET /v1/jobs/{id}`의 `audio_deleted_at`에 시각이 찍히고 대시보드에 `원본 삭제됨`으로 표시됩니다.
전사 결과는 그대로 남으니 `result_url`은 계속 유효합니다.

## 테스트

```bash
.venv/Scripts/python.exe -m pytest -q
```

스케줄 창 경계, 멀티채널 디코딩, 잡 스토어 동시성, API 계약을 덮습니다.
API 테스트는 "지금"을 제외한 시간창을 주입해 워커가 잡을 집어가지 않도록 만든 뒤 돌아갑니다.

## 알아둘 점

- 채널 라벨은 **위치 기준**입니다. `channel_names=["agent","customer"]`면 ch0=agent, ch1=customer입니다.
  녹취 장비가 채널 순서를 바꾸면 라벨도 같이 바뀝니다.
- 채널이 곧 화자가 아닌 녹음(회의실 마이크 어레이 등)에는 이 구조가 맞지 않습니다.
  그 경우엔 다운믹스 후 diarization이 필요하고, 지금은 들어 있지 않습니다.
- `STT_LANGUAGE`를 비우면 채널마다 언어를 자동 판별합니다. 한국어만 다룬다면 `ko`로 고정하는 쪽이
  짧은 발화에서 오판이 없습니다.
- 업로드 상한은 `STT_MAX_UPLOAD_MB`(기본 4GB)입니다.
