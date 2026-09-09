"""호출하는 쪽 서버에 그대로 복사해 쓰는 최소 클라이언트. httpx 하나만 필요합니다.

    from stt_client import SttClient

    stt = SttClient("http://172.17.15.85:8760")

    # (A) 폴링: 결과가 나올 때까지 기다림
    result = stt.transcribe("call.wav", channel_names=["agent", "customer"])
    print(result["text"])

    # (B) 콜백: 던져두고 잊음. 완료되면 callback_url 로 POST 가 온다.
    job_id = stt.submit("call.wav", callback_url="http://내서버:9000/stt-done",
                        client_ref="ticket-42")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

TERMINAL = {"done", "failed", "canceled"}


class SttJobError(RuntimeError):
    """잡이 failed/canceled 로 끝났을 때."""


class SttClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 600.0):
        headers = {"x-api-key": api_key} if api_key else {}
        # 업로드는 파일 크기에 비례해 오래 걸리므로 넉넉히.
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SttClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- 1) 제출 ----------------------------------------------------------
    def submit(
        self,
        audio_path: str | Path,
        *,
        language: str | None = "ko",
        channel_names: list[str] | None = None,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        callback_url: str = "",
        client_ref: str = "",
        priority: int = 0,
    ) -> str:
        """오디오를 큐에 넣고 job_id 를 돌려준다. 바로 반환된다(전사를 기다리지 않음)."""
        path = Path(audio_path)
        options: dict[str, Any] = {}
        if language:
            options["language"] = language
        if channel_names:
            options["channel_names"] = channel_names
        if word_timestamps:
            options["word_timestamps"] = True
        if initial_prompt:
            options["initial_prompt"] = initial_prompt

        with path.open("rb") as fh:
            response = self._http.post(
                "/v1/jobs",
                files={"file": (path.name, fh, "application/octet-stream")},
                data={
                    "options": json.dumps(options, ensure_ascii=False),
                    "callback_url": callback_url,
                    "client_ref": client_ref,
                    "priority": str(priority),
                },
            )
        response.raise_for_status()
        return response.json()["id"]

    # --- 2) 상태 ----------------------------------------------------------
    def status(self, job_id: str) -> dict[str, Any]:
        response = self._http.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def wait(self, job_id: str, poll_sec: float = 30.0, timeout_sec: float | None = None) -> dict[str, Any]:
        """터미널 상태가 될 때까지 폴링. 밤 배치라면 poll_sec 을 크게 잡아도 된다."""
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        while True:
            job = self.status(job_id)
            if job["status"] in TERMINAL:
                return job
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"job {job_id} still {job['status']}")
            time.sleep(poll_sec)

    # --- 3) 결과 ----------------------------------------------------------
    def result(self, job_id: str) -> dict[str, Any]:
        response = self._http.get(f"/v1/jobs/{job_id}/result")
        response.raise_for_status()
        return response.json()

    def transcribe(self, audio_path: str | Path, *, poll_sec: float = 30.0, **kwargs) -> dict[str, Any]:
        """제출 -> 대기 -> 결과까지 한 번에. 스케줄 창이 닫혀 있으면 열릴 때까지 기다린다."""
        job_id = self.submit(audio_path, **kwargs)
        job = self.wait(job_id, poll_sec=poll_sec)
        if job["status"] != "done":
            raise SttJobError(f"job {job_id} {job['status']}: {job.get('error')}")
        return self.result(job_id)

    # --- 부가 ------------------------------------------------------------
    def cancel(self, job_id: str) -> dict[str, Any]:
        response = self._http.delete(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        response = self._http.get("/v1/health")
        response.raise_for_status()
        return response.json()

    def run_now(self, minutes: int = 60) -> dict[str, Any]:
        """스케줄 창을 무시하고 지금 당장 돌린다. 0 이면 원래 스케줄로 복귀."""
        response = self._http.post("/v1/control", json={"override_minutes": minutes})
        response.raise_for_status()
        return response.json()
