"""시작 시 코드·의존성으로 고정하는 빌드 식별자. 환경 비밀값은 읽지 않는다."""
from __future__ import annotations
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


def build_version(root: Path) -> str:
    digest = hashlib.sha256()
    # 커밋되지 않은 품질 개선 파일도 구분하되 체크아웃 절대 경로·줄바꿈은 제외한다.
    paths = [root / "pyproject.toml", *sorted((root / "app").rglob("*.py"))]
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    for name in ("faster-whisper", "ctranslate2", "av", "numpy"):
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "not-installed"
        digest.update(f"{name}={installed}\0".encode("utf-8"))
    release = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    return f"{release}+{digest.hexdigest()[:16]}"


SERVER_VERSION = build_version(Path(__file__).resolve().parent.parent)
