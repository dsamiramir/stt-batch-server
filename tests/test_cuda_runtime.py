from types import SimpleNamespace

from app import cuda_runtime


def test_windows_runtime_registers_wheel_bins_and_keeps_handles(monkeypatch, tmp_path):
    for component in ("cublas", "cudnn"):
        (tmp_path / "nvidia" / component / "bin").mkdir(parents=True)
    monkeypatch.setattr(cuda_runtime.sys, "platform", "win32")
    monkeypatch.setattr(cuda_runtime, "_dll_handles", {})
    monkeypatch.setattr(
        cuda_runtime, "distribution",
        lambda name: SimpleNamespace(locate_file=lambda path: tmp_path / path),
    )
    registered = []
    def register(path):
        handle = object()
        registered.append((path, handle))
        return handle
    monkeypatch.setattr(cuda_runtime.os, "add_dll_directory", register)
    monkeypatch.setenv("PATH", "existing-path")
    cuda_runtime.configure_cuda_runtime()
    cuda_runtime.configure_cuda_runtime()
    assert len(registered) == 2
    assert list(cuda_runtime._dll_handles.values()) == [h for _, h in registered]
    assert cuda_runtime.os.environ["PATH"].split(cuda_runtime.os.pathsep) == [
        str((tmp_path / "nvidia" / "cudnn" / "bin").resolve()),
        str((tmp_path / "nvidia" / "cublas" / "bin").resolve()),
        "existing-path",
    ]


def test_missing_cuda_wheels_leave_cpu_installations_usable(monkeypatch):
    monkeypatch.setattr(cuda_runtime.sys, "platform", "win32")
    def missing(name):
        raise cuda_runtime.PackageNotFoundError(name)
    monkeypatch.setattr(cuda_runtime, "distribution", missing)
    monkeypatch.setenv("PATH", "existing-path")
    cuda_runtime.configure_cuda_runtime()
    assert cuda_runtime.os.environ["PATH"] == "existing-path"
