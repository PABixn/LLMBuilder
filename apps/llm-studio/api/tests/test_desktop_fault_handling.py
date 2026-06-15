from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from app import serve, storage_safety
from app.tokenizer_storage import StudioStore
from app.training_runs.store import TrainingStudioStore

ROOT = Path(__file__).resolve().parents[4]
API_ROOT = ROOT / "apps" / "llm-studio" / "api"


def test_listener_reports_actionable_port_collision() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        port = int(occupied.getsockname()[1])

        with pytest.raises(RuntimeError, match=r"port .*already be in use"):
            serve._bind_listener("127.0.0.1", port)


def test_writable_directory_probe_reports_permission_failure(monkeypatch, tmp_path: Path) -> None:
    def fail_probe(*_args, **_kwargs):
        raise PermissionError("read-only test root")

    monkeypatch.setattr(storage_safety.tempfile, "NamedTemporaryFile", fail_probe)

    with pytest.raises(storage_safety.ManagedStorageError, match="not writable"):
        storage_safety.ensure_writable_directory(
            tmp_path / "managed",
            operation="desktop fault test",
        )


@pytest.mark.parametrize(
    ("store_type", "database_name"),
    [
        (StudioStore, "tokenizer"),
        (TrainingStudioStore, "training"),
    ],
)
def test_corrupt_sqlite_database_reports_recovery_action(
    store_type,
    database_name: str,
    tmp_path: Path,
) -> None:
    database = tmp_path / f"{database_name}.db"
    database.write_bytes(b"not a sqlite database")
    store = store_type(url=f"sqlite:///{database}", sqlite_timeout_seconds=0.05)
    try:
        with pytest.raises(
            storage_safety.ManagedDatabaseError,
            match=rf"{database_name} database is unavailable.*locked.*read-only.*corrupt",
        ):
            store.initialize()
    finally:
        store.dispose()


def test_locked_sqlite_write_reports_recovery_action(tmp_path: Path) -> None:
    database = tmp_path / "tokenizer.db"
    primary = StudioStore(url=f"sqlite:///{database}")
    primary.initialize()
    primary.dispose()

    with sqlite3.connect(database) as lock:
        lock.execute("BEGIN IMMEDIATE")
        contender = StudioStore(
            url=f"sqlite:///{database}",
            sqlite_timeout_seconds=0.05,
        )
        try:
            with pytest.raises(storage_safety.ManagedDatabaseError, match="locked"):
                contender.record_uploaded_file("train", "input.txt", "/managed/input.txt", 1)
        finally:
            contender.dispose()


@pytest.mark.skipif(os.name != "posix", reason="Unix shell-parent watchdog")
def test_desktop_parent_identity_is_required(monkeypatch) -> None:
    monkeypatch.delenv(serve.PARENT_PID_ENV, raising=False)
    with pytest.raises(RuntimeError, match="requires a valid"):
        serve._desktop_parent_pid(True)

    monkeypatch.setenv(serve.PARENT_PID_ENV, str(os.getppid()))
    assert serve._desktop_parent_pid(True) == os.getppid()

    monkeypatch.setenv(serve.PARENT_PID_ENV, str(os.getpid()))
    with pytest.raises(RuntimeError, match="does not match"):
        serve._desktop_parent_pid(True)


@pytest.mark.skipif(os.name != "posix", reason="Unix shell-parent watchdog")
def test_desktop_backend_gracefully_exits_after_abrupt_parent_death(tmp_path: Path) -> None:
    handshake = tmp_path / "handshake.json"
    child_pid_path = tmp_path / "backend.pid"
    log_dir = tmp_path / "logs"
    token = "parent-watchdog-test-token"
    helper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _PARENT_HELPER,
            str(child_pid_path),
            str(handshake),
        ],
        cwd=API_ROOT,
        env={
            **os.environ,
            "LLM_STUDIO_DESKTOP": "1",
            "LLM_STUDIO_HOST": "127.0.0.1",
            "LLM_STUDIO_PORT": "0",
            "LLM_STUDIO_RUNTIME_TOKEN": token,
            "LLM_STUDIO_RUNTIME_VERSION": "parent-watchdog-test",
            "LLM_STUDIO_SOURCE_ROOT": str(ROOT),
            "LLM_STUDIO_DATA_DIR": str(tmp_path / "data"),
            "LLM_STUDIO_CACHE_DIR": str(tmp_path / "cache"),
            "LLM_STUDIO_LOG_DIR": str(log_dir),
            "LLM_STUDIO_STARTUP_HANDSHAKE_PATH": str(handshake),
            "LLM_STUDIO_SERVE_WEB": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    backend_pid: int | None = None
    try:
        _wait_for_path(child_pid_path, timeout=10)
        backend_pid = int(child_pid_path.read_text(encoding="utf-8"))
        startup = _wait_for_json(handshake, timeout=15)
        _wait_for_ready(str(startup["base_url"]), token, timeout=15)

        os.kill(helper.pid, signal.SIGKILL)
        helper.wait(timeout=5)

        _wait_for_process_exit(backend_pid, timeout=15)
        log_text = (log_dir / "backend.jsonl").read_text(encoding="utf-8")
        assert '"event_id":"backend.parent_exit.detected"' in log_text
        assert '"event_id":"backend.shutdown.complete"' in log_text
    finally:
        if helper.poll() is None:
            os.killpg(helper.pid, signal.SIGKILL)
            helper.wait(timeout=5)
        if backend_pid is not None and _process_exists(backend_pid):
            os.killpg(backend_pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason="Unix shell-parent watchdog")
def test_abrupt_desktop_parent_death_stops_active_local_training_tree_only(
    tmp_path: Path,
) -> None:
    handshake = tmp_path / "handshake.json"
    child_pid_path = tmp_path / "backend.pid"
    log_dir = tmp_path / "logs"
    token = "parent-watchdog-active-training-token"
    helper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _PARENT_HELPER,
            str(child_pid_path),
            str(handshake),
        ],
        cwd=API_ROOT,
        env={
            **os.environ,
            "LLM_STUDIO_DESKTOP": "1",
            "LLM_STUDIO_HOST": "127.0.0.1",
            "LLM_STUDIO_PORT": "0",
            "LLM_STUDIO_RUNTIME_TOKEN": token,
            "LLM_STUDIO_RUNTIME_VERSION": "parent-watchdog-training-test",
            "LLM_STUDIO_SOURCE_ROOT": str(ROOT),
            "LLM_STUDIO_DATA_DIR": str(tmp_path / "data"),
            "LLM_STUDIO_CACHE_DIR": str(tmp_path / "cache"),
            "LLM_STUDIO_LOG_DIR": str(log_dir),
            "LLM_STUDIO_STARTUP_HANDSHAKE_PATH": str(handshake),
            "LLM_STUDIO_SERVE_WEB": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    backend_pid: int | None = None
    training_pid: int | None = None
    try:
        _wait_for_path(child_pid_path, timeout=10)
        backend_pid = int(child_pid_path.read_text(encoding="utf-8"))
        startup = _wait_for_json(handshake, timeout=15)
        base_url = str(startup["base_url"])
        _wait_for_ready(base_url, token, timeout=15)
        training_pid = _launch_long_running_local_training(base_url, token, tmp_path)
        assert _process_group_exists(training_pid)

        os.kill(helper.pid, signal.SIGKILL)
        helper.wait(timeout=5)

        _wait_for_process_exit(backend_pid, timeout=25)
        _wait_for_process_group_exit(training_pid, timeout=25)
        assert unrelated.poll() is None
        log_text = (log_dir / "backend.jsonl").read_text(encoding="utf-8")
        assert '"event_id":"backend.parent_exit.detected"' in log_text
        assert '"event_id":"backend.shutdown.complete"' in log_text
    finally:
        if helper.poll() is None:
            os.killpg(helper.pid, signal.SIGKILL)
            helper.wait(timeout=5)
        if backend_pid is not None and _process_exists(backend_pid):
            os.killpg(backend_pid, signal.SIGKILL)
        if training_pid is not None and _process_group_exists(training_pid):
            os.killpg(training_pid, signal.SIGKILL)
        if unrelated.poll() is None:
            os.killpg(unrelated.pid, signal.SIGKILL)
        unrelated.wait(timeout=5)


_PARENT_HELPER = """
import os
from pathlib import Path
import subprocess
import sys
import time

pid_path = Path(sys.argv[1])
handshake = Path(sys.argv[2])
environment = os.environ.copy()
environment["LLM_STUDIO_PARENT_PID"] = str(os.getpid())
child = subprocess.Popen(
    [sys.executable, "-m", "app.serve"],
    env=environment,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
pid_path.write_text(str(child.pid), encoding="utf-8")
deadline = time.monotonic() + 20
while time.monotonic() < deadline and not handshake.is_file():
    if child.poll() is not None:
        raise SystemExit(child.returncode)
    time.sleep(0.05)
if not handshake.is_file():
    raise SystemExit("backend handshake timed out")
time.sleep(60)
"""


def _wait_for_path(path: Path, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for {path}")


def _wait_for_json(path: Path, *, timeout: float) -> dict[str, object]:
    _wait_for_path(path, timeout=timeout)
    return json.loads(path.read_text(encoding="utf-8"))


def _wait_for_ready(base_url: str, token: str, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "backend not ready"
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                f"{base_url}/api/v1/health",
                headers={"X-LLM-Studio-Token": token},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                if json.load(response).get("ready") is True:
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for backend readiness: {last_error}")


def _wait_for_process_exit(pid: int, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return
        time.sleep(0.05)
    raise TimeoutError(f"Process {pid} did not exit after its parent died")


def _wait_for_process_group_exit(pid: int, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_exists(pid):
            return
        time.sleep(0.05)
    raise TimeoutError(f"Process group {pid} did not exit after the desktop parent died")


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _request_json(
    url: str,
    token: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    timeout: float = 30,
) -> dict[str, object]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"X-LLM-Studio-Token": token}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc


def _wait_for_api_job(
    url: str,
    token: str,
    *,
    expected: set[str],
    timeout: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    latest: dict[str, object] = {}
    while time.monotonic() < deadline:
        latest = _request_json(url, token)
        if latest.get("status") in expected:
            return latest
        if latest.get("status") in {"failed", "cancelled"}:
            raise AssertionError(f"Job reached an unexpected terminal state: {latest}")
        time.sleep(0.1)
    raise TimeoutError(f"Job did not reach one of {sorted(expected)}: {latest}")


def _launch_long_running_local_training(base_url: str, token: str, tmp_path: Path) -> int:
    dataset = tmp_path / "parent-watchdog-training.txt"
    dataset.write_text(
        "long-running local training parent watchdog fixture\n" * 50_000,
        encoding="utf-8",
    )
    tokenizer_config = {
        "name": "parent-watchdog-tokenizer",
        "tokenizer_type": "wordpiece",
        "vocab_size": 64,
        "min_frequency": 1,
        "special_tokens": ["<unk>", "<|endoftext|>"],
        "pre_tokenizer": "whitespace",
        "decoder": "wordpiece",
        "unk_token": "<unk>",
    }
    tokenizer_dataloader = {
        "datasets": [
            {
                "name": "text",
                "data_files": {"train": str(dataset)},
                "split": "train",
                "text_columns": ["text"],
                "weight": 1.0,
            }
        ],
        "budget": {"limit": 20_000, "unit": "chars", "behavior": "truncate"},
        "mixing": {"seed": 42, "exhausted_policy": "stop"},
        "record_separator": "",
        "shuffle": False,
    }
    tokenizer_job = _request_json(
        f"{base_url}/api/v1/tokenizer/jobs",
        token,
        method="POST",
        payload={
            "tokenizer_config": tokenizer_config,
            "dataloader_config": tokenizer_dataloader,
            "evaluation_thresholds": [1],
        },
    )
    tokenizer_job = _wait_for_api_job(
        f"{base_url}/api/v1/tokenizer/jobs/{tokenizer_job['id']}",
        token,
        expected={"completed"},
        timeout=60,
    )
    vocab_size = int(dict(tokenizer_job["stats"])["vocab_size"])
    model_config = {
        "context_length": 16,
        "vocab_size": vocab_size,
        "n_embd": 16,
        "weight_tying": True,
        "blocks": [
            {
                "components": [
                    {"norm": {"type": "layernorm"}},
                    {"attention": {"n_head": 4, "n_kv_head": 4}},
                    {"norm": {"type": "layernorm"}},
                    {
                        "mlp": {
                            "multiplier": 2,
                            "sequence": [
                                {"linear": {"bias": True}},
                                {"activation": {"type": "relu"}},
                                {"linear": {"bias": True}},
                            ],
                        }
                    },
                ]
            }
        ],
    }
    project = _request_json(
        f"{base_url}/api/v1/projects",
        token,
        method="POST",
        payload={"name": "Parent watchdog project", "model_config": model_config},
    )
    max_steps = 100_000
    training_config = {
        "max_steps": max_steps,
        "total_batch_size": 8,
        "micro_batch_size": 1,
        "seq_len": 8,
        "sample_every": max_steps,
        "save_every": max_steps,
        "sampler": {
            "prompts": [
                {"prompt": "watchdog", "max_tokens": 2, "temperature": 0.7, "top_k": 4}
            ]
        },
        "optimizer": {
            "lr": 0.001,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "lr_scheduler": {
            "type": "sequential",
            "schedulers": [{"type": "constant", "steps": max_steps, "factor": 1.0}],
        },
    }
    training_dataloader = {
        "datasets": [
            {
                "name": "text",
                "data_files": {"train": str(dataset)},
                "split": "train",
                "streaming": True,
                "text_columns": ["text"],
                "weight": 1.0,
            }
        ],
        "add_eos": True,
        "eos_token": "<|endoftext|>",
        "drop_last": True,
        "mixing": {"seed": 42, "exhausted_policy": "stop"},
        "record_separator": "",
        "shuffle": False,
    }
    training_job = _request_json(
        f"{base_url}/api/v1/training/jobs",
        token,
        method="POST",
        payload={
            "name": "Parent watchdog active local training",
            "project_id": project["id"],
            "tokenizer_job_id": tokenizer_job["id"],
            "training_config": training_config,
            "dataloader_config": training_dataloader,
            "execution_target": {"kind": "local"},
        },
        timeout=60,
    )
    running = _wait_for_api_job(
        f"{base_url}/api/v1/training/jobs/{training_job['id']}",
        token,
        expected={"running"},
        timeout=30,
    )
    process_id = running.get("process_id")
    assert isinstance(process_id, int) and process_id > 1, running
    return process_id
