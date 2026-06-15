#!/usr/bin/env python3
"""Smoke a built runtime through imports, authenticated startup, and core API calls."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--performance-report",
        type=Path,
        help="Write non-browser runtime characterization measurements as JSON.",
    )
    args = parser.parse_args()
    runtime = args.runtime.expanduser().resolve()
    performance_report = (
        None
        if args.performance_report is None
        else args.performance_report.expanduser().resolve()
    )
    if performance_report is not None and performance_report.is_relative_to(runtime):
        parser.error("--performance-report must be outside the immutable runtime directory")
    snapshot = immutable_tree_snapshot(runtime)
    report: dict[str, object] | None = None
    try:
        report = smoke_built_runtime(runtime, args.timeout)
    finally:
        assert_immutable_tree_unchanged(runtime, snapshot)
    if performance_report is not None:
        assert report is not None
        write_json_atomic(performance_report, report)


def smoke_built_runtime(runtime: Path, timeout: float) -> dict[str, object]:
    smoke_started_at = time.monotonic()
    manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
    python = runtime / manifest["python_executable"]
    source = runtime / manifest["source_root"]
    api_root = source / "apps" / "llm-studio" / "api"
    measurements: dict[str, object] = {}

    subprocess.run(
        [
            str(python),
            "-c",
            "import fastapi,sqlalchemy,datasets,tokenizers,torch,model,tokenizer,training; "
            "print('Runtime imports passed')",
        ],
        cwd=api_root,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(source),
            "LLM_STUDIO_SOURCE_ROOT": str(source),
        },
        check=True,
    )

    token = secrets.token_hex(32)
    with tempfile.TemporaryDirectory(prefix="LLM Studio smoke unicode-\u0142-") as temporary:
        temp = Path(temporary)
        handshake = temp / "handshake.json"
        log = temp / "backend.log"
        web_dist = temp / "web-dist"
        web_dist.mkdir()
        (web_dist / "index.html").write_text(
            "<html><body><h1>LLM Studio packaged static smoke</h1></body></html>",
            encoding="utf-8",
        )
        (web_dist / "asset.txt").write_text("packaged-asset-ok", encoding="utf-8")
        environment = {
            **os.environ,
            "LLM_STUDIO_DESKTOP": "1",
            "LLM_STUDIO_HOST": "127.0.0.1",
            "LLM_STUDIO_PORT": "0",
            "LLM_STUDIO_RUNTIME_TOKEN": token,
            "LLM_STUDIO_RUNTIME_VERSION": manifest["runtime_version"],
            "LLM_STUDIO_SOURCE_ROOT": str(source),
            "LLM_STUDIO_DATA_DIR": str(temp / "data"),
            "LLM_STUDIO_CACHE_DIR": str(temp / "cache"),
            "LLM_STUDIO_LOG_DIR": str(temp / "logs"),
            "LLM_STUDIO_STARTUP_HANDSHAKE_PATH": str(handshake),
            "LLM_STUDIO_PARENT_PID": str(os.getpid()),
            "LLM_STUDIO_SERVE_WEB": "1",
            "LLM_STUDIO_INFERENCE_DEVICE": "cpu",
            "LLM_STUDIO_TRAINING_DEVICE": "cpu",
            "LLM_STUDIO_WEB_DIST_DIR": str(web_dist),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(source),
        }
        cold_start_started_at = time.monotonic()
        process = spawn_runtime(python, api_root, environment, log)
        try:
            startup = wait_for_startup(process, handshake, token, timeout)
            cold_start_measurement: dict[str, object] = {
                "first_api_ready_ms": elapsed_ms(cold_start_started_at),
            }
            measurements["cold_start"] = cold_start_measurement
            base = startup["base_url"]
            assert request_json(f"{base}/health", token)["ok"] is True
            health = request_json(f"{base}/api/v1/health", token)
            assert health["ready"] is True
            assert health["api_contract_version"] == manifest["api_contract_version"]
            assert health["runtime_version"] == manifest["runtime_version"]
            assert health["startup_timing"]["ready_in_ms"] is not None
            assert health["startup_timing"]["ready_in_ms"] >= 0
            assert health["startup_timing"]["stage_ms"]["process_imports"] >= 0
            assert health["startup_timing"]["stage_ms"]["migrating_data"] >= 0
            assert health["compute"]["cpu"] is True
            print(
                "Runtime startup timing: "
                + json.dumps(health["startup_timing"], sort_keys=True)
            )
            cold_start_measurement["backend_startup_profile"] = health["startup_timing"]
            measurements["compute_capabilities"] = health["compute"]
            measurements["idle_backend_memory"] = sample_process_rss(process.pid)
            assert request_json(f"{base}/api/v1/config/templates", token)["model_config_template"]
            assert request_json(f"{base}/api/v1/config/schemas", token)["model_config_schema"]
            assert request_json(f"{base}/api/v1/tokenizer/health", token)["ok"] is True
            assert request_json(f"{base}/api/v1/tokenizer/config/templates", token)[
                "tokenizer_config_template"
            ]
            assert request_json(f"{base}/api/v1/tokenizer/config/schemas", token)[
                "tokenizer_schema"
            ]
            assert request_json(f"{base}/api/v1/training/health", token)["ok"] is True
            assert request_json(f"{base}/api/v1/training/config/templates", token)[
                "training_config_template"
            ]
            assert request_json(f"{base}/api/v1/training/config/schemas", token)[
                "training_config_schema"
            ]
            assert request_json(f"{base}/api/v1/training/providers/runpod/defaults", token)
            assert request_json(f"{base}/api/v1/training/providers/runpod/status", token)[
                "configured"
            ] is False
            runpod_catalog = request_json(
                f"{base}/api/v1/training/providers/runpod/catalog",
                token,
            )
            assert runpod_catalog["gpu_options"]
            assert_http_status(
                f"{base}/api/v1/training/providers/runpod/pods",
                409,
                token=token,
            )
            assert_http_status(
                f"{base}/api/v1/training/providers/runpod/network-volumes",
                409,
                token=token,
            )
            assert b"LLM Studio packaged static smoke" in request_bytes(f"{base}/", token)
            assert request_bytes(f"{base}/asset.txt", token) == b"packaged-asset-ok"
            assert b"LLM Studio packaged static smoke" in request_bytes(
                f"{base}/deep/link/without/extension",
                token,
            )
            assert_http_status(f"{base}/missing.js", 404, token=token)
            assert_http_status(f"{base}/api/v1/unknown-route", 404, token=token)
            assert_http_status(f"{base}/%2e%2e/%2e%2e/etc/passwd", 404, token=token)
            workflow_timings: dict[str, float] = {}
            persistent = exercise_core_workflows(base, token, temp, workflow_timings)
            measurements["tiny_jobs"] = workflow_timings
            assert_unauthorized(f"{base}/api/v1/health")

            terminate_tree(process)
            first_port = startup["port"]
            handshake.unlink(missing_ok=True)
            previous_token = token
            token = secrets.token_hex(32)
            environment["LLM_STUDIO_RUNTIME_TOKEN"] = token
            warm_start_started_at = time.monotonic()
            process = spawn_runtime(python, api_root, environment, log)
            startup = wait_for_startup(process, handshake, token, timeout)
            warm_health = request_json(f"{startup['base_url']}/api/v1/health", token)
            measurements["warm_restart"] = {
                "first_api_ready_ms": elapsed_ms(warm_start_started_at),
                "backend_startup_profile": warm_health["startup_timing"],
            }
            base = startup["base_url"]
            assert_token_rejected(f"{base}/api/v1/health", previous_token)
            verify_restart_persistence(base, token, persistent)
            cleanup_core_workflows(base, token, persistent)
            assert isinstance(first_port, int) and isinstance(startup["port"], int)
            print(f"Runtime smoke passed: {runtime}")
            measurements["full_smoke_ms"] = elapsed_ms(smoke_started_at)
            report = build_performance_report(manifest, measurements)
            print("Runtime characterization: " + json.dumps(report["measurements"], sort_keys=True))
            return report
        finally:
            failed = sys.exc_info()[0] is not None
            terminate_tree(process)
            if failed or process.returncode not in (None, 0, -signal.SIGTERM):
                print(log.read_text(encoding="utf-8", errors="replace"))


def immutable_tree_snapshot(root: Path) -> dict[str, tuple[str, int, int, int, str]]:
    """Capture runtime contents and metadata without following packaged symlinks."""
    if not root.is_dir():
        raise FileNotFoundError(f"Runtime directory is missing: {root}")

    snapshot = {".": immutable_path_snapshot(root)}

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                path = Path(entry.path)
                snapshot[path.relative_to(root).as_posix()] = immutable_path_snapshot(path)
                if entry.is_dir(follow_symlinks=False):
                    visit(path)

    visit(root)
    return snapshot


def immutable_path_snapshot(path: Path) -> tuple[str, int, int, int, str]:
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        return ("symlink", mode, metadata.st_size, metadata.st_mtime_ns, os.readlink(path))
    if stat.S_ISREG(metadata.st_mode):
        return ("file", mode, metadata.st_size, metadata.st_mtime_ns, sha256(path))
    if stat.S_ISDIR(metadata.st_mode):
        return ("directory", mode, 0, metadata.st_mtime_ns, "")
    return ("other", mode, metadata.st_size, metadata.st_mtime_ns, "")


def assert_immutable_tree_unchanged(
    runtime: Path,
    expected: dict[str, tuple[str, int, int, int, str]],
) -> None:
    actual = immutable_tree_snapshot(runtime)
    added = sorted(actual.keys() - expected.keys())
    removed = sorted(expected.keys() - actual.keys())
    changed = sorted(path for path in actual.keys() & expected.keys() if actual[path] != expected[path])
    if added or removed or changed:
        details = []
        for label, paths in (("added", added), ("removed", removed), ("changed", changed)):
            if paths:
                suffix = " ..." if len(paths) > 20 else ""
                details.append(f"{label}={paths[:20]}{suffix}")
        raise AssertionError(
            "Packaged runtime resources changed during the smoke workflow: " + "; ".join(details)
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def elapsed_ms(started_at: float) -> float:
    return round((time.monotonic() - started_at) * 1000, 3)


def build_performance_report(
    manifest: dict[str, object],
    measurements: dict[str, object],
) -> dict[str, object]:
    provenance = manifest.get("provenance")
    build_mode = provenance.get("build_mode") if isinstance(provenance, dict) else None
    size = manifest.get("size")
    runtime_size = {
        key: size[key]
        for key in ("payload_file_count", "payload_total_bytes")
        if isinstance(size, dict) and key in size
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "classification": "characterization-only",
        "target": {
            "platform": manifest.get("platform"),
            "architecture": manifest.get("architecture"),
        },
        "runtime": {
            "runtime_version": manifest.get("runtime_version"),
            "python_version": manifest.get("python_version"),
            "build_mode": build_mode,
            "size": runtime_size,
        },
        "measurements": measurements,
        "scope": {
            "idle_memory": "backend-process-only",
            "startup": "backend-spawn-to-first-authenticated-api-readiness",
            "tiny_jobs": "authenticated-create-request-to-terminal-status",
            "route_navigation": "excluded-by-product-owner-no-browser-checks",
            "shell_and_webview_memory": "not-measured",
            "installer_and_update_size": "not-available",
        },
        "thresholds": "not-approved",
    }


def sample_process_rss(
    pid: int,
    *,
    sample_count: int = 3,
    interval_seconds: float = 0.2,
) -> dict[str, object]:
    samples: list[int] = []
    failure_types: set[str] = set()
    for index in range(sample_count):
        try:
            samples.append(measure_process_rss_bytes(pid))
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            failure_types.add(type(error).__name__)
        if index + 1 < sample_count:
            time.sleep(interval_seconds)
    if not samples:
        return {
            "status": "unavailable",
            "requested_samples": sample_count,
            "failure_types": sorted(failure_types),
        }
    ordered = sorted(samples)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2 == 1
        else (ordered[middle - 1] + ordered[middle]) // 2
    )
    return {
        "status": "measured",
        "requested_samples": sample_count,
        "measured_samples": len(samples),
        "unavailable_samples": sample_count - len(samples),
        "samples_bytes": samples,
        "minimum_bytes": ordered[0],
        "median_bytes": median,
        "maximum_bytes": ordered[-1],
    }


def measure_process_rss_bytes(pid: int) -> int:
    if pid <= 0:
        raise ValueError("PID must be positive")
    if sys.platform.startswith("linux"):
        return parse_linux_rss_bytes(Path(f"/proc/{pid}/status").read_text(encoding="utf-8"))
    if os.name == "nt":
        return measure_windows_process_rss_bytes(pid)
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return parse_ps_rss_bytes(result.stdout)


def parse_linux_rss_bytes(status: str) -> int:
    for line in status.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        parts = line.split()
        if len(parts) != 3 or parts[2] != "kB":
            raise ValueError("Unexpected VmRSS format")
        return int(parts[1]) * 1024
    raise ValueError("VmRSS is unavailable")


def parse_ps_rss_bytes(output: str) -> int:
    value = output.strip()
    if not value:
        raise ValueError("RSS is unavailable")
    return int(value) * 1024


def measure_windows_process_rss_bytes(pid: int) -> int:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "OpenProcess failed")
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def wait_for_startup(
    process: subprocess.Popen[bytes],
    handshake: Path,
    token: str,
    timeout: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last_error = "waiting for handshake"
    startup: dict[str, object] | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Runtime exited during startup with code {process.returncode}")
        if startup is None and handshake.is_file():
            startup = json.loads(handshake.read_text(encoding="utf-8"))
        if startup is not None:
            try:
                health = request_json(
                    f"{startup['base_url']}/api/v1/health",
                    token,
                    timeout=2,
                )
                if health.get("ready") is True:
                    return startup
                last_error = str(health.get("startup_detail", "not ready"))
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                last_error = str(error)
        time.sleep(0.25)
    raise TimeoutError(f"Runtime startup timed out: {last_error}")


def spawn_runtime(
    python: Path,
    api_root: Path,
    environment: dict[str, str],
    log: Path,
) -> subprocess.Popen[bytes]:
    with log.open("ab") as output:
        return subprocess.Popen(
            [str(python), "-m", "app.serve"],
            cwd=api_root,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )


def exercise_core_workflows(
    base: str,
    token: str,
    temp: Path,
    timings: dict[str, float] | None = None,
) -> dict[str, str]:
    timings = timings if timings is not None else {}
    dataset = temp / "smoke-input.txt"
    dataset.write_text(
        "\n".join(
            [
                "hello world from llm studio",
                "small deterministic desktop runtime smoke",
                "model tokenizer training inference",
            ]
            * 64
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer_config = {
        "name": "desktop-smoke-wordpiece",
        "tokenizer_type": "wordpiece",
        "vocab_size": 64,
        "min_frequency": 1,
        "special_tokens": ["<unk>", "<|endoftext|>"],
        "pre_tokenizer": "whitespace",
        "decoder": "wordpiece",
        "unk_token": "<unk>",
    }
    uploaded_train = request_multipart_file(
        f"{base}/api/v1/tokenizer/files/train",
        token,
        filename="../../desktop smoke train.txt",
        content=dataset.read_bytes(),
    )
    assert Path(str(uploaded_train["file_path"])).resolve().is_relative_to(
        (temp / "data").resolve()
    )
    train_stats = request_json(
        f"{base}/api/v1/tokenizer/files/stats?file_path="
        f"{urllib.parse.quote(str(uploaded_train['file_path']))}",
        token,
    )
    assert train_stats["size_bytes"] == uploaded_train["size_bytes"]
    uploaded_validation = request_multipart_file(
        f"{base}/api/v1/tokenizer/files/validation",
        token,
        filename="desktop smoke validation.txt",
        content=b"validation text\n",
    )
    assert Path(str(uploaded_validation["file_path"])).resolve().is_relative_to(
        (temp / "data").resolve()
    )
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
    assert request_json(
        f"{base}/api/v1/tokenizer/validate/tokenizer",
        token,
        method="POST",
        payload={"config": tokenizer_config},
    )["valid"] is True
    assert request_json(
        f"{base}/api/v1/tokenizer/validate/dataloader",
        token,
        method="POST",
        payload={"config": tokenizer_dataloader},
    )["valid"] is True
    tokenizer_job_started_at = time.monotonic()
    tokenizer_job = request_json(
        f"{base}/api/v1/tokenizer/jobs",
        token,
        method="POST",
        payload={
            "tokenizer_config": tokenizer_config,
            "dataloader_config": tokenizer_dataloader,
            "evaluation_thresholds": [1, 5],
        },
    )
    tokenizer_job = wait_for_job(
        f"{base}/api/v1/tokenizer/jobs/{tokenizer_job['id']}",
        token,
        timeout=90,
    )
    assert tokenizer_job["status"] == "completed", tokenizer_job
    timings["tokenizer_create_to_terminal_ms"] = elapsed_ms(tokenizer_job_started_at)
    assert request_json(
        f"{base}/api/v1/tokenizer/jobs/{tokenizer_job['id']}/artifact/meta",
        token,
    )["exists"] is True
    assert request_bytes(
        f"{base}/api/v1/tokenizer/jobs/{tokenizer_job['id']}/artifact",
        token,
    ).startswith(b"{")
    assert any(
        job["id"] == tokenizer_job["id"]
        for job in request_json(f"{base}/api/v1/tokenizer/jobs", token)["jobs"]
    )
    assert request_json(
        f"{base}/api/v1/tokenizer/jobs/{tokenizer_job['id']}",
        token,
    )["status"] == "completed"
    assert request_json(
        f"{base}/api/v1/tokenizer/jobs/{tokenizer_job['id']}/preview",
        token,
        method="POST",
        payload={"text": "hello world"},
    )["num_tokens"] > 0

    vocab_size = int(tokenizer_job["stats"]["vocab_size"])
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
    assert request_json(
        f"{base}/api/v1/validate/model",
        token,
        method="POST",
        payload={"config": model_config},
    )["valid"] is True
    analysis = request_json(
        f"{base}/api/v1/analyze/model",
        token,
        method="POST",
        payload={"config": model_config},
    )
    assert analysis["instantiated"] is True
    project = request_json(
        f"{base}/api/v1/projects",
        token,
        method="POST",
        payload={"name": "Desktop runtime smoke", "model_config": model_config},
    )
    project_id = str(project["id"])
    assert any(
        item["id"] == project_id
        for item in request_json(f"{base}/api/v1/projects", token)["projects"]
    )
    assert request_json(f"{base}/api/v1/projects/{project_id}", token)["name"] == (
        "Desktop runtime smoke"
    )
    updated_project = request_json(
        f"{base}/api/v1/projects/{project_id}",
        token,
        method="PUT",
        payload={"name": "Desktop runtime smoke updated", "model_config": model_config},
    )
    assert updated_project["name"] == "Desktop runtime smoke updated"
    assert request_bytes(f"{base}/api/v1/projects/{project_id}/artifact", token).startswith(b"{")

    training_config = {
        "max_steps": 1,
        "total_batch_size": 8,
        "micro_batch_size": 1,
        "seq_len": 8,
        "sample_every": 1,
        "sampler": {
            "prompts": [
                {"prompt": "hello", "max_tokens": 2, "temperature": 0.7, "top_k": 4}
            ]
        },
        "save_every": 1,
        "optimizer": {
            "lr": 0.001,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "lr_scheduler": {
            "type": "sequential",
            "schedulers": [{"type": "constant", "steps": 1, "factor": 1.0}],
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
    preflight_request = {
        "project_id": project_id,
        "tokenizer_job_id": tokenizer_job["id"],
        "training_config": training_config,
        "dataloader_config": training_dataloader,
    }
    assert request_json(
        f"{base}/api/v1/training/validate/training-config",
        token,
        method="POST",
        payload={"config": training_config},
    )["valid"] is True
    assert request_json(
        f"{base}/api/v1/training/validate/dataloader",
        token,
        method="POST",
        payload={"config": training_dataloader},
    )["valid"] is True
    preflight = request_json(
        f"{base}/api/v1/training/validate/preflight",
        token,
        method="POST",
        payload=preflight_request,
    )
    assert preflight["valid"] is True, preflight
    missing_key_runpod_job = request_json(
        f"{base}/api/v1/training/jobs",
        token,
        method="POST",
        payload={
            "name": "Desktop runtime missing-key RunPod smoke",
            **preflight_request,
            "execution_target": {"kind": "runpod_pod"},
        },
    )
    missing_key_runpod_job_id = str(missing_key_runpod_job["id"])
    missing_key_runpod_job = wait_for_job(
        f"{base}/api/v1/training/jobs/{missing_key_runpod_job_id}",
        token,
        timeout=30,
    )
    assert missing_key_runpod_job["status"] == "failed", missing_key_runpod_job
    assert "RunPod API key is required" in str(missing_key_runpod_job["remote_error"])
    resynced_runpod_job = request_json(
        f"{base}/api/v1/training/jobs/{missing_key_runpod_job_id}/remote/resync",
        token,
        method="POST",
    )
    assert resynced_runpod_job["id"] == missing_key_runpod_job_id
    cleaned_runpod_job = request_json(
        f"{base}/api/v1/training/jobs/{missing_key_runpod_job_id}/remote/cleanup",
        token,
        method="POST",
    )
    assert cleaned_runpod_job["executor_status"] == "cleaned_up"
    reattached_runpod_job = request_json(
        f"{base}/api/v1/training/jobs/{missing_key_runpod_job_id}/remote/reattach",
        token,
        method="POST",
    )
    assert "unavailable" in str(reattached_runpod_job["remote_error"]).lower()
    assert request_status(
        f"{base}/api/v1/training/jobs/{missing_key_runpod_job_id}",
        token,
        method="DELETE",
    ) == 204
    training_job_started_at = time.monotonic()
    training_job = request_json(
        f"{base}/api/v1/training/jobs",
        token,
        method="POST",
        payload={
            "name": "Desktop runtime smoke",
            **preflight_request,
            "execution_target": {"kind": "local"},
        },
    )
    training_job = wait_for_job(
        f"{base}/api/v1/training/jobs/{training_job['id']}",
        token,
        timeout=120,
    )
    assert training_job["status"] == "completed", training_job
    timings["local_training_create_to_terminal_ms"] = elapsed_ms(training_job_started_at)
    training_job_id = str(training_job["id"])
    assert any(
        item["id"] == training_job_id
        for item in request_json(f"{base}/api/v1/training/jobs", token)["jobs"]
    )
    assert request_json(f"{base}/api/v1/training/jobs/{training_job_id}", token)[
        "status"
    ] == "completed"
    assert request_json(f"{base}/api/v1/training/jobs/{training_job_id}/metrics", token)[
        "metrics"
    ]
    assert request_json(f"{base}/api/v1/training/jobs/{training_job_id}/samples", token)[
        "samples"
    ]
    logs = request_json(f"{base}/api/v1/training/jobs/{training_job_id}/logs", token)
    assert "stdout_lines" in logs and "stderr_lines" in logs
    checkpoints = request_json(
        f"{base}/api/v1/training/jobs/{training_job_id}/checkpoints",
        token,
    )
    assert checkpoints["checkpoints"]
    generated = request_json(
        f"{base}/api/v1/training/jobs/{training_job_id}/generate",
        token,
        method="POST",
        timeout=120,
        payload={
            "prompt": "hello",
            "max_tokens": 2,
            "temperature": 0.7,
            "top_k": 4,
            "checkpoint_step": 1,
        },
    )
    assert generated["generated_token_ids"] is not None
    stream_events = request_ndjson(
        f"{base}/api/v1/training/jobs/{training_job_id}/generate/stream",
        token,
        payload={
            "prompt": "hello",
            "max_tokens": 2,
            "temperature": 0.7,
            "top_k": 4,
            "checkpoint_step": 1,
        },
        timeout=120,
    )
    assert stream_events[0]["type"] == "start"
    assert stream_events[-1]["type"] == "done"
    artifact = request_bytes(
        f"{base}/api/v1/training/jobs/{training_job_id}/artifact",
        token,
    )
    assert artifact.startswith(b"PK")

    stoppable_training_config = {
        **training_config,
        "max_steps": 1_000,
        "sample_every": 1_000,
        "save_every": 1_000,
        "lr_scheduler": {
            "type": "sequential",
            "schedulers": [{"type": "constant", "steps": 1_000, "factor": 1.0}],
        },
    }
    stoppable_dataset = temp / "stop-smoke-input.txt"
    stoppable_dataset.write_text(
        "long-running deterministic cancellation dataset line\n" * 100_000,
        encoding="utf-8",
    )
    stoppable_dataloader = {
        **training_dataloader,
        "datasets": [
            {
                **training_dataloader["datasets"][0],
                "data_files": {"train": str(stoppable_dataset)},
            }
        ],
    }
    stoppable_preflight = request_json(
        f"{base}/api/v1/training/validate/preflight",
        token,
        method="POST",
        payload={
            **preflight_request,
            "training_config": stoppable_training_config,
            "dataloader_config": stoppable_dataloader,
        },
    )
    assert stoppable_preflight["valid"] is True, stoppable_preflight
    stoppable_job = request_json(
        f"{base}/api/v1/training/jobs",
        token,
        method="POST",
        payload={
            "name": "Desktop runtime stop smoke",
            **preflight_request,
            "training_config": stoppable_training_config,
            "dataloader_config": stoppable_dataloader,
            "execution_target": {"kind": "local"},
        },
    )
    stoppable_job_id = str(stoppable_job["id"])
    wait_for_status(
        f"{base}/api/v1/training/jobs/{stoppable_job_id}",
        token,
        expected={"running"},
        timeout=30,
    )
    stopped = request_json(
        f"{base}/api/v1/training/jobs/{stoppable_job_id}/stop",
        token,
        method="POST",
    )
    assert stopped["status"] == "cancelled", stopped
    assert request_status(
        f"{base}/api/v1/training/jobs/{stoppable_job_id}",
        token,
        method="DELETE",
    ) == 204

    return {
        "project_id": project_id,
        "tokenizer_job_id": str(tokenizer_job["id"]),
        "training_job_id": training_job_id,
    }


def verify_restart_persistence(base: str, token: str, persistent: dict[str, str]) -> None:
    project_id = persistent["project_id"]
    tokenizer_job_id = persistent["tokenizer_job_id"]
    training_job_id = persistent["training_job_id"]

    assert any(
        item["id"] == project_id
        for item in request_json(f"{base}/api/v1/projects", token)["projects"]
    )
    assert request_json(f"{base}/api/v1/projects/{project_id}", token)["name"] == (
        "Desktop runtime smoke updated"
    )
    assert request_bytes(f"{base}/api/v1/projects/{project_id}/artifact", token).startswith(b"{")
    assert any(
        item["id"] == tokenizer_job_id
        for item in request_json(f"{base}/api/v1/tokenizer/jobs", token)["jobs"]
    )
    assert request_json(f"{base}/api/v1/tokenizer/jobs/{tokenizer_job_id}", token)[
        "status"
    ] == "completed"
    assert request_bytes(f"{base}/api/v1/tokenizer/jobs/{tokenizer_job_id}/artifact", token).startswith(
        b"{"
    )
    assert any(
        item["id"] == training_job_id
        for item in request_json(f"{base}/api/v1/training/jobs", token)["jobs"]
    )
    assert request_json(f"{base}/api/v1/training/jobs/{training_job_id}", token)[
        "status"
    ] == "completed"
    assert request_bytes(f"{base}/api/v1/training/jobs/{training_job_id}/artifact", token).startswith(
        b"PK"
    )


def cleanup_core_workflows(base: str, token: str, persistent: dict[str, str]) -> None:
    assert request_status(
        f"{base}/api/v1/training/jobs/{persistent['training_job_id']}",
        token,
        method="DELETE",
    ) == 204
    assert request_status(
        f"{base}/api/v1/tokenizer/jobs/{persistent['tokenizer_job_id']}",
        token,
        method="DELETE",
    ) == 204
    assert request_status(
        f"{base}/api/v1/projects/{persistent['project_id']}",
        token,
        method="DELETE",
    ) == 204


def wait_for_job(url: str, token: str, *, timeout: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    latest: dict[str, object] = {}
    while time.monotonic() < deadline:
        latest = request_json(url, token)
        if latest.get("status") in {"completed", "failed", "cancelled"}:
            return latest
        time.sleep(0.25)
    raise TimeoutError(f"Job did not reach a terminal state: {latest}")


def wait_for_status(
    url: str,
    token: str,
    *,
    expected: set[str],
    timeout: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    latest: dict[str, object] = {}
    while time.monotonic() < deadline:
        latest = request_json(url, token)
        if latest.get("status") in expected:
            return latest
        if latest.get("status") in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.1)
    raise TimeoutError(f"Job did not reach one of {sorted(expected)}: {latest}")


def request_json(
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
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {error.code}: {detail}") from error


def request_bytes(url: str, token: str) -> bytes:
    request = urllib.request.Request(url, headers={"X-LLM-Studio-Token": token})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def request_multipart_file(
    url: str,
    token: str,
    *,
    filename: str,
    content: bytes,
) -> dict[str, object]:
    boundary = f"----LLMStudioSmoke{secrets.token_hex(12)}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "X-LLM-Studio-Token": token,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 201
        return json.load(response)


def request_ndjson(
    url: str,
    token: str,
    *,
    payload: dict[str, object],
    timeout: float,
) -> list[dict[str, object]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "X-LLM-Studio-Token": token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        assert response.headers.get_content_type() == "application/x-ndjson"
        return [
            json.loads(line)
            for line in response.read().decode("utf-8").splitlines()
            if line.strip()
        ]


def request_status(url: str, token: str, *, method: str) -> int:
    request = urllib.request.Request(
        url,
        headers={"X-LLM-Studio-Token": token},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status


def assert_unauthorized(url: str) -> None:
    try:
        urllib.request.urlopen(url, timeout=5)
    except urllib.error.HTTPError as error:
        if error.code == 401:
            return
        raise
    raise AssertionError("Protected endpoint accepted a request without the runtime token")


def assert_token_rejected(url: str, token: str) -> None:
    try:
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"X-LLM-Studio-Token": token}),
            timeout=5,
        )
    except urllib.error.HTTPError as error:
        if error.code == 401:
            return
        raise
    raise AssertionError("Protected endpoint accepted the previous launch token")


def assert_http_status(url: str, expected: int, *, token: str | None = None) -> None:
    headers = {} if token is None else {"X-LLM-Studio-Token": token}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers),
            timeout=5,
        ) as response:
            actual = response.status
    except urllib.error.HTTPError as error:
        actual = error.code
    assert actual == expected, f"{url} returned HTTP {actual}, expected {expected}"


def terminate_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


if __name__ == "__main__":
    main()
