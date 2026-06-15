from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Callable

import uvicorn

from .config import get_settings
from .logging_config import configure_backend_logging

PARENT_PID_ENV = "LLM_STUDIO_PARENT_PID"
_PARENT_WATCH_INTERVAL_SECONDS = 0.2
logger = logging.getLogger("llm_studio.backend")
_PROCESS_STARTED_AT = time.monotonic()


def _read_access_log_default() -> bool:
    raw = os.getenv("LLM_STUDIO_ACCESS_LOG", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def run() -> None:
    os.environ["LLM_STUDIO_PROCESS_STARTED_MONOTONIC"] = str(_PROCESS_STARTED_AT)
    settings = get_settings()
    configure_backend_logging(settings.log_dir)
    expected_parent_pid = _desktop_parent_pid(settings.desktop_mode)
    with _bind_listener(settings.host, settings.port) as listener:
        actual_port = int(listener.getsockname()[1])
        _write_startup_handshake(settings.startup_handshake_path, settings.host, actual_port)

        config = uvicorn.Config(
            "app.main:app",
            host=settings.host,
            port=actual_port,
            reload=False,
            access_log=_read_access_log_default(),
            log_level=os.getenv("LLM_STUDIO_LOG_LEVEL", "info"),
        )
        server = uvicorn.Server(config)
        parent_watchdog = _start_parent_watchdog(
            expected_parent_pid,
            on_parent_exit=lambda: _request_parent_exit_shutdown(server),
        )
        try:
            server.run(sockets=[listener])
        finally:
            _stop_parent_watchdog(parent_watchdog)


def _desktop_parent_pid(desktop_mode: bool) -> int | None:
    if not desktop_mode or os.name != "posix":
        return None
    raw = os.getenv(PARENT_PID_ENV, "").strip()
    try:
        expected = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Desktop mode requires a valid {PARENT_PID_ENV} shell process id."
        ) from exc
    actual = os.getppid()
    if expected <= 1 or expected != actual:
        raise RuntimeError(
            "Desktop backend parent identity does not match the supervising shell; "
            "refusing to start an unmanaged runtime."
        )
    return expected


def _start_parent_watchdog(
    expected_parent_pid: int | None,
    *,
    on_parent_exit: Callable[[], None],
    interval_seconds: float = _PARENT_WATCH_INTERVAL_SECONDS,
) -> tuple[threading.Event, threading.Thread] | None:
    if expected_parent_pid is None:
        return None
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(interval_seconds):
            if os.getppid() != expected_parent_pid:
                on_parent_exit()
                return

    thread = threading.Thread(
        target=watch,
        name="llm-studio-parent-watchdog",
        daemon=True,
    )
    thread.start()
    return stop, thread


def _stop_parent_watchdog(
    watchdog: tuple[threading.Event, threading.Thread] | None,
) -> None:
    if watchdog is None:
        return
    stop, thread = watchdog
    stop.set()
    thread.join(timeout=1)


def _request_parent_exit_shutdown(server: uvicorn.Server) -> None:
    logger.warning(
        "Desktop shell parent exited; requesting graceful backend shutdown.",
        extra={"event_id": "backend.parent_exit.detected"},
    )
    server.should_exit = True


def _bind_listener(host: str, port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(2048)
    except OSError as exc:
        listener.close()
        requested = "an available ephemeral port" if port == 0 else f"port {port}"
        raise RuntimeError(
            f"Could not bind the local LLM Studio backend to {requested} on {host}. "
            "The port may already be in use or endpoint security may be blocking loopback access."
        ) from exc
    return listener


def _write_startup_handshake(path: Path | None, host: str, port: int) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    payload = {
        "schema_version": 1,
        "host": host,
        "port": port,
        "base_url": f"http://{host}:{port}",
        "pid": os.getpid(),
    }
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


if __name__ == "__main__":
    run()
