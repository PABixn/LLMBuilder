from __future__ import annotations

import pytest

from app.desktop_runtime import (
    PROCESS_STARTED_MONOTONIC_ENV,
    StartupProfiler,
    read_process_started_at,
)


def test_startup_profiler_records_stages_and_freezes_ready_time() -> None:
    profiler = StartupProfiler.begin("initializing", now=10.0)
    profiler.transition("validating_runtime", now=10.1)
    profiler.transition("migrating_data", now=10.3)
    profiler.transition("ready", ready=True, now=10.55)

    payload = profiler.payload(now=20.0)

    assert payload["ready_in_ms"] == pytest.approx(550)
    assert payload["elapsed_ms"] == pytest.approx(550)
    assert payload["stage_ms"] == {
        "initializing": pytest.approx(100),
        "validating_runtime": pytest.approx(200),
        "migrating_data": pytest.approx(250),
    }


def test_startup_profiler_reports_current_incomplete_stage() -> None:
    profiler = StartupProfiler.begin("initializing", now=5.0)
    profiler.transition("initializing_data", now=5.2)

    payload = profiler.payload(now=5.5)

    assert payload["ready_in_ms"] is None
    assert payload["elapsed_ms"] == pytest.approx(500)
    assert payload["stage_ms"]["initializing"] == pytest.approx(200)
    assert payload["stage_ms"]["initializing_data"] == pytest.approx(300)


def test_startup_profiler_can_include_process_import_time() -> None:
    profiler = StartupProfiler.begin("process_imports", now=10.0, started_at=9.25)
    profiler.transition("initializing", now=10.0)
    profiler.transition("ready", ready=True, now=10.5)

    payload = profiler.payload(now=20.0)

    assert payload["ready_in_ms"] == pytest.approx(1250)
    assert payload["stage_ms"]["process_imports"] == pytest.approx(750)
    assert payload["stage_ms"]["initializing"] == pytest.approx(500)


def test_process_start_marker_rejects_invalid_future_or_stale_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PROCESS_STARTED_MONOTONIC_ENV, "99")
    assert read_process_started_at(now=100) == 99

    monkeypatch.setenv(PROCESS_STARTED_MONOTONIC_ENV, "101")
    assert read_process_started_at(now=100) is None

    monkeypatch.setenv(PROCESS_STARTED_MONOTONIC_ENV, "-1")
    assert read_process_started_at(now=100) is None

    monkeypatch.setenv(PROCESS_STARTED_MONOTONIC_ENV, "not-a-number")
    assert read_process_started_at(now=100) is None
