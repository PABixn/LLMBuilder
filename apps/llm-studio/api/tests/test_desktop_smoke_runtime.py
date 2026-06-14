from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "desktop" / "smoke_runtime.py"
SPEC = importlib.util.spec_from_file_location("desktop_smoke_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke_runtime)


def test_immutable_tree_snapshot_detects_content_metadata_and_structure_changes(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    resource = runtime / "resource.txt"
    resource.write_text("original", encoding="utf-8")
    snapshot = smoke_runtime.immutable_tree_snapshot(runtime)

    smoke_runtime.assert_immutable_tree_unchanged(runtime, snapshot)

    resource.write_text("changed", encoding="utf-8")
    with pytest.raises(AssertionError, match=r"changed=.*resource\.txt"):
        smoke_runtime.assert_immutable_tree_unchanged(runtime, snapshot)

    resource.write_text("original", encoding="utf-8")
    resource.chmod(0o600)
    with pytest.raises(AssertionError, match=r"changed=.*resource\.txt"):
        smoke_runtime.assert_immutable_tree_unchanged(runtime, snapshot)

    (runtime / "added.txt").write_text("added", encoding="utf-8")
    with pytest.raises(AssertionError, match=r"added=.*added\.txt"):
        smoke_runtime.assert_immutable_tree_unchanged(runtime, snapshot)


def test_immutable_tree_snapshot_does_not_follow_packaged_symlinks(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    target = outside / "target.txt"
    target.write_text("original", encoding="utf-8")
    (runtime / "linked.txt").symlink_to(target)
    snapshot = smoke_runtime.immutable_tree_snapshot(runtime)

    target.write_text("changed outside the packaged runtime", encoding="utf-8")

    smoke_runtime.assert_immutable_tree_unchanged(runtime, snapshot)


def test_rss_parsers_and_current_process_measurement() -> None:
    assert smoke_runtime.parse_linux_rss_bytes("Name:\tpython\nVmRSS:\t  123 kB\n") == 125_952
    assert smoke_runtime.parse_ps_rss_bytes(" 456\n") == 466_944

    measurement = smoke_runtime.sample_process_rss(
        os.getpid(),
        sample_count=1,
        interval_seconds=0,
    )
    assert measurement["status"] in {"measured", "unavailable"}
    if measurement["status"] == "measured":
        assert measurement["median_bytes"] > 0
    else:
        assert measurement["failure_types"]


def test_sample_process_rss_reports_partial_measurements_without_error_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: list[int | OSError] = [100, OSError("/private/secret/path"), 300]

    def fake_measure(_pid: int) -> int:
        value = values.pop(0)
        if isinstance(value, OSError):
            raise value
        return value

    monkeypatch.setattr(smoke_runtime, "measure_process_rss_bytes", fake_measure)

    measurement = smoke_runtime.sample_process_rss(123, sample_count=3, interval_seconds=0)

    assert measurement == {
        "status": "measured",
        "requested_samples": 3,
        "measured_samples": 2,
        "unavailable_samples": 1,
        "samples_bytes": [100, 300],
        "minimum_bytes": 100,
        "median_bytes": 200,
        "maximum_bytes": 300,
    }
    assert "/private/secret/path" not in json.dumps(measurement)


def test_performance_report_is_structured_characterization_without_paths() -> None:
    report = smoke_runtime.build_performance_report(
        {
            "platform": "macos",
            "architecture": "aarch64",
            "runtime_version": "1.2.3",
            "python_version": "3.12.7",
            "provenance": {
                "build_mode": "portable",
                "builder_path": "/private/developer/path",
            },
            "size": {
                "payload_file_count": 100,
                "payload_total_bytes": 200,
                "max_payload_bytes": 300,
            },
        },
        {"full_smoke_ms": 42.5},
    )

    assert report["schema_version"] == 1
    assert report["classification"] == "characterization-only"
    assert report["runtime"] == {
        "runtime_version": "1.2.3",
        "python_version": "3.12.7",
        "build_mode": "portable",
        "size": {
            "payload_file_count": 100,
            "payload_total_bytes": 200,
        },
    }
    assert report["scope"]["route_navigation"] == "excluded-by-product-owner-no-browser-checks"
    assert "/private/developer/path" not in json.dumps(report)


def test_write_json_atomic_replaces_report_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    report = tmp_path / "nested" / "performance.json"
    report.parent.mkdir()
    report.write_text('{"stale": true}\n', encoding="utf-8")

    smoke_runtime.write_json_atomic(report, {"schema_version": 1})

    assert json.loads(report.read_text(encoding="utf-8")) == {"schema_version": 1}
    assert list(report.parent.iterdir()) == [report]
