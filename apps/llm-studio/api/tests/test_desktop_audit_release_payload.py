from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "desktop" / "audit_release_payload.py"
SPEC = importlib.util.spec_from_file_location("desktop_audit_release_payload", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_release_payload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_release_payload)


def test_release_payload_audit_accepts_clean_verified_runtime(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    source = runtime / "source" / "app.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('clean release')\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (runtime / "manifest.json").write_text(
        json.dumps({"file_hashes": {"source/app.py": digest}}),
        encoding="utf-8",
    )

    report = audit_release_payload.audit_payloads([runtime])

    assert report["file_count"] == 2
    assert report["findings"] == []


def test_release_payload_audit_finds_secrets_development_files_and_hash_drift(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    cache = runtime / "__pycache__"
    cache.mkdir(parents=True)
    secret = cache / "secret.pyc"
    secret.write_bytes(b"LLM_STUDIO_RUNTIME_TOKEN=0123456789abcdef0123456789abcdef")
    (runtime / "manifest.json").write_text(
        json.dumps({"file_hashes": {"__pycache__/secret.pyc": "0" * 64}}),
        encoding="utf-8",
    )

    report = audit_release_payload.audit_payloads([runtime])
    kinds = {finding["kind"] for finding in report["findings"]}

    assert "forbidden-name" in kinds
    assert "forbidden-suffix" in kinds
    assert "runtime-token-assignment" in kinds
    assert "runtime-hash-mismatch" in kinds


def test_release_payload_audit_rejects_nonportable_manifest_paths(tmp_path: Path) -> None:
    unsafe_paths = [
        "",
        ".",
        "../outside",
        "source/../outside",
        "/etc/passwd",
        "//server/share",
        r"\server\share",
        r"C:\secret",
        "C:/secret",
        "C:secret",
        "source\\secret",
        "source//secret",
        "source/./secret",
    ]
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "manifest.json").write_text(
        json.dumps({"file_hashes": {path: "0" * 64 for path in unsafe_paths}}),
        encoding="utf-8",
    )

    report = audit_release_payload.audit_payloads([runtime])

    assert len(report["findings"]) == len(unsafe_paths)
    assert {finding["kind"] for finding in report["findings"]} == {
        "unsafe-runtime-manifest-path"
    }


def test_release_payload_audit_rejects_nonportable_runtime_entrypoints(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "manifest.json").write_text(
        json.dumps(
            {
                "file_hashes": {},
                "python_executable": r"C:\outside\python.exe",
                "source_root": 42,
            }
        ),
        encoding="utf-8",
    )

    report = audit_release_payload.audit_payloads([runtime])

    assert {
        (finding["kind"], finding["path"]) for finding in report["findings"]
    } == {
        ("unsafe-runtime-manifest-path", "runtime/python_executable"),
        ("unsafe-runtime-manifest-path", "runtime/source_root"),
    }


def test_release_payload_audit_finds_symlink_and_developer_path(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    target = payload / "target.txt"
    target.write_bytes(b"/Users/developer/project")
    link = payload / "link.txt"
    link.symlink_to(target)

    report = audit_release_payload.audit_payloads([payload])
    kinds = {finding["kind"] for finding in report["findings"]}

    assert "symlink" in kinds
    assert "macos-developer-home" in kinds


def test_release_payload_audit_distinguishes_code_from_assigned_secrets(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    source = payload / "source.py"
    source.write_text(
        'api_key = settings.runpod_api_key\n'
        'headers["Authorization"] = f"Bearer {token}"\n'
        'localStorage.setItem("llm-studio-tokenizer-form", tokenizerForm)\n',
        encoding="utf-8",
    )
    secret = payload / ".env"
    secret.write_text(
        "RUNPOD_API_KEY=rpa_0123456789abcdef0123456789abcdef\n",
        encoding="utf-8",
    )

    report = audit_release_payload.audit_payloads([payload])

    assert {
        (finding["kind"], finding["path"]) for finding in report["findings"]
    } == {
        ("forbidden-name", "payload/.env"),
        ("provider-key-assignment", "payload/.env"),
    }


def test_release_payload_audit_rejects_web_storage_secret_keys(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "app.js").write_text(
        'localStorage.setItem("llm-studio-runtime-token", runtimeToken);\n'
        'sessionStorage.setItem("runpod-api-key", apiKey);\n',
        encoding="utf-8",
    )

    report = audit_release_payload.audit_payloads([payload])

    assert {
        (finding["kind"], finding["path"]) for finding in report["findings"]
    } == {("web-storage-secret-key", "payload/app.js")}


def test_release_payload_console_report_bounds_large_finding_output() -> None:
    report = {
        "schema_version": 1,
        "findings": [{"kind": "test", "path": str(index)} for index in range(150)],
    }

    displayed = audit_release_payload.console_report(report)

    assert len(displayed["findings"]) == audit_release_payload.MAX_CONSOLE_FINDINGS
    assert displayed["findings_omitted_from_console"] == 50
