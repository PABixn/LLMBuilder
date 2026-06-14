from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "desktop" / "audit_dependencies.py"
SPEC = importlib.util.spec_from_file_location("desktop_audit_dependencies", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_dependencies = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_dependencies)


def test_python_audit_accepts_only_reviewed_newer_torch_finding() -> None:
    policy = audit_dependencies.load_policy(
        ROOT / "scripts" / "desktop" / "dependency-audit-policy.json"
    )
    payload = {
        "dependencies": [
            {
                "name": "torch",
                "version": "2.12.0",
                "vulns": [{"id": "CVE-2025-3000", "aliases": [], "fix_versions": []}],
            },
            {
                "name": "example",
                "version": "1.0",
                "vulns": [{"id": "CVE-2099-0001", "aliases": [], "fix_versions": ["1.1"]}],
            },
        ]
    }

    accepted, blocking = audit_dependencies.classify_python_findings(payload, policy)

    assert [(item["package"], item["id"]) for item in accepted] == [
        ("torch", "CVE-2025-3000")
    ]
    assert [(item["package"], item["id"]) for item in blocking] == [
        ("example", "CVE-2099-0001")
    ]
    assert accepted[0]["source"] == "https://api.osv.dev/v1/vulns/CVE-2025-3000"


def test_python_audit_does_not_accept_last_affected_torch_version() -> None:
    policy = audit_dependencies.load_policy(
        ROOT / "scripts" / "desktop" / "dependency-audit-policy.json"
    )
    payload = {
        "dependencies": [
            {
                "name": "torch",
                "version": "2.6.0",
                "vulns": [{"id": "CVE-2025-3000", "aliases": [], "fix_versions": []}],
            }
        ]
    }

    accepted, blocking = audit_dependencies.classify_python_findings(payload, policy)

    assert accepted == []
    assert [(item["package"], item["id"]) for item in blocking] == [
        ("torch", "CVE-2025-3000")
    ]


def test_python_site_packages_preserves_virtualenv_identity() -> None:
    assert audit_dependencies.python_site_packages(Path(sys.executable)) == Path(
        sysconfig.get_paths()["purelib"]
    ).resolve()


def test_python_site_packages_probe_disables_bytecode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess([], 0, stdout=f"{tmp_path}\n", stderr="")

    monkeypatch.setattr(audit_dependencies.subprocess, "run", fake_run)

    assert audit_dependencies.python_site_packages(Path(sys.executable)) == tmp_path
    assert captured["PYTHONDONTWRITEBYTECODE"] == "1"


def test_cargo_binary_audit_requires_complete_auditable_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    binary = tmp_path / "llm_studio_desktop"
    binary.write_bytes(b"binary")
    payload = {
        "database": {"advisory-count": 1},
        "lockfile": {"dependency-count": 71},
        "vulnerabilities": {"found": False, "list": []},
        "warnings": {},
    }

    monkeypatch.setattr(
        audit_dependencies.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(payload),
            stderr=(
                "warning: binary was not built with 'cargo auditable', "
                "the report will be incomplete"
            ),
        ),
    )

    report = audit_dependencies.audit_cargo_binary(binary, {"cargo_audit": {}})

    assert report["auditable"] is False
    assert report["passed"] is False
    assert report["dependency_count"] == 71


def test_cargo_binary_audit_accepts_complete_vulnerability_free_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    binary = tmp_path / "llm_studio_desktop"
    binary.write_bytes(b"binary")
    payload = {
        "database": {"advisory-count": 1},
        "lockfile": {"dependency-count": 291},
        "vulnerabilities": {"found": False, "list": []},
        "warnings": {"unmaintained": []},
    }

    monkeypatch.setattr(
        audit_dependencies.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )

    report = audit_dependencies.audit_cargo_binary(binary, {"cargo_audit": {}})

    assert report["auditable"] is True
    assert report["passed"] is True
    assert report["dependency_count"] == 291
