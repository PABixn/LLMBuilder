from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest

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


def test_python_audit_merges_duplicate_blockers_by_vulnerability_identity() -> None:
    payload = {
        "dependencies": [
            {
                "name": "setuptools",
                "version": "70.2.0",
                "vulns": [
                    {
                        "id": "PYSEC-2025-49",
                        "aliases": ["CVE-2025-47273"],
                        "fix_versions": ["78.1.1"],
                    },
                    {
                        "id": "PYSEC-2025-49",
                        "aliases": ["GHSA-5rjg-fvgr-3xxf", "CVE-2025-47273"],
                        "fix_versions": ["78.1.1"],
                    },
                ],
            }
        ]
    }

    accepted, blocking = audit_dependencies.classify_python_findings(
        payload,
        {"pip_audit_accepted_findings": []},
    )

    assert accepted == []
    assert blocking == [
        {
            "package": "setuptools",
            "version": "70.2.0",
            "id": "PYSEC-2025-49",
            "aliases": ["CVE-2025-47273", "GHSA-5rjg-fvgr-3xxf"],
            "fix_versions": ["78.1.1"],
        }
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


def test_python_audit_normalizes_only_manifest_backed_cpu_torch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    policy = audit_dependencies.load_policy(
        ROOT / "scripts" / "desktop" / "dependency-audit-policy.json"
    )
    runtime = tmp_path / "runtime"
    python = runtime / "python" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    site_packages = runtime / "python" / "lib" / "site-packages"
    site_packages.mkdir(parents=True)
    inventory = {"example": "1.0", "torch": "2.12.0+cpu"}
    (runtime / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python_executable": "python/bin/python",
                "dependency_versions": inventory,
                "provenance": {
                    "build_mode": "portable-unlocked-development",
                    "development_torch_channel": "pytorch-cpu",
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(audit_dependencies, "python_site_packages", lambda _python: site_packages)
    monkeypatch.setattr(
        audit_dependencies,
        "python_dependency_inventory",
        lambda _python: inventory,
    )

    def fake_run(command, **_kwargs):
        requirements = Path(command[command.index("--requirement") + 1])
        captured["command"] = command
        captured["requirements"] = requirements.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "dependencies": [
                        {"name": "example", "version": "1.0", "vulns": []},
                        {
                            "name": "torch",
                            "version": "2.12.0",
                            "vulns": [
                                {
                                    "id": "CVE-2025-3000",
                                    "aliases": [],
                                    "fix_versions": [],
                                }
                            ],
                        },
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(audit_dependencies.subprocess, "run", fake_run)

    report = audit_dependencies.audit_python(python, policy)

    assert "--path" not in captured["command"]
    assert "--no-deps" in captured["command"]
    assert "--disable-pip" in captured["command"]
    assert captured["requirements"] == "example==1.0\ntorch==2.12.0\n"
    assert report["dependency_count"] == 2
    assert report["normalized_local_versions"] == [
        {
            "package": "torch",
            "installed_version": "2.12.0+cpu",
            "audited_version": "2.12.0",
            "local_version": "cpu",
            "rationale": (
                "PyPI vulnerability records identify official PyTorch CPU wheels by their "
                "public release version rather than the PEP 440 +cpu local version published "
                "on the official CPU wheel channel."
            ),
            "source": "https://download.pytorch.org/whl/cpu",
        }
    ]
    assert report["accepted_findings"][0]["version"] == "2.12.0+cpu"


@pytest.mark.parametrize(
    ("package", "version", "provenance"),
    [
        ("torch", "2.12.0+cpu", {}),
        ("example", "1.0+local", {}),
        (
            "torch",
            "2.12.0+cuda",
            {
                "build_mode": "portable-unlocked-development",
                "development_torch_channel": "pytorch-cpu",
            },
        ),
        (
            "torch",
            "2.12.0+cpu",
            {
                "build_mode": "portable-unlocked-development",
                "development_torch_channel": "unreviewed-channel",
            },
        ),
    ],
)
def test_python_audit_rejects_unreviewed_local_versions(
    tmp_path: Path,
    package: str,
    version: str,
    provenance: dict[str, str],
) -> None:
    policy = audit_dependencies.load_policy(
        ROOT / "scripts" / "desktop" / "dependency-audit-policy.json"
    )
    python = tmp_path / "runtime" / "python" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    inventory = {package: version}
    if provenance:
        (tmp_path / "runtime" / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "python_executable": "python/bin/python",
                    "dependency_versions": inventory,
                    "provenance": provenance,
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(SystemExit, match="refuses unreviewed local version"):
        audit_dependencies.prepare_python_audit_inventory(python, inventory, policy)


def test_python_audit_rejects_runtime_manifest_inventory_drift(tmp_path: Path) -> None:
    policy = audit_dependencies.load_policy(
        ROOT / "scripts" / "desktop" / "dependency-audit-policy.json"
    )
    python = tmp_path / "runtime" / "python" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    (tmp_path / "runtime" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python_executable": "python/bin/python",
                "dependency_versions": {"example": "1.0"},
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="does not match the installed audit target"):
        audit_dependencies.prepare_python_audit_inventory(
            python,
            {"example": "2.0"},
            policy,
        )


def test_python_audit_rejects_incomplete_scanner_coverage() -> None:
    with pytest.raises(SystemExit, match="missing=example"):
        audit_dependencies.validate_python_audit_coverage(
            {"dependencies": []},
            {
                "example": {
                    "name": "example",
                    "installed_version": "1.0",
                    "audited_version": "1.0",
                }
            },
        )


def test_python_audit_rejects_non_object_scanner_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    python = tmp_path / "python"
    python.write_bytes(b"python")
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    monkeypatch.setattr(audit_dependencies, "python_site_packages", lambda _python: site_packages)
    monkeypatch.setattr(
        audit_dependencies,
        "python_dependency_inventory",
        lambda _python: {"example": "1.0"},
    )
    monkeypatch.setattr(
        audit_dependencies.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            stdout="[]",
            stderr="",
        ),
    )

    with pytest.raises(SystemExit, match="must be an object"):
        audit_dependencies.audit_python(python, {"schema_version": 1})


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
