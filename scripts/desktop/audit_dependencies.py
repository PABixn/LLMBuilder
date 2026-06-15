#!/usr/bin/env python3
"""Audit desktop Python and Rust dependencies with a narrow reviewed policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
from typing import Any

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = Path(__file__).with_name("dependency-audit-policy.json")
DEFAULT_CARGO_LOCK = ROOT / "apps" / "llm-studio" / "desktop" / "src-tauri" / "Cargo.lock"
SUPPORTED_RUNTIME_MANIFEST_SCHEMA = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--cargo-lock", type=Path, default=DEFAULT_CARGO_LOCK)
    parser.add_argument("--cargo-binary", type=Path, action="append", default=[])
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--skip-cargo", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_python and args.skip_cargo and not args.cargo_binary:
        raise SystemExit("At least one dependency ecosystem must be audited.")

    policy = load_policy(args.policy)
    report: dict[str, Any] = {"schema_version": 1}
    blockers: list[str] = []
    if not args.skip_python:
        python_report = audit_python(args.python, policy)
        report["python"] = python_report
        blockers.extend(
            f"Python: {finding['package']} {finding['version']} / {finding['id']}"
            for finding in python_report["blocking_findings"]
        )
    if not args.skip_cargo:
        cargo_report = audit_cargo(args.cargo_lock, policy)
        report["cargo"] = cargo_report
        if not cargo_report["passed"]:
            blockers.append("Rust: cargo-audit reported one or more vulnerabilities")
    if args.cargo_binary:
        binary_reports = [
            audit_cargo_binary(binary, policy)
            for binary in args.cargo_binary
        ]
        report["cargo_binaries"] = binary_reports
        for binary_report in binary_reports:
            if not binary_report["auditable"]:
                blockers.append(
                    f"Rust binary: {binary_report['binary']} lacks complete cargo-auditable metadata"
                )
            elif not binary_report["passed"]:
                blockers.append(
                    f"Rust binary: {binary_report['binary']} contains one or more vulnerabilities"
                )

    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if blockers:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("Dependency audit failed:\n- " + "\n- ".join(blockers))

    accepted = len(report.get("python", {}).get("accepted_findings", []))
    binaries = len(report.get("cargo_binaries", []))
    print(
        "Dependency audit passed; "
        f"reviewed Python findings: {accepted}; Rust vulnerabilities: 0; "
        f"auditable Rust binaries: {binaries}."
    )


def load_policy(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        policy = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Dependency audit policy is unavailable or invalid: {resolved}: {error}") from error
    if policy.get("schema_version") != 1:
        raise SystemExit(f"Unsupported dependency audit policy schema: {policy.get('schema_version')!r}")
    for entry in policy.get("pip_audit_accepted_findings", []):
        required = {
            "package",
            "vulnerability_id",
            "installed_version_min_exclusive",
            "rationale",
            "source",
            "reviewed_on",
        }
        missing = sorted(required - entry.keys())
        if missing:
            raise SystemExit(f"Dependency audit policy entry is missing: {', '.join(missing)}")
        if not str(entry["source"]).startswith("https://api.osv.dev/v1/vulns/"):
            raise SystemExit("Accepted pip-audit findings require a direct OSV vulnerability source.")
    for entry in policy.get("pip_audit_local_version_normalizations", []):
        required = {
            "package",
            "local_version",
            "rationale",
            "required_runtime_provenance",
            "source",
        }
        missing = sorted(required - entry.keys())
        if missing:
            raise SystemExit(
                "Dependency audit local-version policy entry is missing: "
                + ", ".join(missing)
            )
        if not isinstance(entry["required_runtime_provenance"], dict) or not entry[
            "required_runtime_provenance"
        ]:
            raise SystemExit(
                "Dependency audit local-version policy requires non-empty runtime provenance."
            )
        if not str(entry["source"]).startswith("https://"):
            raise SystemExit("Dependency audit local-version policy requires an HTTPS source.")
    return policy


def audit_python(python: Path, policy: dict[str, Any]) -> dict[str, Any]:
    # Resolving a venv's Python symlink changes sys.prefix and audits the base
    # interpreter instead of the requested environment.
    python = python.expanduser().absolute()
    if not python.is_file():
        raise SystemExit(f"Python audit target does not exist: {python}")
    site_packages = python_site_packages(python)
    inventory = python_dependency_inventory(python)
    requirements, normalized, manifest_path = prepare_python_audit_inventory(
        python,
        inventory,
        policy,
    )
    with tempfile.TemporaryDirectory(prefix="llm-studio-pip-audit-") as temporary:
        requirements_path = Path(temporary) / "requirements.txt"
        requirements_path.write_text(
            "".join(
                f"{package['name']}=={package['audited_version']}\n"
                for package in requirements.values()
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "pip_audit",
            "--requirement",
            str(requirements_path),
            "--no-deps",
            "--disable-pip",
            "--strict",
            "--progress-spinner",
            "off",
            "--desc",
            "off",
            "--format",
            "json",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode not in {0, 1}:
        raise SystemExit(f"pip-audit failed ({result.returncode}): {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        detail = result.stderr.strip() or result.stdout.strip() or "no scanner output"
        raise SystemExit(f"pip-audit returned invalid JSON: {error}; {detail}") from error
    if not isinstance(payload, dict):
        raise SystemExit("pip-audit JSON output must be an object.")
    validate_python_audit_coverage(payload, requirements)
    restore_installed_versions(payload, normalized)
    accepted, blocking = classify_python_findings(payload, policy)
    return {
        "target_python": display_path(python),
        "site_packages": display_path(site_packages),
        "runtime_manifest": display_path(manifest_path) if manifest_path is not None else None,
        "dependency_count": len(requirements),
        "normalized_local_versions": normalized,
        "accepted_findings": accepted,
        "blocking_findings": blocking,
    }


def python_site_packages(python: Path) -> Path:
    script = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    result = subprocess.run(
        [str(python), "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    path = Path(result.stdout.strip()).resolve()
    if not path.is_dir():
        raise SystemExit(f"Python audit target site-packages does not exist: {path}")
    return path


def python_dependency_inventory(python: Path) -> dict[str, str]:
    script = (
        "import importlib.metadata,json;"
        "print(json.dumps([[d.metadata['Name'],d.version] "
        "for d in importlib.metadata.distributions() if d.metadata.get('Name')]))"
    )
    result = subprocess.run(
        [str(python), "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(f"Python dependency inventory returned invalid JSON: {error}") from error
    if not isinstance(payload, list):
        raise SystemExit("Python dependency inventory must be a list.")

    inventory: dict[str, str] = {}
    canonical_names: dict[str, str] = {}
    for item in payload:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
        ):
            raise SystemExit("Python dependency inventory contains an invalid package entry.")
        name, version = item
        canonical = canonicalize_name(name)
        if canonical in canonical_names:
            raise SystemExit(
                "Python dependency inventory contains duplicate package identities: "
                f"{canonical_names[canonical]!r} and {name!r}."
            )
        canonical_names[canonical] = name
        inventory[name] = version
    return dict(sorted(inventory.items(), key=lambda item: canonicalize_name(item[0])))


def prepare_python_audit_inventory(
    python: Path,
    inventory: dict[str, str],
    policy: dict[str, Any],
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]], Path | None]:
    manifest_path, manifest = runtime_manifest_for_python(python, inventory)
    requirements: dict[str, dict[str, str]] = {}
    normalized: list[dict[str, str]] = []
    for name, installed_version in sorted(
        inventory.items(),
        key=lambda item: canonicalize_name(item[0]),
    ):
        canonical = canonicalize_name(name)
        if canonical in requirements:
            raise SystemExit(f"Python dependency inventory contains duplicate package: {canonical}.")
        try:
            parsed = Version(installed_version)
        except InvalidVersion as error:
            raise SystemExit(
                f"Python dependency {name!r} has an invalid installed version: {installed_version!r}."
            ) from error

        audited_version = installed_version
        if parsed.local is not None:
            entry = approved_local_version_normalization(
                name,
                parsed,
                manifest,
                policy,
            )
            audited_version = parsed.public
            normalized.append(
                {
                    "package": name,
                    "installed_version": installed_version,
                    "audited_version": audited_version,
                    "local_version": parsed.local,
                    "rationale": str(entry["rationale"]),
                    "source": str(entry["source"]),
                }
            )
        requirements[canonical] = {
            "name": canonical,
            "installed_version": installed_version,
            "audited_version": audited_version,
        }
    return requirements, normalized, manifest_path


def runtime_manifest_for_python(
    python: Path,
    inventory: dict[str, str],
) -> tuple[Path | None, dict[str, Any] | None]:
    for runtime_root in list(python.parents)[:4]:
        manifest_path = runtime_root / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"Runtime manifest is unavailable or invalid: {manifest_path}: {error}"
            ) from error
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != SUPPORTED_RUNTIME_MANIFEST_SCHEMA
        ):
            raise SystemExit(f"Runtime manifest has an unsupported schema: {manifest_path}")
        expected_python = python.relative_to(runtime_root).as_posix()
        if manifest.get("python_executable") != expected_python:
            raise SystemExit(
                "Runtime manifest Python executable does not match audit target: "
                f"{manifest.get('python_executable')!r} != {expected_python!r}."
            )
        if manifest.get("dependency_versions") != inventory:
            raise SystemExit(
                "Runtime manifest dependency inventory does not match the installed audit target."
            )
        return manifest_path, manifest
    return None, None


def approved_local_version_normalization(
    package: str,
    version: Version,
    manifest: dict[str, Any] | None,
    policy: dict[str, Any],
) -> dict[str, Any]:
    provenance = manifest.get("provenance", {}) if manifest is not None else {}
    for entry in policy.get("pip_audit_local_version_normalizations", []):
        if canonicalize_name(package) != canonicalize_name(str(entry["package"])):
            continue
        if version.local != str(entry["local_version"]):
            continue
        required = entry["required_runtime_provenance"]
        if all(provenance.get(key) == value for key, value in required.items()):
            return entry
    raise SystemExit(
        "Python dependency audit refuses unreviewed local version "
        f"{package}=={version}; no matching runtime provenance policy exists."
    )


def validate_python_audit_coverage(
    payload: dict[str, Any],
    requirements: dict[str, dict[str, str]],
) -> None:
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        raise SystemExit("pip-audit JSON does not contain a dependency list.")
    audited: dict[str, str] = {}
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise SystemExit("pip-audit JSON contains an invalid dependency entry.")
        name = str(dependency.get("name", ""))
        version = str(dependency.get("version", ""))
        canonical = canonicalize_name(name)
        if not name or not version or canonical in audited:
            raise SystemExit("pip-audit JSON contains an invalid or duplicate dependency entry.")
        audited[canonical] = version

    missing = sorted(requirements.keys() - audited.keys())
    unexpected = sorted(audited.keys() - requirements.keys())
    mismatched = sorted(
        package
        for package in requirements.keys() & audited.keys()
        if not versions_equal(audited[package], requirements[package]["audited_version"])
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if unexpected:
            details.append(f"unexpected={','.join(unexpected)}")
        if mismatched:
            details.append(f"version_mismatch={','.join(mismatched)}")
        raise SystemExit(
            "pip-audit did not cover the exact installed dependency inventory: "
            + "; ".join(details)
        )


def versions_equal(left: str, right: str) -> bool:
    try:
        return Version(left) == Version(right)
    except InvalidVersion:
        return left == right


def restore_installed_versions(
    payload: dict[str, Any],
    normalized: list[dict[str, str]],
) -> None:
    installed = {
        canonicalize_name(item["package"]): item["installed_version"]
        for item in normalized
    }
    for dependency in payload.get("dependencies", []):
        canonical = canonicalize_name(str(dependency.get("name", "")))
        if canonical in installed:
            dependency["version"] = installed[canonical]


def classify_python_findings(
    payload: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    accepted: list[dict[str, str]] = []
    blocking: list[dict[str, Any]] = []
    entries = policy.get("pip_audit_accepted_findings", [])
    for dependency in payload.get("dependencies", []):
        package = str(dependency.get("name", ""))
        version = str(dependency.get("version", ""))
        for vulnerability in dependency.get("vulns", []):
            entry = accepted_policy_entry(package, version, vulnerability, entries)
            if entry is None:
                blocking.append(
                    {
                        "package": package,
                        "version": version,
                        "id": str(vulnerability.get("id", "")),
                        "aliases": sorted(str(alias) for alias in vulnerability.get("aliases", [])),
                        "fix_versions": sorted(
                            str(fix) for fix in vulnerability.get("fix_versions", [])
                        ),
                    }
                )
                continue
            accepted.append(
                {
                    "package": package,
                    "version": version,
                    "id": str(vulnerability.get("id", "")),
                    "rationale": str(entry["rationale"]),
                    "source": str(entry["source"]),
                    "reviewed_on": str(entry["reviewed_on"]),
                }
            )
    return deduplicate_findings(accepted), deduplicate_findings(blocking)


def accepted_policy_entry(
    package: str,
    installed_version: str,
    vulnerability: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    identifiers = {
        str(vulnerability.get("id", "")),
        *(str(alias) for alias in vulnerability.get("aliases", [])),
    }
    try:
        installed = Version(installed_version)
    except InvalidVersion:
        return None
    for entry in entries:
        if package.lower() != str(entry["package"]).lower():
            continue
        if str(entry["vulnerability_id"]) not in identifiers:
            continue
        try:
            last_affected = Version(str(entry["installed_version_min_exclusive"]))
        except InvalidVersion:
            continue
        if installed > last_affected:
            return entry
    return None


def audit_cargo(lock: Path, policy: dict[str, Any]) -> dict[str, Any]:
    lock = lock.expanduser().resolve()
    if not lock.is_file():
        raise SystemExit(f"Cargo audit lockfile does not exist: {lock}")
    try:
        result = subprocess.run(
            ["cargo", "audit", "--file", str(lock), "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise SystemExit("cargo-audit is required; install the pinned Cargo Audit tool.") from error
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(f"cargo-audit returned invalid JSON: {error}; {result.stderr.strip()}") from error
    return {
        "passed": result.returncode == 0 and not payload.get("vulnerabilities", {}).get("found"),
        "warning_policy": str(policy.get("cargo_audit", {}).get("warning_policy", "")),
        "database": payload.get("database", {}),
        "dependency_count": payload.get("lockfile", {}).get("dependency-count", 0),
        "vulnerabilities": [
            summarize_cargo_finding(finding)
            for finding in payload.get("vulnerabilities", {}).get("list", [])
        ],
        "warnings": [
            summarize_cargo_finding(finding)
            for findings in payload.get("warnings", {}).values()
            for finding in findings
        ],
    }


def audit_cargo_binary(binary: Path, policy: dict[str, Any]) -> dict[str, Any]:
    binary = binary.expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"Cargo audit binary does not exist: {binary}")
    try:
        result = subprocess.run(
            ["cargo", "audit", "--no-fetch", "--json", "bin", str(binary)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise SystemExit("cargo-audit is required; install the pinned Cargo Audit tool.") from error
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        detail = result.stderr.strip() or result.stdout.strip() or "no scanner output"
        raise SystemExit(f"cargo-audit bin returned invalid JSON: {error}; {detail}") from error
    incomplete_marker = "was not built with 'cargo auditable'"
    auditable = incomplete_marker not in result.stderr
    vulnerabilities = [
        summarize_cargo_finding(finding)
        for finding in payload.get("vulnerabilities", {}).get("list", [])
    ]
    warnings = [
        summarize_cargo_finding(finding)
        for findings in payload.get("warnings", {}).values()
        for finding in findings
    ]
    return {
        "binary": display_path(binary),
        "auditable": auditable,
        "passed": result.returncode == 0 and auditable and not vulnerabilities,
        "warning_policy": str(policy.get("cargo_audit", {}).get("warning_policy", "")),
        "database": payload.get("database", {}),
        "dependency_count": payload.get("lockfile", {}).get("dependency-count", 0),
        "vulnerabilities": vulnerabilities,
        "warnings": warnings,
    }


def summarize_cargo_finding(finding: dict[str, Any]) -> dict[str, Any]:
    advisory = finding.get("advisory", {})
    package = finding.get("package", {})
    return {
        "kind": str(finding.get("kind") or advisory.get("informational") or "vulnerability"),
        "id": str(advisory.get("id", "")),
        "package": str(package.get("name", "")),
        "version": str(package.get("version", "")),
        "title": str(advisory.get("title", "")),
        "patched_versions": list(finding.get("versions", {}).get("patched", [])),
        "url": str(advisory.get("url", "")),
    }


def deduplicate_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in findings:
        key = json.dumps(finding, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique


def display_path(path: Path) -> str:
    try:
        return path.absolute().relative_to(ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


if __name__ == "__main__":
    main()
