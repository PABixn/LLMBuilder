#!/usr/bin/env python3
"""Audit extracted release payloads for secrets, development files, and hash drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

FORBIDDEN_NAMES = {
    ".env",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "tests",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
CONTENT_PATTERNS = {
    "private-key": re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    "runtime-token-assignment": re.compile(
        rb"(?i)LLM_STUDIO_RUNTIME_TOKEN\s*=\s*[\"']?[a-f0-9]{32,}"
    ),
    "provider-key-assignment": re.compile(
        rb"(?i)(?:RUNPOD_)?API_KEY\s*=\s*[\"']?(?:rpa|rps)_[a-z0-9_-]{16,}"
    ),
    "authorization-bearer": re.compile(
        rb"(?i)Authorization\s*:\s*Bearer\s+[a-z0-9._~+/-]{24,}"
    ),
    "web-storage-secret-key": re.compile(
        rb"""(?ix)
        (?:localStorage|sessionStorage)\s*\.\s*setItem\s*\(
        \s*["'][^"']*
        (?:runtime[-_]?token|auth[-_]?token|access[-_]?token|refresh[-_]?token|
           pod[-_]?token|runpod[-_]?key|api[-_]?key|secret)
        [^"']*["']
        """
    ),
    "macos-developer-home": re.compile(rb"/Users/[^/\x00\r\n]+/"),
    "linux-developer-home": re.compile(rb"/home/[^/\x00\r\n]+/"),
    "windows-developer-home": re.compile(rb"(?i)[A-Z]:\\Users\\[^\\\x00\r\n]+\\"),
}
CHUNK_BYTES = 1024 * 1024
PATTERN_OVERLAP = 512
MAX_CONSOLE_FINDINGS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("payloads", nargs="+", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = validate_payloads(args.payloads)
    report = audit_payloads(payloads)
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["findings"]:
        print(json.dumps(console_report(report), indent=2, sort_keys=True))
        raise SystemExit(f"Release payload audit failed with {len(report['findings'])} finding(s).")
    print(
        f"Release payload audit passed: {report['file_count']} files, "
        f"{report['total_bytes']} bytes."
    )


def validate_payloads(values: list[Path]) -> list[Path]:
    payloads: list[Path] = []
    for value in values:
        raw = value.expanduser().absolute()
        if raw.is_symlink():
            raise SystemExit(f"Release payload root must not be a symlink: {raw}")
        path = raw.resolve()
        if not path.exists():
            raise SystemExit(f"Release payload does not exist: {path}")
        payloads.append(path)
    return payloads


def audit_payloads(payloads: list[Path]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    files: list[tuple[Path, str]] = []
    for payload in payloads:
        for path, display in walk_payload(payload):
            inspect_path(path, display, findings)
            if path.is_file() and not path.is_symlink():
                files.append((path, display))
                inspect_content(path, display, findings)
        if payload.is_dir() and (payload / "manifest.json").is_file():
            verify_runtime_manifest(payload, findings)
    return {
        "schema_version": 1,
        "payload_count": len(payloads),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path, _display in files),
        "findings": findings,
    }


def console_report(report: dict[str, Any]) -> dict[str, Any]:
    findings = report["findings"]
    if len(findings) <= MAX_CONSOLE_FINDINGS:
        return report
    summary = dict(report)
    summary["findings"] = findings[:MAX_CONSOLE_FINDINGS]
    summary["findings_omitted_from_console"] = len(findings) - MAX_CONSOLE_FINDINGS
    return summary


def walk_payload(payload: Path) -> Iterable[tuple[Path, str]]:
    if payload.is_file() or payload.is_symlink():
        yield payload, payload.name
        return
    yield payload, payload.name
    for path in sorted(payload.rglob("*")):
        yield path, f"{payload.name}/{path.relative_to(payload).as_posix()}"


def inspect_path(path: Path, display: str, findings: list[dict[str, str]]) -> None:
    if path.is_symlink():
        findings.append({"kind": "symlink", "path": display})
        return
    if path.name in FORBIDDEN_NAMES:
        findings.append({"kind": "forbidden-name", "path": display})
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        findings.append({"kind": "forbidden-suffix", "path": display})


def inspect_content(path: Path, display: str, findings: list[dict[str, str]]) -> None:
    overlap = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            sample = overlap + chunk
            for kind, pattern in CONTENT_PATTERNS.items():
                if pattern.search(sample):
                    findings.append({"kind": kind, "path": display})
            overlap = sample[-PATTERN_OVERLAP:]
    deduplicate_findings(findings)


def verify_runtime_manifest(runtime: Path, findings: list[dict[str, str]]) -> None:
    try:
        manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        findings.append({"kind": "invalid-runtime-manifest", "path": runtime.name, "detail": str(exc)})
        return
    for relative, expected in manifest.get("file_hashes", {}).items():
        candidate = safe_manifest_path(runtime, str(relative))
        display = f"{runtime.name}/{relative}"
        if candidate is None:
            findings.append({"kind": "unsafe-runtime-manifest-path", "path": display})
        elif not candidate.is_file():
            findings.append({"kind": "missing-runtime-file", "path": display})
        elif sha256(candidate).lower() != str(expected).lower():
            findings.append({"kind": "runtime-hash-mismatch", "path": display})


def safe_manifest_path(runtime: Path, relative: str) -> Path | None:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    candidate = runtime / path
    try:
        resolved = candidate.resolve()
        resolved.relative_to(runtime.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def deduplicate_findings(findings: list[dict[str, str]]) -> None:
    seen: set[tuple[tuple[str, str], ...]] = set()
    unique: list[dict[str, str]] = []
    for finding in findings:
        key = tuple(sorted(finding.items()))
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    findings[:] = unique


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
