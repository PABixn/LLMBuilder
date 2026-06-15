from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "desktop" / "check_toolchains.py"
SPEC = importlib.util.spec_from_file_location("desktop_check_toolchains", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_toolchains = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_toolchains)


def test_installed_cargo_package_version_reads_pinned_install_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cargo_home = tmp_path / "cargo"
    cargo_home.mkdir()
    (cargo_home / ".crates.toml").write_text(
        '[v1]\n'
        '"cargo-auditable 0.7.4 (registry+https://github.com/rust-lang/crates.io-index)" = '
        '["cargo-auditable"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CARGO_HOME", str(cargo_home))
    monkeypatch.setattr(check_toolchains.shutil, "which", lambda _name: "/bin/tool")

    assert check_toolchains.installed_cargo_package_version("cargo-auditable") == "0.7.4"


def test_installed_cargo_package_version_requires_executable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CARGO_HOME", str(tmp_path))
    monkeypatch.setattr(check_toolchains.shutil, "which", lambda _name: None)

    assert check_toolchains.installed_cargo_package_version("cargo-auditable") is None
