from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest
from fastapi import HTTPException


REPO_ROOT = Path(__file__).resolve().parents[4]
LLM_STUDIO_ROOT = REPO_ROOT / "apps" / "llm-studio"
if str(LLM_STUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_STUDIO_ROOT))

from remote_agent.app import ranged_file  # noqa: E402
from remote_agent.bundle import extract_bundle, safe_join  # noqa: E402
from remote_agent.runner import RemoteTrainingRunner  # noqa: E402


def test_safe_join_blocks_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc_info:
        safe_join(tmp_path, "../outside.txt")

    assert exc_info.value.status_code == 403


def test_bundle_extraction_rejects_unsafe_tar_members(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        payload = b"unsafe"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(HTTPException) as exc_info:
        extract_bundle(
            buffer.getvalue(),
            content_type="application/gzip",
            incoming_path=tmp_path / "incoming" / "job.bundle",
            job_root=tmp_path / "jobs" / "job123456",
        )

    assert exc_info.value.status_code == 403


def test_ranged_file_reports_size_and_start_offset_headers(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    path.write_text("abcdef", encoding="utf-8")

    partial = ranged_file(path, 2)
    eof = ranged_file(path, 6)

    assert partial.body == b"cdef"
    assert partial.headers["X-File-Size"] == "6"
    assert partial.headers["X-Start-Offset"] == "2"
    assert eof.body == b""
    assert eof.headers["X-File-Size"] == "6"
    assert eof.headers["X-Start-Offset"] == "6"


def test_cancel_without_process_is_noop() -> None:
    runner = RemoteTrainingRunner()

    runner.cancel("job123456")

    assert runner.status("job123456") == {"process_id": None, "exit_code": None, "running": False}
