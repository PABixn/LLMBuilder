from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Response
from fastapi.responses import PlainTextResponse


def ranged_file(path: Path, offset: int) -> Response:
    if not path.exists():
        return PlainTextResponse("", media_type="text/plain")
    size = path.stat().st_size
    if offset >= size:
        return PlainTextResponse(
            "",
            media_type="text/plain",
            headers={"X-File-Size": str(size), "X-Start-Offset": str(offset)},
        )
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    return Response(
        data,
        media_type="application/octet-stream",
        headers={"X-File-Size": str(size), "X-Start-Offset": str(offset)},
    )


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None
