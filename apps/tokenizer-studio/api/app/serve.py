from __future__ import annotations

import os

import uvicorn

from .config import get_settings


def _read_access_log_default() -> bool:
    raw = os.getenv("TOKENIZER_STUDIO_ACCESS_LOG", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        access_log=_read_access_log_default(),
        log_level=os.getenv("TOKENIZER_STUDIO_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    run()
