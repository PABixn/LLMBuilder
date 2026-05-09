from __future__ import annotations

import re

JOB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")
PROJECT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")


def validate_identifier(value: str, pattern: re.Pattern[str]) -> None:
    if not pattern.fullmatch(value):
        raise KeyError(value)
