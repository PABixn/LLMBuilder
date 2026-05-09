from __future__ import annotations

from pydantic import ValidationError

from ..schemas import TrainingIssue


def issue(code: str, message: str, path: str, *, severity: str = "error") -> TrainingIssue:
    return TrainingIssue(code=code, message=message, path=path, severity=severity)


def validation_issues(code: str, base_path: str, exc: ValidationError) -> list[TrainingIssue]:
    items = exc.errors(include_url=False, include_input=False, include_context=False)
    if not items:
        return [issue(code, "Validation failed.", base_path)]
    return [
        issue(
            code,
            humanize_validation_message(str(item.get("msg") or "Validation failed.")),
            json_path(base_path, item.get("loc")),
        )
        for item in items
    ]


def humanize_validation_message(message: str) -> str:
    cleaned = message.strip()
    if cleaned.startswith("Value error, "):
        cleaned = cleaned.removeprefix("Value error, ").strip()
    if cleaned == "sum of lr_scheduler steps must equal max_steps":
        return (
            "LR scheduler steps must add up to max_steps. "
            "Use the suggested fix below or edit training_config.lr_scheduler."
        )
    return cleaned[:1].upper() + cleaned[1:] if cleaned else "Validation failed."


def json_path(base_path: str, loc: object) -> str:
    if not isinstance(loc, (list, tuple)) or not loc:
        return base_path

    path = base_path
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        elif isinstance(part, str) and part:
            path += f".{part}"
    return path
