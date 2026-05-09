from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schemas import load_json
from .schemas import (
    TrainingCheckpointEntry,
    TrainingMetricPoint,
    TrainingSampleEntry,
    TrainingSampleText,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            items.append(parsed)
    return items


def latest_metric_updates(stats_path: Path) -> dict[str, Any]:
    payloads = read_jsonl(stats_path)
    if not payloads:
        return {}
    latest = payloads[-1]
    updates: dict[str, Any] = {}
    if "step" in latest:
        updates["last_step"] = int(latest["step"])
    if "loss" in latest:
        updates["latest_loss"] = None if latest["loss"] is None else float(latest["loss"])
    if "norm" in latest:
        updates["latest_grad_norm"] = None if latest["norm"] is None else float(latest["norm"])
    if "lr" in latest:
        updates["latest_lr"] = None if latest["lr"] is None else float(latest["lr"])
    if "tok_per_sec" in latest:
        updates["latest_tokens_per_sec"] = (
            None if latest["tok_per_sec"] is None else float(latest["tok_per_sec"])
        )
    return updates


def read_metric_points(stats_path: Path, *, limit: int | None = None) -> list[TrainingMetricPoint]:
    payloads = read_jsonl(stats_path)
    if limit is not None and limit > 0:
        payloads = payloads[-limit:]
    return [TrainingMetricPoint.model_validate(item) for item in payloads]


def read_sample_entries(samples_path: Path, *, limit: int = 50) -> list[TrainingSampleEntry]:
    payloads = read_jsonl(samples_path)[-max(limit, 1) :]
    samples: list[TrainingSampleEntry] = []
    for item in payloads:
        entries = [
            TrainingSampleText.model_validate(sample)
            for sample in item.get("samples", [])
            if isinstance(sample, dict)
        ]
        samples.append(
            TrainingSampleEntry(
                step=int(item.get("step", 0)),
                samples=entries,
            )
        )
    return samples


def read_training_log_lines(
    *,
    artifact_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    executor_kind: str | None,
    lines: int | None,
) -> tuple[list[str], list[str]]:
    stdout_lines = tail_lines(stdout_path, lines)
    if executor_kind == "runpod_pod":
        runpod_lines: list[str] = []
        for log_name in (
            "runpod_lifecycle.log",
            "runpod_startup.log",
            "runpod_agent.log",
            "runpod_runner.log",
        ):
            runpod_lines.extend(tail_lines(artifact_dir / log_name, lines))
        stdout_lines = runpod_lines + stdout_lines
        if lines is not None and lines > 0:
            stdout_lines = stdout_lines[-lines:]
    return stdout_lines, tail_lines(stderr_path, lines)


def list_checkpoint_entries(artifact_dir: Path) -> list[TrainingCheckpointEntry]:
    checkpoints_root = artifact_dir / "checkpoints"
    checkpoints: list[TrainingCheckpointEntry] = []
    if checkpoints_root.exists():
        for candidate in sorted(
            (path for path in checkpoints_root.iterdir() if path.is_dir()),
            key=lambda path: int(path.name) if path.name.isdigit() else -1,
            reverse=True,
        ):
            files = sorted(str(path.name) for path in candidate.iterdir() if path.is_file())
            checkpoints.append(
                TrainingCheckpointEntry(
                    step=int(candidate.name) if candidate.name.isdigit() else 0,
                    directory=str(candidate),
                    created_at=datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc),
                    size_bytes=directory_size(candidate),
                    files=files,
                )
            )
    return checkpoints


def tail_lines(path: Path, lines: int | None) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if lines is None or lines <= 0:
        return text.splitlines()
    return text.splitlines()[-max(lines, 1) :]


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
