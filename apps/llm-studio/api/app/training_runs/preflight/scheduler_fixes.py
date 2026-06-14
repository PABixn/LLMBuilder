from __future__ import annotations

from typing import Any

from ..schemas import TrainingFixSuggestion, TrainingIssue
from .config_validation import issue
from ...runtime_paths import ensure_source_root_on_path

IMPORT_ROOT = ensure_source_root_on_path()

from training.training_config import TrainingConfig


def add_scheduler_step_fix(
    fixes: list[TrainingFixSuggestion],
    training_config: dict[str, Any],
) -> None:
    max_steps = training_config.get("max_steps")
    scheduler = training_config.get("lr_scheduler")
    schedulers = scheduler.get("schedulers") if isinstance(scheduler, dict) else None
    if not isinstance(max_steps, int) or max_steps <= 0 or not isinstance(schedulers, list):
        return

    current_total = 0
    for item in schedulers:
        if isinstance(item, dict) and isinstance(item.get("steps"), int):
            current_total += int(item["steps"])
    if current_total == max_steps:
        return

    fixes.append(
        TrainingFixSuggestion(
            code="match_scheduler_steps_to_max_steps",
            label="Match scheduler to max_steps",
            description=f"Replace the LR scheduler with a warmup/cosine schedule totaling {max_steps} steps.",
            path="training_config.lr_scheduler",
            value=build_default_scheduler(max_steps),
        )
    )


def build_default_scheduler(max_steps: int) -> dict[str, Any]:
    if max_steps <= 1:
        return {
            "type": "sequential",
            "schedulers": [{"type": "constant", "steps": 1, "factor": 1.0}],
        }

    warmup_steps = max(1, min(50, max_steps // 10))
    decay_steps = max_steps - warmup_steps
    if decay_steps <= 0:
        return {
            "type": "sequential",
            "schedulers": [
                {
                    "type": "linear",
                    "steps": max_steps,
                    "start_factor": 0.1,
                    "end_factor": 1.0,
                }
            ],
        }

    return {
        "type": "sequential",
        "schedulers": [
            {
                "type": "linear",
                "steps": warmup_steps,
                "start_factor": 0.1,
                "end_factor": 1.0,
            },
            {
                "type": "cosine_annealing",
                "steps": decay_steps,
                "eta_min": 1e-5,
            },
        ],
    }


def collect_training_config_warnings_and_fixes(
    training_config: TrainingConfig,
) -> tuple[list[TrainingIssue], list[TrainingFixSuggestion]]:
    warnings: list[TrainingIssue] = []
    fixes: list[TrainingFixSuggestion] = []

    sparse_checkpoint_threshold = cadence_for_fraction(training_config.max_steps, 0.2)
    if training_config.save_every > sparse_checkpoint_threshold:
        warnings.append(
            issue(
                "save_every_sparse",
                f"save_every is larger than 20% of max_steps ({sparse_checkpoint_threshold}), so checkpoints will be sparse.",
                "$.training_config.save_every",
                severity="warning",
            )
        )
        suggested_save_every = cadence_for_fraction(training_config.max_steps, 0.1)
        fixes.append(
            TrainingFixSuggestion(
                code="set_save_every_to_periodic_cadence",
                label="Save every 10% of the run",
                description=f"Set save_every to {suggested_save_every}, creating roughly 10 checkpoints across this run.",
                path="training_config.save_every",
                value=int(suggested_save_every),
            )
        )

    if training_config.sample_every > training_config.max_steps:
        warnings.append(
            issue(
                "sample_every_exceeds_max_steps",
                "sample_every is larger than max_steps, so no intermediate samples will be generated.",
                "$.training_config.sample_every",
                severity="warning",
            )
        )
        suggested_sample_every = cadence_for_fraction(training_config.max_steps, 0.2)
        fixes.append(
            TrainingFixSuggestion(
                code="set_sample_every_to_run_cadence",
                label="Sample during the run",
                description=f"Set sample_every to {suggested_sample_every} so samples appear before training completes.",
                path="training_config.sample_every",
                value=int(suggested_sample_every),
            )
        )

    if training_config.optimizer.lr >= 0.01:
        warnings.append(
            issue(
                "optimizer_lr_high",
                "Learning rate is unusually high for transformer training and may destabilize the run.",
                "$.training_config.optimizer.lr",
                severity="warning",
            )
        )
        fixes.append(
            TrainingFixSuggestion(
                code="set_optimizer_lr_to_starter_safe_value",
                label="Use a safer learning rate",
                description="Set optimizer.lr to 3e-4, a conservative starter value for this trainer.",
                path="training_config.optimizer.lr",
                value=0.0003,
            )
        )

    return warnings, fixes


def runtime_batch_fixes(
    error_message: str,
    training_config: TrainingConfig,
    max_memory_batch_size: int,
) -> list[TrainingFixSuggestion]:
    fixes: list[TrainingFixSuggestion] = []
    if training_config.micro_batch_size is not None:
        fixes.append(
            TrainingFixSuggestion(
                code="auto_select_micro_batch_size",
                label="Auto-select micro batch size",
                description="Remove micro_batch_size so preflight can choose the largest valid value for memory and accumulation.",
                path="training_config.micro_batch_size",
                value=None,
            )
        )

    if "total_batch_size must be divisible by seq_len" in error_message:
        nearest_total_batch_size = nearest_divisible_total_batch_size(
            training_config.total_batch_size,
            training_config.seq_len,
        )
        if nearest_total_batch_size != training_config.total_batch_size:
            fixes.append(
                TrainingFixSuggestion(
                    code="make_total_batch_size_divisible",
                    label="Make total batch size divisible",
                    description=f"Set total_batch_size to {nearest_total_batch_size}, the nearest multiple of seq_len.",
                    path="training_config.total_batch_size",
                    value=nearest_total_batch_size,
                )
            )

    if "micro_batch_size exceeds the memory-estimated maximum" in error_message and max_memory_batch_size > 0:
        fixes.append(
            TrainingFixSuggestion(
                code="cap_micro_batch_size_to_memory",
                label="Fit micro batch size to memory",
                description=f"Set micro_batch_size to the memory-estimated maximum ({max_memory_batch_size}).",
                path="training_config.micro_batch_size",
                value=max_memory_batch_size,
            )
        )

    return fixes


def nearest_divisible_total_batch_size(total_batch_size: int, seq_len: int) -> int:
    if seq_len <= 0:
        return total_batch_size
    lower = max(seq_len, (total_batch_size // seq_len) * seq_len)
    upper = lower if lower == total_batch_size else lower + seq_len
    if abs(total_batch_size - lower) <= abs(upper - total_batch_size):
        return lower
    return upper


def cadence_for_fraction(max_steps: int, fraction: float) -> int:
    return max(1, min(max_steps, round(max_steps * fraction)))
