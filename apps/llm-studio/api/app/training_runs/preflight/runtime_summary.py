from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..schemas import DerivedRuntimeSummary, TrainingFixSuggestion, TrainingIssue
from .config_validation import issue
from .scheduler_fixes import runtime_batch_fixes
from ...runtime_paths import ensure_source_root_on_path

IMPORT_ROOT = ensure_source_root_on_path()

from model.loader import LLMConfig
from model.model import ConfigurableGPT
from training.dataloader_config import TrainingDataloaderConfig
from training.memory_estimator import MemoryEstimator, StepSizeEstimate
from training.training_config import BatchRuntimePlan, TrainingConfig, derive_batch_runtime_plan
from training.utils import resolve_training_device_type


@dataclass(slots=True)
class RuntimeSummaryResult:
    model: ConfigurableGPT
    memory_estimate: dict[str, Any] | None
    derived_runtime: DerivedRuntimeSummary | None
    recommendation_estimate: StepSizeEstimate
    batch_plan: BatchRuntimePlan | None
    errors: list[TrainingIssue]
    fixes: list[TrainingFixSuggestion]


def build_runtime_summary(
    *,
    model_config: LLMConfig,
    training_config: TrainingConfig,
    dataloader_config: TrainingDataloaderConfig,
) -> RuntimeSummaryResult:
    model = ConfigurableGPT(model_config)
    optimizer = model.setup_optimizer(
        lr=training_config.optimizer.lr,
        weight_decay=training_config.optimizer.weight_decay,
        betas=training_config.optimizer.betas,
        eps=training_config.optimizer.eps,
    )
    device = default_training_device()
    memory_estimator = MemoryEstimator(
        model=model,
        optimizer=optimizer,
        device=device,
        token_dtype=dataloader_config.token_dtype,
    )
    estimate = memory_estimator.estimate(
        seq_len=training_config.seq_len,
        batch_size=None,
    )
    recommendation_estimate = estimate
    batch_plan = None
    errors: list[TrainingIssue] = []
    fixes: list[TrainingFixSuggestion] = []

    try:
        batch_plan = derive_batch_runtime_plan(
            total_batch_size=training_config.total_batch_size,
            seq_len=training_config.seq_len,
            max_memory_batch_size=estimate.max_batch_size,
            world_size=1,
            requested_micro_batch_size=training_config.micro_batch_size,
        )
    except ValueError as exc:
        runtime_error = str(exc)
        errors.append(
            issue(
                "invalid_micro_batch_size",
                runtime_error,
                "$.training_config.total_batch_size",
            )
        )
        fixes.extend(runtime_batch_fixes(runtime_error, training_config, estimate.max_batch_size))
        memory_estimate = estimate.to_dict()
        derived_runtime = None
    else:
        configured_estimate = memory_estimator.estimate(
            seq_len=training_config.seq_len,
            batch_size=batch_plan.micro_batch_size,
        )
        recommendation_estimate = configured_estimate
        memory_estimate = configured_estimate.to_dict()
        derived_runtime = DerivedRuntimeSummary(
            device=str(device),
            device_type=device.type,
            micro_batch_size=batch_plan.micro_batch_size,
            tokens_per_micro_step=batch_plan.tokens_per_micro_step,
            tokens_per_world_step=batch_plan.tokens_per_world_step,
            grad_accum_steps=batch_plan.grad_accum_steps,
            max_batch_size_from_total=batch_plan.max_batch_size_from_total,
            max_batch_size_from_memory=batch_plan.max_batch_size_from_memory,
            max_allowed_batch_size=batch_plan.max_allowed_batch_size,
            ddp=False,
            ddp_rank=0,
            ddp_world_size=1,
        )

    return RuntimeSummaryResult(
        model=model,
        memory_estimate=memory_estimate,
        derived_runtime=derived_runtime,
        recommendation_estimate=recommendation_estimate,
        batch_plan=batch_plan,
        errors=errors,
        fixes=fixes,
    )


def default_training_device() -> torch.device:
    return torch.device(resolve_training_device_type())
