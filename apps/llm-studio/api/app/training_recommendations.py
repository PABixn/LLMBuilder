from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from llm_builder.local_text_data import is_local_text_dataset, resolve_local_data_files
from model.loader import ActivationComponent, AttentionComponent, LLMConfig, MLPComponent, NormComponent
from model.model import ConfigurableGPT
from training.dataloader_config import TrainingDataloaderConfig
from training.memory_estimator import StepSizeEstimate
from training.training_config import (
    BatchRuntimePlan,
    ConstantLRSchedulerConfig,
    CosineAnnealingLRSchedulerConfig,
    CosineAnnealingWarmRestartsLRSchedulerConfig,
    ExponentialLRSchedulerConfig,
    LinearLRSchedulerConfig,
    MultiStepLRSchedulerConfig,
    SchedulerConfig,
    StepLRSchedulerConfig,
    TrainingConfig,
)

from .training_models import (
    TrainingBatchLrRecommendation,
    TrainingBatchLrRecommendationFactor,
    TrainingBatchLrRecommendationOption,
    TrainingBatchLrRecommendationSignals,
)

_REFERENCE_PARAMETER_COUNT = 124_000_000
_REFERENCE_TOTAL_BATCH_SIZE = 524_288
_REFERENCE_LEARNING_RATE = 3e-4
_DEFAULT_BYTES_PER_TOKEN = 4.0
_STABILITY_PROFILE_LR_MULTIPLIER = 0.68
_THROUGHPUT_PROFILE_LR_MULTIPLIER = 1.28
_CANONICAL_LR_MANTISSAS = (1, 2, 3, 4, 6, 8)


@dataclass(frozen=True, slots=True)
class _ModelSummary:
    total_parameters: int
    parameter_memory_bytes_bf16: int
    estimated_kv_cache_bytes_for_context_fp16: int
    block_count: int
    attention_component_count: int
    max_mlp_multiplier: float
    activation_types: tuple[str, ...]
    norm_types: tuple[str, ...]
    uses_gqa: bool
    weight_tying: bool


@dataclass(frozen=True, slots=True)
class _DatasetSummary:
    dataset_count: int
    local_dataset_count: int
    streaming_dataset_count: int
    local_file_count: int
    local_total_size_bytes: int | None
    dominant_dataset_weight: float
    dataset_scale: str
    approx_local_tokens: int | None
    step_budget_cap_tokens: int | None
    tokenizer_bytes_per_token_assumption: float


@dataclass(frozen=True, slots=True)
class _ScheduleSummary:
    peak_factor: float
    warmup_fraction: float
    label: str


@dataclass(frozen=True, slots=True)
class _BatchCandidate:
    total_batch_size: int
    micro_batch_size: int
    grad_accum_steps: int


def build_batch_and_lr_recommendation(
    *,
    model_config: LLMConfig,
    model: ConfigurableGPT,
    training_config: TrainingConfig,
    dataloader_config: TrainingDataloaderConfig,
    tokenizer_stats: dict[str, Any] | None,
    memory_estimate: StepSizeEstimate,
    current_batch_plan: BatchRuntimePlan | None,
) -> TrainingBatchLrRecommendation | None:
    if memory_estimate.max_batch_size <= 0:
        return None

    model_summary = _summarize_model(model_config, model)
    dataset_summary = _summarize_dataset(
        config=dataloader_config,
        max_steps=training_config.max_steps,
        seq_len=training_config.seq_len,
        tokenizer_stats=tokenizer_stats,
    )
    schedule_summary = _summarize_schedule(training_config)

    preferred_grad_accum = _preferred_grad_accum_steps(
        total_parameters=model_summary.total_parameters,
        device_type=memory_estimate.device.type,
        dataset_scale=dataset_summary.dataset_scale,
    )
    max_grad_accum = _max_grad_accum_steps(
        total_parameters=model_summary.total_parameters,
        device_type=memory_estimate.device.type,
    )

    model_target_batch = _model_target_total_batch_size(
        total_parameters=model_summary.total_parameters,
        seq_len=training_config.seq_len,
        max_memory_micro_batch_size=memory_estimate.max_batch_size,
        max_grad_accum=max_grad_accum,
        dataset_cap_tokens=dataset_summary.step_budget_cap_tokens,
    )

    balanced_candidate = _pick_candidate(
        seq_len=training_config.seq_len,
        max_memory_micro_batch_size=memory_estimate.max_batch_size,
        target_total_batch_size=model_target_batch,
        preferred_grad_accum_steps=preferred_grad_accum,
        max_grad_accum=max_grad_accum,
        dataset_cap_tokens=dataset_summary.step_budget_cap_tokens,
        variant="balanced",
    )
    stability_candidate = _pick_candidate(
        seq_len=training_config.seq_len,
        max_memory_micro_batch_size=memory_estimate.max_batch_size,
        target_total_batch_size=max(training_config.seq_len, round(model_target_batch * 0.65)),
        preferred_grad_accum_steps=max(1, preferred_grad_accum - 1),
        max_grad_accum=max_grad_accum,
        dataset_cap_tokens=dataset_summary.step_budget_cap_tokens,
        variant="stability",
    )
    throughput_target = min(
        training_config.seq_len * memory_estimate.max_batch_size * max(1, preferred_grad_accum + 2),
        training_config.seq_len * memory_estimate.max_batch_size * max_grad_accum,
    )
    if dataset_summary.step_budget_cap_tokens is not None:
        throughput_target = min(throughput_target, dataset_summary.step_budget_cap_tokens)
    throughput_candidate = _pick_candidate(
        seq_len=training_config.seq_len,
        max_memory_micro_batch_size=memory_estimate.max_batch_size,
        target_total_batch_size=max(model_target_batch, throughput_target),
        preferred_grad_accum_steps=min(max_grad_accum, preferred_grad_accum + 2),
        max_grad_accum=max_grad_accum,
        dataset_cap_tokens=dataset_summary.step_budget_cap_tokens,
        variant="throughput",
    )

    options = _build_options(
        training_config=training_config,
        model_summary=model_summary,
        dataset_summary=dataset_summary,
        schedule_summary=schedule_summary,
        current_micro_batch_size=training_config.micro_batch_size,
        balanced_candidate=balanced_candidate,
        stability_candidate=stability_candidate,
        throughput_candidate=throughput_candidate,
    )
    recommended_option = next((option for option in options if option.key == "balanced"), None)
    if recommended_option is None and options:
        recommended_option = options[0]
    if recommended_option is None:
        return None

    factors = _build_factors(
        training_config=training_config,
        current_batch_plan=current_batch_plan,
        recommended_option=recommended_option,
        model_summary=model_summary,
        dataset_summary=dataset_summary,
        schedule_summary=schedule_summary,
        memory_estimate=memory_estimate,
    )

    confidence = _recommendation_confidence(
        device_type=memory_estimate.device.type,
        dataset_scale=dataset_summary.dataset_scale,
        schedule_peak_factor=schedule_summary.peak_factor,
    )

    signals = TrainingBatchLrRecommendationSignals(
        device=str(memory_estimate.device),
        device_type=memory_estimate.device.type,
        total_parameters=model_summary.total_parameters,
        parameter_memory_bytes_bf16=model_summary.parameter_memory_bytes_bf16,
        estimated_kv_cache_bytes_for_context_fp16=model_summary.estimated_kv_cache_bytes_for_context_fp16,
        block_count=model_summary.block_count,
        attention_component_count=model_summary.attention_component_count,
        max_mlp_multiplier=model_summary.max_mlp_multiplier,
        dataset_count=dataset_summary.dataset_count,
        local_dataset_count=dataset_summary.local_dataset_count,
        streaming_dataset_count=dataset_summary.streaming_dataset_count,
        local_file_count=dataset_summary.local_file_count,
        local_total_size_bytes=dataset_summary.local_total_size_bytes,
        dominant_dataset_weight=round(dataset_summary.dominant_dataset_weight, 6),
        dataset_scale=dataset_summary.dataset_scale,
        schedule_peak_factor=round(schedule_summary.peak_factor, 4),
        warmup_fraction=round(schedule_summary.warmup_fraction, 4),
        max_memory_micro_batch_size=memory_estimate.max_batch_size,
        recommended_batch_target=model_target_batch,
    )

    return TrainingBatchLrRecommendation(
        headline=(
            f"Recommended training plan: total batch {recommended_option.total_batch_size:,} tokens "
            f"and base LR {recommended_option.learning_rate:.3e}."
        ),
        summary=_build_summary(
            recommended_option=recommended_option,
            model_summary=model_summary,
            dataset_summary=dataset_summary,
            schedule_summary=schedule_summary,
            memory_estimate=memory_estimate,
            training_config=training_config,
        ),
        confidence=confidence,
        current_total_batch_size=training_config.total_batch_size,
        current_learning_rate=training_config.optimizer.lr,
        current_micro_batch_size=current_batch_plan.micro_batch_size if current_batch_plan is not None else None,
        current_grad_accum_steps=current_batch_plan.grad_accum_steps if current_batch_plan is not None else None,
        recommended_option_key=recommended_option.key,
        options=options,
        factors=factors,
        signals=signals,
    )


def _summarize_model(model_config: LLMConfig, model: ConfigurableGPT) -> _ModelSummary:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    block_count = len(model_config.blocks)
    attention_component_count = 0
    max_mlp_multiplier = 0.0
    activation_types: set[str] = set()
    norm_types: set[str] = set()
    uses_gqa = False
    kv_cache_bytes_per_token_fp16 = 0

    for block in model_config.blocks:
        for component in block.components:
            if isinstance(component, AttentionComponent):
                attention_component_count += 1
                head_dim = model_config.n_embd // component.attention.n_head
                kv_cache_bytes_per_token_fp16 += (
                    2 * component.attention.n_kv_head * head_dim * 2
                )
                uses_gqa = uses_gqa or component.attention.n_kv_head != component.attention.n_head
            elif isinstance(component, MLPComponent):
                max_mlp_multiplier = max(max_mlp_multiplier, float(component.mlp.multiplier))
                for step in component.mlp.sequence:
                    if isinstance(step, ActivationComponent):
                        activation_types.add(step.activation.type)
                    elif isinstance(step, NormComponent):
                        norm_types.add(step.norm.type)
            elif isinstance(component, ActivationComponent):
                activation_types.add(component.activation.type)
            elif isinstance(component, NormComponent):
                norm_types.add(component.norm.type)

    if max_mlp_multiplier <= 0:
        max_mlp_multiplier = 1.0

    return _ModelSummary(
        total_parameters=total_parameters,
        parameter_memory_bytes_bf16=total_parameters * 2,
        estimated_kv_cache_bytes_for_context_fp16=kv_cache_bytes_per_token_fp16 * model_config.context_length,
        block_count=block_count,
        attention_component_count=attention_component_count,
        max_mlp_multiplier=max_mlp_multiplier,
        activation_types=tuple(sorted(activation_types)),
        norm_types=tuple(sorted(norm_types)),
        uses_gqa=uses_gqa,
        weight_tying=bool(model_config.weight_tying),
    )


def _summarize_dataset(
    *,
    config: TrainingDataloaderConfig,
    max_steps: int,
    seq_len: int,
    tokenizer_stats: dict[str, Any] | None,
) -> _DatasetSummary:
    local_dataset_count = 0
    streaming_dataset_count = 0
    local_file_count = 0
    local_total_size_bytes = 0
    dominant_dataset_weight = 0.0

    for dataset in config.datasets:
        dominant_dataset_weight = max(dominant_dataset_weight, float(dataset.weight))
        if is_local_text_dataset(dataset.name, dataset.data_files, split=dataset.split):
            local_dataset_count += 1
            resolved_paths = resolve_local_data_files(dataset.data_files, split=dataset.split)
            local_file_count += len(resolved_paths)
            for path in resolved_paths:
                if path.exists() and path.is_file():
                    local_total_size_bytes += path.stat().st_size
        else:
            streaming_dataset_count += 1

    bytes_per_token = _tokenizer_bytes_per_token_assumption(tokenizer_stats)
    approx_local_tokens = None
    step_budget_cap_tokens = None
    dataset_scale = "streaming"

    if local_dataset_count > 0 and streaming_dataset_count == 0:
        approx_local_tokens = max(0, int(local_total_size_bytes / max(bytes_per_token, 1e-9)))
        updates_per_pass_target = max(8, min(max_steps, 64))
        step_budget_cap_tokens = _round_down_to_multiple(
            max(seq_len, approx_local_tokens // max(updates_per_pass_target, 1)),
            seq_len,
        )
        dataset_scale = _local_dataset_scale_label(local_total_size_bytes)
    elif local_dataset_count > 0 and streaming_dataset_count > 0:
        dataset_scale = "mixed"

    return _DatasetSummary(
        dataset_count=len(config.datasets),
        local_dataset_count=local_dataset_count,
        streaming_dataset_count=streaming_dataset_count,
        local_file_count=local_file_count,
        local_total_size_bytes=local_total_size_bytes if local_dataset_count > 0 else None,
        dominant_dataset_weight=dominant_dataset_weight,
        dataset_scale=dataset_scale,
        approx_local_tokens=approx_local_tokens,
        step_budget_cap_tokens=step_budget_cap_tokens,
        tokenizer_bytes_per_token_assumption=bytes_per_token,
    )


def _tokenizer_bytes_per_token_assumption(tokenizer_stats: dict[str, Any] | None) -> float:
    if not isinstance(tokenizer_stats, dict):
        return _DEFAULT_BYTES_PER_TOKEN
    chars_per_token = tokenizer_stats.get("chars_per_token")
    if isinstance(chars_per_token, (int, float)) and math.isfinite(chars_per_token) and chars_per_token > 0:
        return min(max(float(chars_per_token), 2.0), 12.0)
    return _DEFAULT_BYTES_PER_TOKEN


def _local_dataset_scale_label(total_size_bytes: int) -> str:
    if total_size_bytes < 5 * 1024 * 1024:
        return "tiny_local"
    if total_size_bytes < 50 * 1024 * 1024:
        return "small_local"
    if total_size_bytes < 500 * 1024 * 1024:
        return "medium_local"
    return "large_local"


def _summarize_schedule(training_config: TrainingConfig) -> _ScheduleSummary:
    schedulers = training_config.lr_scheduler.schedulers
    if not schedulers:
        return _ScheduleSummary(peak_factor=1.0, warmup_fraction=0.0, label="Constant")

    total_steps = max(training_config.max_steps, 1)
    first = schedulers[0]
    warmup_fraction = 0.0
    if (
        isinstance(first, LinearLRSchedulerConfig)
        and first.end_factor > first.start_factor
        and first.end_factor >= 1.0
    ):
        warmup_fraction = min(max(first.steps / total_steps, 0.0), 1.0)

    peak_factor = max(_scheduler_peak_factor(config) for config in schedulers)
    label = _scheduler_label(schedulers)
    return _ScheduleSummary(peak_factor=peak_factor, warmup_fraction=warmup_fraction, label=label)


def _scheduler_peak_factor(config: SchedulerConfig) -> float:
    if isinstance(config, LinearLRSchedulerConfig):
        return max(config.start_factor, config.end_factor)
    if isinstance(config, ConstantLRSchedulerConfig):
        return config.factor
    if isinstance(config, StepLRSchedulerConfig):
        if config.gamma <= 1:
            return 1.0
        return config.gamma ** max(1, config.steps // config.step_size)
    if isinstance(config, MultiStepLRSchedulerConfig):
        if config.gamma <= 1:
            return 1.0
        return config.gamma ** len(config.milestones)
    if isinstance(config, ExponentialLRSchedulerConfig):
        if config.gamma <= 1:
            return 1.0
        return config.gamma ** config.steps
    if isinstance(config, (CosineAnnealingLRSchedulerConfig, CosineAnnealingWarmRestartsLRSchedulerConfig)):
        return 1.0
    return 1.0


def _scheduler_label(schedulers: list[SchedulerConfig]) -> str:
    scheduler_types = [scheduler.type for scheduler in schedulers]
    if scheduler_types[:2] == ["linear", "cosine_annealing"]:
        return "Warmup + cosine decay"
    if scheduler_types == ["linear"]:
        return "Linear schedule"
    if any(kind == "cosine_annealing_warm_restarts" for kind in scheduler_types):
        return "Cosine warm restarts"
    if any(kind == "cosine_annealing" for kind in scheduler_types):
        return "Cosine decay"
    if any(kind in {"step", "multistep", "exponential"} for kind in scheduler_types):
        return "Stepwise decay"
    return "Sequential custom schedule"


def _preferred_grad_accum_steps(
    *,
    total_parameters: int,
    device_type: str,
    dataset_scale: str,
) -> int:
    if total_parameters < 30_000_000:
        preferred = 4
    elif total_parameters < 250_000_000:
        preferred = 8
    elif total_parameters < 1_000_000_000:
        preferred = 12
    else:
        preferred = 16

    if device_type == "mps":
        preferred = min(preferred, 8)
    elif device_type != "cuda":
        preferred = min(preferred, 6)

    if dataset_scale in {"tiny_local", "small_local"}:
        preferred = max(2, preferred - 2)
    elif dataset_scale == "medium_local":
        preferred = max(2, preferred - 1)

    return preferred


def _max_grad_accum_steps(*, total_parameters: int, device_type: str) -> int:
    if device_type == "cuda":
        if total_parameters >= 1_000_000_000:
            return 96
        return 64
    if device_type == "mps":
        return 32
    return 24


def _model_target_total_batch_size(
    *,
    total_parameters: int,
    seq_len: int,
    max_memory_micro_batch_size: int,
    max_grad_accum: int,
    dataset_cap_tokens: int | None,
) -> int:
    # Anchor recommendations to a canonical GPT-2-class pretraining regime:
    # ~524k tokens per optimizer step for a ~124M model. Data and hardware
    # constraints can still pull this down, but the baseline should reflect
    # real pretraining practice rather than a tiny local default.
    model_scale = max(total_parameters, 1) / _REFERENCE_PARAMETER_COUNT
    raw_target = _REFERENCE_TOTAL_BATCH_SIZE * (model_scale ** 0.25)
    lower_bound = seq_len * max(1, min(max_memory_micro_batch_size, 4))
    upper_bound = seq_len * max_memory_micro_batch_size * max_grad_accum
    target = _normalize_total_batch_size_target(
        int(round(raw_target)),
        seq_len=seq_len,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    if dataset_cap_tokens is not None:
        target = _normalize_total_batch_size_target(
            min(target, max(seq_len, dataset_cap_tokens)),
            seq_len=seq_len,
            lower_bound=lower_bound,
            upper_bound=min(upper_bound, max(seq_len, dataset_cap_tokens)),
            prefer_downward=True,
        )
    return max(seq_len, target)


def _pick_candidate(
    *,
    seq_len: int,
    max_memory_micro_batch_size: int,
    target_total_batch_size: int,
    preferred_grad_accum_steps: int,
    max_grad_accum: int,
    dataset_cap_tokens: int | None,
    variant: str,
) -> _BatchCandidate:
    micro_candidates = _micro_batch_candidates(max_memory_micro_batch_size)
    best_candidate: _BatchCandidate | None = None
    best_score = float("inf")

    for micro_batch_size in micro_candidates:
        for grad_accum_steps in range(1, max_grad_accum + 1):
            total_batch_size = seq_len * micro_batch_size * grad_accum_steps
            score = _candidate_score(
                total_batch_size=total_batch_size,
                micro_batch_size=micro_batch_size,
                grad_accum_steps=grad_accum_steps,
                target_total_batch_size=target_total_batch_size,
                preferred_grad_accum_steps=preferred_grad_accum_steps,
                max_memory_micro_batch_size=max_memory_micro_batch_size,
                dataset_cap_tokens=dataset_cap_tokens,
                variant=variant,
            )
            if score < best_score:
                best_score = score
                best_candidate = _BatchCandidate(
                    total_batch_size=total_batch_size,
                    micro_batch_size=micro_batch_size,
                    grad_accum_steps=grad_accum_steps,
                )

    if best_candidate is not None:
        return best_candidate

    return _BatchCandidate(
        total_batch_size=seq_len * max_memory_micro_batch_size,
        micro_batch_size=max_memory_micro_batch_size,
        grad_accum_steps=1,
    )


def _micro_batch_candidates(max_memory_micro_batch_size: int) -> list[int]:
    if max_memory_micro_batch_size <= 128:
        return list(range(1, max_memory_micro_batch_size + 1))

    values = set(range(1, 33))
    values.update(range(max(1, max_memory_micro_batch_size - 32), max_memory_micro_batch_size + 1))
    for ratio in (0.25, 0.333, 0.5, 0.667, 0.75, 0.875, 1.0):
        values.add(max(1, int(round(max_memory_micro_batch_size * ratio))))
    power = 1
    while power <= max_memory_micro_batch_size:
        values.add(power)
        power *= 2
    return sorted(value for value in values if 1 <= value <= max_memory_micro_batch_size)


def _candidate_score(
    *,
    total_batch_size: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    target_total_batch_size: int,
    preferred_grad_accum_steps: int,
    max_memory_micro_batch_size: int,
    dataset_cap_tokens: int | None,
    variant: str,
) -> float:
    target_ratio = total_batch_size / max(target_total_batch_size, 1)
    score = abs(math.log(max(target_ratio, 1e-9))) * 5.0
    score += abs(math.log2(max(grad_accum_steps, 1) / max(preferred_grad_accum_steps, 1))) * 0.85

    memory_utilization = micro_batch_size / max(max_memory_micro_batch_size, 1)
    if variant == "throughput":
        score += (1.0 - memory_utilization) * 2.2
    elif variant == "stability":
        score += memory_utilization * 0.25
    else:
        score += (1.0 - memory_utilization) * 0.8

    if dataset_cap_tokens is not None and total_batch_size > dataset_cap_tokens:
        score += 8.0 * ((total_batch_size / dataset_cap_tokens) - 1.0)

    if _is_power_of_two(seq_len := total_batch_size // max(micro_batch_size * grad_accum_steps, 1)):
        if not _is_power_of_two(total_batch_size):
            score += 2.4
        if not _is_power_of_two(micro_batch_size):
            score += 0.45
        if not _is_power_of_two(grad_accum_steps):
            score += 0.45
    else:
        if not _is_power_of_two(micro_batch_size):
            score += 0.2
        if not _is_power_of_two(grad_accum_steps):
            score += 0.2

    if grad_accum_steps not in {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96}:
        score += 0.15

    return score


def _build_options(
    *,
    training_config: TrainingConfig,
    model_summary: _ModelSummary,
    dataset_summary: _DatasetSummary,
    schedule_summary: _ScheduleSummary,
    current_micro_batch_size: int | None,
    balanced_candidate: _BatchCandidate,
    stability_candidate: _BatchCandidate,
    throughput_candidate: _BatchCandidate,
) -> list[TrainingBatchLrRecommendationOption]:
    option_specs = [
        (
            "balanced",
            "Balanced",
            "Best default for stable optimization and efficient accumulation on the current hardware.",
            "recommended",
            balanced_candidate,
            1.0,
        ),
        (
            "stability",
            "Stability First",
            "Uses a smaller optimizer step to make the run more forgiving when the dataset is small or the schedule is aggressive.",
            "neutral",
            stability_candidate,
            _STABILITY_PROFILE_LR_MULTIPLIER,
        ),
        (
            "throughput",
            "Throughput First",
            "Pushes toward fewer optimizer syncs and more tokens per step when memory headroom allows it.",
            "neutral",
            throughput_candidate,
            _THROUGHPUT_PROFILE_LR_MULTIPLIER,
        ),
    ]

    options: list[TrainingBatchLrRecommendationOption] = []
    for key, label, description, tone, candidate, lr_multiplier in option_specs:
        learning_rate = _recommend_learning_rate(
            total_batch_size=candidate.total_batch_size,
            seq_len=training_config.seq_len,
            model_summary=model_summary,
            dataset_summary=dataset_summary,
            schedule_summary=schedule_summary,
            variant_multiplier=lr_multiplier,
        )
        options.append(
            TrainingBatchLrRecommendationOption(
                key=key,
                label=label,
                description=description,
                tone=tone,
                total_batch_size=candidate.total_batch_size,
                micro_batch_size=candidate.micro_batch_size,
                grad_accum_steps=candidate.grad_accum_steps,
                learning_rate=learning_rate,
                estimated_tokens_per_run=candidate.total_batch_size * training_config.max_steps,
                clear_manual_micro_batch=(
                    current_micro_batch_size is not None
                    and current_micro_batch_size != candidate.micro_batch_size
                ),
            )
        )

    if not options:
        learning_rate = _recommend_learning_rate(
            total_batch_size=balanced_candidate.total_batch_size,
            seq_len=training_config.seq_len,
            model_summary=model_summary,
            dataset_summary=dataset_summary,
            schedule_summary=schedule_summary,
            variant_multiplier=1.0,
        )
        options.append(
            TrainingBatchLrRecommendationOption(
                key="balanced",
                label="Balanced",
                description="Best default for the current training setup.",
                tone="recommended",
                total_batch_size=balanced_candidate.total_batch_size,
                micro_batch_size=balanced_candidate.micro_batch_size,
                grad_accum_steps=balanced_candidate.grad_accum_steps,
                learning_rate=learning_rate,
                estimated_tokens_per_run=balanced_candidate.total_batch_size * training_config.max_steps,
                clear_manual_micro_batch=False,
            )
        )

    _ensure_profile_learning_rate_separation(
        options=options,
        total_parameters=model_summary.total_parameters,
    )
    return options


def _ensure_profile_learning_rate_separation(
    *,
    options: list[TrainingBatchLrRecommendationOption],
    total_parameters: int,
) -> None:
    if not options:
        return

    option_by_key = {option.key: option for option in options}
    balanced = option_by_key.get("balanced")
    if balanced is None:
        return

    min_lr, max_lr = _learning_rate_bounds(total_parameters)
    candidates = _canonical_learning_rate_candidates(lower=min_lr, upper=max_lr)
    if len(candidates) < 2:
        return

    balanced_index = _nearest_canonical_learning_rate_index(
        balanced.learning_rate,
        candidates=candidates,
    )

    stability = option_by_key.get("stability")
    if stability is not None and stability.learning_rate >= balanced.learning_rate and balanced_index > 0:
        stability.learning_rate = candidates[balanced_index - 1]

    throughput = option_by_key.get("throughput")
    if (
        throughput is not None
        and throughput.learning_rate <= balanced.learning_rate
        and balanced_index + 1 < len(candidates)
    ):
        throughput.learning_rate = candidates[balanced_index + 1]


def _recommend_learning_rate(
    *,
    total_batch_size: int,
    seq_len: int,
    model_summary: _ModelSummary,
    dataset_summary: _DatasetSummary,
    schedule_summary: _ScheduleSummary,
    variant_multiplier: float,
) -> float:
    batch_factor = _clamp((total_batch_size / _REFERENCE_TOTAL_BATCH_SIZE) ** 0.08, 0.55, 1.3)
    parameter_factor = _clamp(
        (_REFERENCE_PARAMETER_COUNT / max(model_summary.total_parameters, 1)) ** 0.2,
        0.55,
        1.35,
    )
    context_factor = _context_lr_factor(seq_len)
    dataset_factor = _dataset_lr_factor(dataset_summary)
    schedule_factor = _schedule_lr_factor(schedule_summary)
    architecture_factor = _architecture_lr_factor(model_summary)

    learning_rate = (
        _REFERENCE_LEARNING_RATE
        * batch_factor
        * parameter_factor
        * context_factor
        * dataset_factor
        * schedule_factor
        * architecture_factor
        * variant_multiplier
        / max(schedule_summary.peak_factor, 1.0)
    )

    min_lr, max_lr = _learning_rate_bounds(model_summary.total_parameters)
    clipped = _clamp(learning_rate, min_lr, max_lr)
    snapped = _round_learning_rate_to_canonical_mantissa(clipped, lower=min_lr, upper=max_lr)
    return float(round(snapped, 12))


def _context_lr_factor(seq_len: int) -> float:
    if seq_len >= 8_192:
        return 0.78
    if seq_len >= 4_096:
        return 0.88
    if seq_len >= 2_048:
        return 0.95
    if seq_len >= 1_024:
        return 1.0
    if seq_len <= 128:
        return 1.03
    return 1.0


def _dataset_lr_factor(dataset_summary: _DatasetSummary) -> float:
    if dataset_summary.dataset_scale == "tiny_local":
        return 0.76
    if dataset_summary.dataset_scale == "small_local":
        return 0.84
    if dataset_summary.dataset_scale == "medium_local":
        return 0.92
    if dataset_summary.dataset_scale == "large_local":
        return 0.97
    if dataset_summary.dataset_scale == "mixed":
        return 0.98
    return 1.0


def _schedule_lr_factor(schedule_summary: _ScheduleSummary) -> float:
    if schedule_summary.warmup_fraction == 0:
        return 0.87
    if schedule_summary.warmup_fraction < 0.03:
        return 0.91
    if schedule_summary.warmup_fraction < 0.08:
        return 0.96
    if schedule_summary.warmup_fraction > 0.2:
        return 1.02
    return 1.0


def _architecture_lr_factor(model_summary: _ModelSummary) -> float:
    factor = 1.0
    activation_types = set(model_summary.activation_types)
    if activation_types.intersection({"relu", "squared_relu"}):
        factor *= 0.93
    if activation_types.intersection({"tanh", "sigmoid"}):
        factor *= 0.9
    if model_summary.max_mlp_multiplier >= 8:
        factor *= 0.9
    elif model_summary.max_mlp_multiplier >= 6:
        factor *= 0.95
    if not model_summary.weight_tying:
        factor *= 0.97
    if model_summary.attention_component_count >= 32 and model_summary.total_parameters >= 300_000_000:
        factor *= 0.95
    return factor


def _learning_rate_bounds(total_parameters: int) -> tuple[float, float]:
    if total_parameters < 30_000_000:
        return 6e-5, 9e-4
    if total_parameters < 300_000_000:
        return 4e-5, 6e-4
    if total_parameters < 1_000_000_000:
        return 2.5e-5, 4.5e-4
    return 1.5e-5, 3e-4


def _build_summary(
    *,
    recommended_option: TrainingBatchLrRecommendationOption,
    model_summary: _ModelSummary,
    dataset_summary: _DatasetSummary,
    schedule_summary: _ScheduleSummary,
    memory_estimate: StepSizeEstimate,
    training_config: TrainingConfig,
) -> str:
    model_scale = _format_compact_count(model_summary.total_parameters)
    data_note = "streaming-scale data assumptions" if dataset_summary.dataset_scale == "streaming" else dataset_summary.dataset_scale.replace("_", " ")
    return (
        f"This uses micro batch {recommended_option.micro_batch_size:,} with "
        f"{recommended_option.grad_accum_steps:,} accumulation step"
        f"{'' if recommended_option.grad_accum_steps == 1 else 's'} on {memory_estimate.device.type}, "
        f"targets a {model_scale}-parameter model, respects the current {schedule_summary.label.lower()}, "
        f"and keeps the recommendation grounded in {data_note}."
    )


def _build_factors(
    *,
    training_config: TrainingConfig,
    current_batch_plan: BatchRuntimePlan | None,
    recommended_option: TrainingBatchLrRecommendationOption,
    model_summary: _ModelSummary,
    dataset_summary: _DatasetSummary,
    schedule_summary: _ScheduleSummary,
    memory_estimate: StepSizeEstimate,
) -> list[TrainingBatchLrRecommendationFactor]:
    factors = [
        TrainingBatchLrRecommendationFactor(
            code="memory_fit",
            label="Memory headroom and accumulation",
            detail=(
                f"The live {memory_estimate.device.type} estimate fits up to {memory_estimate.max_batch_size:,} "
                f"sequences per micro-step at seq_len {training_config.seq_len:,}. "
                f"The recommended plan turns that into total batch {recommended_option.total_batch_size:,} tokens "
                f"with {recommended_option.grad_accum_steps:,} accumulation step"
                f"{'' if recommended_option.grad_accum_steps == 1 else 's'}."
            ),
            tone="good",
        ),
        TrainingBatchLrRecommendationFactor(
            code="model_scale",
            label="Model scale and architecture",
            detail=(
                f"The current model is {_format_compact_count(model_summary.total_parameters)} parameters across "
                f"{model_summary.block_count:,} block"
                f"{'' if model_summary.block_count == 1 else 's'}, with max MLP multiplier "
                f"{_format_decimal(model_summary.max_mlp_multiplier)} and "
                f"{model_summary.attention_component_count:,} attention module"
                f"{'' if model_summary.attention_component_count == 1 else 's'}. "
                f"That profile supports a pretraining-scale token batch when the current data and hardware allow it."
            ),
            tone="neutral",
        ),
        TrainingBatchLrRecommendationFactor(
            code="schedule_shape",
            label="Current LR schedule",
            detail=(
                f"The schedule behaves like {schedule_summary.label.lower()} and peaks at "
                f"{_format_decimal(schedule_summary.peak_factor)}x the configured base LR, with "
                f"{_format_percentage(schedule_summary.warmup_fraction)} warmup. "
                f"The recommendation keeps the peak LR inside a conservative range for this trainer."
            ),
            tone="neutral" if schedule_summary.warmup_fraction > 0 else "warning",
        ),
    ]

    dataset_detail = _dataset_factor_detail(dataset_summary, training_config.max_steps)
    if dataset_detail is not None:
        factors.append(dataset_detail)

    reference_target = _round_to_nearest_multiple(
        int(
            round(
                _REFERENCE_TOTAL_BATCH_SIZE
                * ((max(model_summary.total_parameters, 1) / _REFERENCE_PARAMETER_COUNT) ** 0.25)
            )
        ),
        training_config.seq_len,
    )
    hardware_cap_tokens = (
        training_config.seq_len
        * memory_estimate.max_batch_size
        * _max_grad_accum_steps(
            total_parameters=model_summary.total_parameters,
            device_type=memory_estimate.device.type,
        )
    )
    constraint_reasons: list[str] = []
    if dataset_summary.step_budget_cap_tokens is not None and dataset_summary.step_budget_cap_tokens < reference_target:
        constraint_reasons.append("the current local-corpus token budget")
    if hardware_cap_tokens < reference_target:
        constraint_reasons.append("the current memory and accumulation ceiling")
    if recommended_option.total_batch_size < reference_target * 0.75:
        if constraint_reasons:
            reason_text = " and ".join(constraint_reasons)
        else:
            reason_text = "the current balance between accumulation depth and throughput"
        factors.append(
            TrainingBatchLrRecommendationFactor(
                code="pretraining_anchor",
                label="Reference pretraining target",
                detail=(
                    f"For a model of this scale, the unconstrained target is about {reference_target:,} tokens per optimizer step. "
                    f"The current recommendation lands lower because of {reason_text}."
                ),
                tone="warning",
            )
        )

    if current_batch_plan is not None:
        if (
            current_batch_plan.micro_batch_size != recommended_option.micro_batch_size
            or training_config.total_batch_size != recommended_option.total_batch_size
        ):
            factors.append(
                TrainingBatchLrRecommendationFactor(
                    code="current_batch_gap",
                    label="Current batch settings",
                    detail=(
                        f"The current setup resolves to micro batch {current_batch_plan.micro_batch_size:,} and "
                        f"{current_batch_plan.grad_accum_steps:,} accumulation step"
                        f"{'' if current_batch_plan.grad_accum_steps == 1 else 's'}. "
                        f"The recommended plan shifts that to {recommended_option.micro_batch_size:,} and "
                        f"{recommended_option.grad_accum_steps:,} to better match the model and device."
                    ),
                    tone="warning",
                )
            )

    current_lr = training_config.optimizer.lr
    if abs(current_lr - recommended_option.learning_rate) / max(current_lr, 1e-9) >= 0.2:
        factors.append(
            TrainingBatchLrRecommendationFactor(
                code="current_lr_gap",
                label="Current learning rate",
                detail=(
                    f"The current base LR is {_format_learning_rate(current_lr)}. "
                    f"The recommended target is {_format_learning_rate(recommended_option.learning_rate)} "
                    f"for the current batch size, schedule, and model scale."
                ),
                tone="warning",
            )
        )

    return factors


def _dataset_factor_detail(
    dataset_summary: _DatasetSummary,
    max_steps: int,
) -> TrainingBatchLrRecommendationFactor | None:
    if dataset_summary.dataset_scale == "streaming":
        return TrainingBatchLrRecommendationFactor(
            code="dataset_scale",
            label="Dataset scale and mixing",
            detail=(
                f"The run pulls from {dataset_summary.dataset_count:,} streaming dataset"
                f"{'' if dataset_summary.dataset_count == 1 else 's'} with a dominant weight of "
                f"{_format_percentage(dataset_summary.dominant_dataset_weight)}. "
                f"Because corpus size is effectively open-ended here, the recommendation leans on model scale, memory, and scheduler shape."
            ),
            tone="neutral",
        )

    if dataset_summary.dataset_scale == "mixed":
        return TrainingBatchLrRecommendationFactor(
            code="dataset_scale",
            label="Mixed local and streaming data",
            detail=(
                f"The dataloader mixes {dataset_summary.local_dataset_count:,} local dataset"
                f"{'' if dataset_summary.local_dataset_count == 1 else 's'} with "
                f"{dataset_summary.streaming_dataset_count:,} streaming source"
                f"{'' if dataset_summary.streaming_dataset_count == 1 else 's'}. "
                f"The recommendation avoids an oversized batch so the local data is not washed out in each optimizer step."
            ),
            tone="neutral",
        )

    if dataset_summary.local_total_size_bytes is None or dataset_summary.approx_local_tokens is None:
        return None

    epochs_over_run = (
        max_steps * max(dataset_summary.step_budget_cap_tokens or 0, 1)
        / max(dataset_summary.approx_local_tokens, 1)
    )
    return TrainingBatchLrRecommendationFactor(
        code="dataset_scale",
        label="Local corpus size",
        detail=(
            f"The local corpus resolves to {dataset_summary.local_file_count:,} file"
            f"{'' if dataset_summary.local_file_count == 1 else 's'} and roughly "
            f"{_format_bytes(dataset_summary.local_total_size_bytes)}. "
            f"That is treated as {dataset_summary.dataset_scale.replace('_', ' ')}, so the batch target is capped to avoid consuming too much of the corpus in a single update. "
            f"At the cap, a {max_steps:,}-step run would still cover about {_format_decimal(epochs_over_run, digits=1)} estimated passes over the local data."
        ),
        tone="warning" if dataset_summary.dataset_scale in {"tiny_local", "small_local"} else "neutral",
    )


def _recommendation_confidence(*, device_type: str, dataset_scale: str, schedule_peak_factor: float) -> str:
    if device_type == "cpu":
        return "low"
    if dataset_scale in {"tiny_local", "small_local"} or schedule_peak_factor > 1.25:
        return "medium"
    return "high"


def _normalize_total_batch_size_target(
    value: int,
    *,
    seq_len: int,
    lower_bound: int,
    upper_bound: int,
    prefer_downward: bool = False,
) -> int:
    clamped = max(lower_bound, min(value, upper_bound))
    if _is_power_of_two(seq_len):
        power_target = _nearest_power_of_two_in_range(
            clamped,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            prefer_downward=prefer_downward,
        )
        if power_target is not None:
            return power_target
    return (
        _round_down_to_multiple(clamped, seq_len)
        if prefer_downward
        else _round_to_nearest_multiple(clamped, seq_len)
    )


def _nearest_power_of_two_in_range(
    value: int,
    *,
    lower_bound: int,
    upper_bound: int,
    prefer_downward: bool = False,
) -> int | None:
    if upper_bound < 1 or lower_bound > upper_bound:
        return None

    candidates: list[int] = []
    power = 1
    while power < lower_bound:
        power <<= 1
    while power <= upper_bound:
        candidates.append(power)
        power <<= 1

    if not candidates:
        return None
    if prefer_downward:
        not_above = [candidate for candidate in candidates if candidate <= value]
        if not_above:
            return max(not_above)
    return min(
        candidates,
        key=lambda candidate: (abs(math.log(candidate / max(value, 1))), abs(candidate - value)),
    )


def _round_to_nearest_multiple(value: int, base: int) -> int:
    if base <= 0:
        return value
    lower = _round_down_to_multiple(value, base)
    upper = lower if lower == value else lower + base
    if abs(value - lower) <= abs(upper - value):
        return lower
    return upper


def _round_down_to_multiple(value: int, base: int) -> int:
    if base <= 0:
        return value
    return max(base, (value // base) * base)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _round_learning_rate_to_canonical_mantissa(value: float, *, lower: float, upper: float) -> float:
    if value <= 0 or not math.isfinite(value):
        return value

    candidates = _canonical_learning_rate_candidates(lower=lower, upper=upper)
    if not candidates:
        return value

    return candidates[_nearest_canonical_learning_rate_index(value, candidates=candidates)]


def _canonical_learning_rate_candidates(*, lower: float, upper: float) -> list[float]:
    lower_bound = max(lower, 1e-12)
    upper_bound = max(upper, lower_bound)
    min_exponent = int(math.floor(math.log10(lower_bound))) - 1
    max_exponent = int(math.ceil(math.log10(upper_bound))) + 1
    candidates = {
        mantissa * (10.0 ** exponent)
        for exponent in range(min_exponent, max_exponent + 1)
        for mantissa in _CANONICAL_LR_MANTISSAS
        if lower_bound <= mantissa * (10.0 ** exponent) <= upper_bound
    }
    return sorted(candidates)


def _nearest_canonical_learning_rate_index(value: float, *, candidates: list[float]) -> int:
    return min(
        range(len(candidates)),
        key=lambda index: (abs(math.log(candidates[index] / value)), -candidates[index]),
    )


def _format_compact_count(value: int) -> str:
    absolute = abs(int(value))
    if absolute >= 1_000_000_000:
        scaled = f"{value / 1_000_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{scaled}B"
    if absolute >= 1_000_000:
        scaled = f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{scaled}M"
    if absolute >= 1_000:
        scaled = f"{value / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{scaled}K"
    return f"{value:,}"


def _format_decimal(value: float, *, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "n/a"
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _format_percentage(value: float) -> str:
    percentage = max(0.0, value) * 100.0
    if percentage == 0 or percentage >= 100:
        return f"{round(percentage)}%"
    if percentage < 10:
        return f"{percentage:.1f}%"
    return f"{round(percentage)}%"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    absolute = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if absolute < 1024.0 or unit == "TiB":
            return f"{absolute:.1f} {unit}" if unit != "B" else f"{int(absolute)} {unit}"
        absolute /= 1024.0
    return f"{value} B"


def _format_learning_rate(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.3e}"
