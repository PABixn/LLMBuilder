import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OptimizerConfig(StrictModel):
    lr: Annotated[float, Field(gt=0)]
    weight_decay: Annotated[float, Field(ge=0)]
    betas: Annotated[List[float], Field(min_length=2, max_length=2)]
    eps: Annotated[float, Field(gt=0)]

    @model_validator(mode="after")
    def validate_betas(self) -> "OptimizerConfig":
        if len(self.betas) != 2:
            raise ValueError("betas must have length 2")
        for beta in self.betas:
            if beta <= 0 or beta >= 1:
                raise ValueError("betas values must be in (0, 1)")
        return self


class LinearLRSchedulerConfig(StrictModel):
    type: Literal["linear"]
    steps: Annotated[int, Field(gt=0)]
    start_factor: Annotated[float, Field(gt=0)]
    end_factor: Annotated[float, Field(gt=0)]


class CosineAnnealingLRSchedulerConfig(StrictModel):
    type: Literal["cosine_annealing"]
    steps: Annotated[int, Field(gt=0)]
    eta_min: Annotated[float, Field(ge=0)] = 0.0


class StepLRSchedulerConfig(StrictModel):
    type: Literal["step"]
    steps: Annotated[int, Field(gt=0)]
    step_size: Annotated[int, Field(gt=0)]
    gamma: Annotated[float, Field(gt=0)] = 0.1

    @model_validator(mode="after")
    def validate_step_size(self) -> "StepLRSchedulerConfig":
        if self.step_size > self.steps:
            raise ValueError("step_size must be <= steps")
        return self


class MultiStepLRSchedulerConfig(StrictModel):
    type: Literal["multistep"]
    steps: Annotated[int, Field(gt=0)]
    milestones: Annotated[List[int], Field(min_length=1)]
    gamma: Annotated[float, Field(gt=0)] = 0.1

    @model_validator(mode="after")
    def validate_milestones(self) -> "MultiStepLRSchedulerConfig":
        if any(milestone <= 0 or milestone >= self.steps for milestone in self.milestones):
            raise ValueError("milestones must be in the range (0, steps)")
        if sorted(self.milestones) != self.milestones:
            raise ValueError("milestones must be sorted ascending")
        if len(set(self.milestones)) != len(self.milestones):
            raise ValueError("milestones must be unique")
        return self


class ExponentialLRSchedulerConfig(StrictModel):
    type: Literal["exponential"]
    steps: Annotated[int, Field(gt=0)]
    gamma: Annotated[float, Field(gt=0)]


class ConstantLRSchedulerConfig(StrictModel):
    type: Literal["constant"]
    steps: Annotated[int, Field(gt=0)]
    factor: Annotated[float, Field(gt=0)]


class CosineAnnealingWarmRestartsLRSchedulerConfig(StrictModel):
    type: Literal["cosine_annealing_warm_restarts"]
    steps: Annotated[int, Field(gt=0)]
    t_0: Annotated[int, Field(gt=0)]
    t_mult: Annotated[int, Field(ge=1)] = 1
    eta_min: Annotated[float, Field(ge=0)] = 0.0


SchedulerConfig = Annotated[
    Union[
        LinearLRSchedulerConfig,
        CosineAnnealingLRSchedulerConfig,
        StepLRSchedulerConfig,
        MultiStepLRSchedulerConfig,
        ExponentialLRSchedulerConfig,
        ConstantLRSchedulerConfig,
        CosineAnnealingWarmRestartsLRSchedulerConfig,
    ],
    Field(discriminator="type"),
]


class SequentialLRSchedulerConfig(StrictModel):
    type: Literal["sequential"]
    schedulers: Annotated[List[SchedulerConfig], Field(min_length=1)]


class SamplerPromptConfig(StrictModel):
    prompt: Annotated[str, Field(min_length=1)]
    max_tokens: Annotated[int, Field(gt=0)]
    temperature: Annotated[float, Field(ge=0)]
    top_k: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_prompt(self) -> "SamplerPromptConfig":
        if not self.prompt.strip():
            raise ValueError("prompt must contain at least one non-whitespace character")
        return self


class SamplerConfig(StrictModel):
    prompts: Annotated[List[SamplerPromptConfig], Field(min_length=1)]


class TrainingConfig(StrictModel):
    max_steps: Annotated[int, Field(gt=0)]
    total_batch_size: Annotated[int, Field(gt=0)]
    micro_batch_size: Annotated[int, Field(gt=0)] | None = None
    seq_len: Annotated[int, Field(gt=0)]
    sample_every: Annotated[int, Field(gt=0)]
    sampler: SamplerConfig
    save_every: Annotated[int, Field(gt=0)]
    optimizer: OptimizerConfig
    lr_scheduler: SequentialLRSchedulerConfig

    @model_validator(mode="after")
    def validate_scheduler_steps(self) -> "TrainingConfig":
        total_steps = sum(scheduler.steps for scheduler in self.lr_scheduler.schedulers)
        if total_steps != self.max_steps:
            raise ValueError(
                "sum of lr_scheduler steps must equal max_steps"
            )
        return self


@dataclass(frozen=True)
class BatchRuntimePlan:
    micro_batch_size: int
    tokens_per_micro_step: int
    tokens_per_world_step: int
    grad_accum_steps: int
    max_batch_size_from_total: int
    max_batch_size_from_memory: int
    max_allowed_batch_size: int


def derive_batch_runtime_plan(
    *,
    total_batch_size: int,
    seq_len: int,
    max_memory_batch_size: int,
    world_size: int = 1,
    requested_micro_batch_size: int | None = None,
) -> BatchRuntimePlan:
    if total_batch_size <= 0:
        raise ValueError("total_batch_size must be > 0")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if max_memory_batch_size <= 0:
        raise ValueError("memory estimate did not produce a positive micro-batch size")

    world_token_size = seq_len * world_size
    if total_batch_size % world_token_size != 0:
        raise ValueError("total_batch_size must be divisible by seq_len * world_size.")

    max_batch_size_from_total = total_batch_size // world_token_size
    max_allowed_batch_size = min(max_memory_batch_size, max_batch_size_from_total)
    if max_allowed_batch_size <= 0:
        raise ValueError(
            "Could not derive a positive micro-batch size from memory estimate and total_batch_size/seq_len."
        )

    if requested_micro_batch_size is not None:
        if requested_micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be > 0")
        if requested_micro_batch_size > max_memory_batch_size:
            raise ValueError("micro_batch_size exceeds the memory-estimated maximum.")
        if requested_micro_batch_size > max_batch_size_from_total:
            raise ValueError("micro_batch_size exceeds total_batch_size / (seq_len * world_size).")
        if max_batch_size_from_total % requested_micro_batch_size != 0:
            raise ValueError(
                "total_batch_size must be divisible by micro_batch_size * seq_len * world_size."
            )
        micro_batch_size = requested_micro_batch_size
    else:
        micro_batch_size = max(
            candidate
            for candidate in range(1, max_allowed_batch_size + 1)
            if max_batch_size_from_total % candidate == 0
        )

    tokens_per_micro_step = micro_batch_size * seq_len
    tokens_per_world_step = tokens_per_micro_step * world_size
    grad_accum_steps = total_batch_size // tokens_per_world_step
    if grad_accum_steps <= 0:
        raise ValueError("Derived grad_accum_steps must be positive.")

    return BatchRuntimePlan(
        micro_batch_size=micro_batch_size,
        tokens_per_micro_step=tokens_per_micro_step,
        tokens_per_world_step=tokens_per_world_step,
        grad_accum_steps=grad_accum_steps,
        max_batch_size_from_total=max_batch_size_from_total,
        max_batch_size_from_memory=max_memory_batch_size,
        max_allowed_batch_size=max_allowed_batch_size,
    )


def load_training_config(config_path: str | Path) -> TrainingConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        return TrainingConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
