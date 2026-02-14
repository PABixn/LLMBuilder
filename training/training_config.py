import json
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


def load_training_config(config_path: str | Path) -> TrainingConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        return TrainingConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
