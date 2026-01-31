from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ConstantLR,
    ExponentialLR,
    LinearLR,
    MultiStepLR,
    SequentialLR,
    StepLR,
)

from training.training_config import (
    CosineAnnealingLRSchedulerConfig,
    CosineAnnealingWarmRestartsLRSchedulerConfig,
    ConstantLRSchedulerConfig,
    ExponentialLRSchedulerConfig,
    LinearLRSchedulerConfig,
    MultiStepLRSchedulerConfig,
    SchedulerConfig,
    SequentialLRSchedulerConfig,
    StepLRSchedulerConfig,
)


def build_lr_scheduler(
    optimizer: Optimizer, config: SequentialLRSchedulerConfig
):
    schedulers: List[object] = []
    milestones: List[int] = []
    completed_steps = 0

    for idx, scheduler_config in enumerate(config.schedulers):
        schedulers.append(_build_single_scheduler(optimizer, scheduler_config))
        completed_steps += scheduler_config.steps
        if idx < len(config.schedulers) - 1:
            milestones.append(completed_steps)

    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def _build_single_scheduler(
    optimizer: Optimizer, config: SchedulerConfig
):
    if isinstance(config, LinearLRSchedulerConfig):
        return LinearLR(
            optimizer,
            start_factor=config.start_factor,
            end_factor=config.end_factor,
            total_iters=config.steps,
        )
    if isinstance(config, CosineAnnealingLRSchedulerConfig):
        return CosineAnnealingLR(
            optimizer,
            T_max=config.steps,
            eta_min=config.eta_min,
        )
    if isinstance(config, StepLRSchedulerConfig):
        return StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    if isinstance(config, MultiStepLRSchedulerConfig):
        return MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
        )
    if isinstance(config, ExponentialLRSchedulerConfig):
        return ExponentialLR(
            optimizer,
            gamma=config.gamma,
        )
    if isinstance(config, ConstantLRSchedulerConfig):
        return ConstantLR(
            optimizer,
            factor=config.factor,
            total_iters=config.steps,
        )
    if isinstance(config, CosineAnnealingWarmRestartsLRSchedulerConfig):
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.t_0,
            T_mult=config.t_mult,
            eta_min=config.eta_min,
        )
    raise ValueError(f"Unsupported scheduler config type: {type(config)!r}")
