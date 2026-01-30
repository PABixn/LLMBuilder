import json
from pathlib import Path
from typing import Annotated, List

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


class TrainingConfig(StrictModel):
    max_steps: Annotated[int, Field(gt=0)]
    total_batch_size: Annotated[int, Field(gt=0)]
    seq_len: Annotated[int, Field(gt=0)]
    sample_every: Annotated[int, Field(gt=0)]
    sample_max_tokens: Annotated[int, Field(gt=0)]
    save_every: Annotated[int, Field(gt=0)]
    optimizer: OptimizerConfig


def load_training_config(config_path: str | Path) -> TrainingConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        return TrainingConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
