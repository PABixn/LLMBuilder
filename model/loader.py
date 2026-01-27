import json
from pathlib import Path
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Attention(StrictModel):
    n_head: Annotated[int, Field(gt=0)]
    n_kv_head: Annotated[int, Field(gt=0)]


class Linear(StrictModel):
    bias: bool


class Activation(StrictModel):
    type: Literal["gelu", "relu", "squared_relu", "silu", "tanh", "sigmoid"]


class RMSNorm(StrictModel):
    type: Literal["rmsnorm"]
    learnable_gamma: bool


class LayerNorm(StrictModel):
    type: Literal["layernorm"]


Norm = Union[RMSNorm, LayerNorm]


class AttentionComponent(StrictModel):
    attention: Attention


class MLP(StrictModel):
    multiplier: Annotated[float, Field(gt=0)] = 4
    sequence: List["MLPStep"]


class MLPComponent(StrictModel):
    mlp: MLP


class NormComponent(StrictModel):
    norm: Norm


class ActivationComponent(StrictModel):
    activation: Activation


class LinearStep(StrictModel):
    linear: Linear


MLPStep = Union[LinearStep, ActivationComponent, NormComponent]
Component = Union[AttentionComponent, MLPComponent, NormComponent, ActivationComponent]


class Block(StrictModel):
    components: List[Component]


class LLMConfig(StrictModel):
    context_length: Annotated[int, Field(gt=0)]
    vocab_size: Annotated[int, Field(gt=0)]
    n_embd: Annotated[int, Field(gt=0)]
    weight_tying: bool = True
    blocks: List[Block]


def load_config(config_path: str | Path) -> LLMConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        return LLMConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
