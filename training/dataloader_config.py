import json
from pathlib import Path
from typing import Annotated, Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NormalizationConfig(StrictModel):
    normalize_newlines: bool = True
    collapse_whitespace: bool = False
    strip: bool = True
    lowercase: bool = False
    drop_empty: bool = True


class ShuffleConfig(StrictModel):
    buffer_size: Annotated[int, Field(gt=0)] = 1000
    seed: Optional[int] = None


class DatasetSpec(StrictModel):
    name: Annotated[str, Field(min_length=1)]
    config: Optional[str] = None
    split: Annotated[str, Field(min_length=1)] = "train"
    streaming: bool = True
    text_columns: Annotated[List[str], Field(min_length=1)] = Field(
        default_factory=lambda: ["text"]
    )
    text_joiner: str = "\n"
    weight: Annotated[float, Field(gt=0)] = 1.0
    filters: Optional[List[List[Any]]] = None
    columns: Optional[List[str]] = None
    data_files: Optional[Any] = None
    normalization: Optional[NormalizationConfig] = None
    record_separator: Optional[str] = None
    shuffle: Optional[ShuffleConfig | Literal[False]] = None

    @model_validator(mode="after")
    def validate_filters(self) -> "DatasetSpec":
        if self.filters is None:
            return self
        for filt in self.filters:
            if not isinstance(filt, list) or len(filt) != 3:
                raise ValueError(
                    "filters must be a list of [column, op, value] entries"
                )
        return self


class MixingConfig(StrictModel):
    seed: Optional[int] = None
    exhausted_policy: Literal["stop", "repeat", "drop"] = "stop"


class TrainingDataloaderConfig(StrictModel):
    datasets: Annotated[List[DatasetSpec], Field(min_length=1)]
    add_bos: bool = False
    add_eos: bool = False
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    pad_token: Optional[str] = None
    drop_last: bool = True
    token_dtype: Literal["int64", "int32", "int16", "uint8"] = "int64"
    pretokenize_batch_size: Annotated[int, Field(gt=0)] = 1000
    cache_dir: Optional[str] = None
    mixing: MixingConfig = MixingConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    record_separator: str = ""
    shuffle: Optional[ShuffleConfig | Literal[False]] = None
    node_split: bool = False
    node_rank: Optional[int] = None
    node_world_size: Optional[int] = None

    @model_validator(mode="after")
    def validate_special_tokens(self) -> "TrainingDataloaderConfig":
        if self.add_bos and not self.bos_token:
            raise ValueError("bos_token is required when add_bos is true")
        if self.add_eos and not self.eos_token:
            raise ValueError("eos_token is required when add_eos is true")
        if not self.drop_last and not (self.pad_token or self.eos_token):
            raise ValueError(
                "pad_token (or eos_token) is required when drop_last is false"
            )
        if self.node_rank is None and self.node_world_size is None:
            return self
        if self.node_rank is None or self.node_world_size is None:
            raise ValueError("node_rank and node_world_size must be provided together")
        if self.node_world_size <= 0:
            raise ValueError("node_world_size must be > 0")
        if self.node_rank < 0 or self.node_rank >= self.node_world_size:
            raise ValueError("node_rank must be in [0, node_world_size)")
        return self


def load_training_dataloader_config(
    config_path: str | Path,
) -> TrainingDataloaderConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        return TrainingDataloaderConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
