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
    hf_token: Optional[str] = None
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


class TextBudgetConfig(StrictModel):
    limit: Annotated[int, Field(gt=0)]
    unit: Literal["chars", "bytes"] = "chars"
    behavior: Literal["stop", "truncate"] = "stop"


class ChunkingConfig(StrictModel):
    chunk_size: Annotated[int, Field(gt=0)]
    drop_last: bool = False


class MixingConfig(StrictModel):
    seed: Optional[int] = None
    exhausted_policy: Literal["stop", "repeat", "drop"] = "stop"


class DataloaderConfig(StrictModel):
    datasets: Annotated[List[DatasetSpec], Field(min_length=1)]
    budget: TextBudgetConfig
    chunking: Optional[ChunkingConfig] = None
    mixing: MixingConfig = MixingConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    record_separator: str = ""
    shuffle: Optional[ShuffleConfig | Literal[False]] = None


def load_tokenizer_dataloader_config(config_path: str | Path) -> DataloaderConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        return DataloaderConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
