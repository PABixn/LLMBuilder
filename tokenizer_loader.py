import json
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokenizerConfig(StrictModel):
    name: Annotated[str, Field(min_length=1)]
    tokenizer_type: Literal["bpe", "wordpiece", "unigram"]
    vocab_size: Annotated[int, Field(gt=0)]
    min_frequency: Annotated[int, Field(ge=1)]
    special_tokens: Annotated[List[str], Field(min_length=1)]
    pre_tokenizer: Literal["byte_level", "whitespace", "metaspace"]
    decoder: Literal["byte_level", "wordpiece", "metaspace"]
    byte_fallback: Optional[bool] = None
    unk_token: Optional[str] = None

    @model_validator(mode="after")
    def enforce_type_specific_fields(self) -> "TokenizerConfig":
        if self.tokenizer_type == "bpe":
            if self.byte_fallback is None:
                raise ValueError("byte_fallback is required for bpe tokenizer_type")
            if self.byte_fallback is False and self.unk_token is None:
                raise ValueError(
                    "unk_token is required when tokenizer_type is bpe and byte_fallback is false"
                )
            if self.byte_fallback is True and self.unk_token is not None:
                raise ValueError(
                    "unk_token is only valid for wordpiece or bpe with byte_fallback false"
                )
        elif self.tokenizer_type == "wordpiece":
            if self.unk_token is None:
                raise ValueError("unk_token is required for wordpiece tokenizer_type")
            if self.byte_fallback is not None:
                raise ValueError("byte_fallback is only valid for bpe tokenizer_type")
        else:
            if self.byte_fallback is not None:
                raise ValueError("byte_fallback is only valid for bpe tokenizer_type")
            if self.unk_token is not None:
                raise ValueError(
                    "unk_token is only valid for wordpiece or bpe with byte_fallback false"
                )
        return self


def load_tokenizer_config(config_path: str | Path) -> TokenizerConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        return TokenizerConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
