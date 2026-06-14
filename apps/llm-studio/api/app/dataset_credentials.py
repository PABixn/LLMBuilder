from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

HF_DATASET_TOKENS_ENV = "LLM_STUDIO_HF_DATASET_TOKENS"


def split_dataset_hf_tokens(
    config: dict[str, Any],
    *,
    fallback_token: str | None = None,
) -> tuple[dict[str, Any], list[str | None]]:
    sanitized = deepcopy(config)
    datasets = sanitized.get("datasets")
    if not isinstance(datasets, list):
        return sanitized, []

    fallback = _normalize_token(fallback_token)
    tokens: list[str | None] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            tokens.append(fallback)
            continue
        embedded = _normalize_token(dataset.pop("hf_token", None))
        tokens.append(embedded or fallback)
    return sanitized, tokens


def inject_dataset_hf_tokens(
    config: dict[str, Any],
    tokens: list[str | None],
) -> dict[str, Any]:
    hydrated = deepcopy(config)
    datasets = hydrated.get("datasets")
    if not isinstance(datasets, list):
        return hydrated
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict) or index >= len(tokens):
            continue
        token = _normalize_token(tokens[index])
        if token is not None:
            dataset["hf_token"] = token
    return hydrated


def strip_hf_tokens(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): strip_hf_tokens(item)
            for key, item in value.items()
            if _normalized_key(str(key)) != "hftoken"
        }
    if isinstance(value, list):
        return [strip_hf_tokens(item) for item in value]
    if isinstance(value, tuple):
        return [strip_hf_tokens(item) for item in value]
    return value


def encode_dataset_hf_tokens(tokens: list[str | None]) -> str | None:
    normalized = [_normalize_token(token) for token in tokens]
    if not any(normalized):
        return None
    return json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))


def _normalize_token(value: object | None) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())
