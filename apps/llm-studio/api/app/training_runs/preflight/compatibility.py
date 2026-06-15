from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenizers import Tokenizer

from ..schemas import TrainingCompatibilitySummary, TrainingFixSuggestion, TrainingIssue
from .config_validation import issue
from ...runtime_paths import ensure_source_root_on_path

IMPORT_ROOT = ensure_source_root_on_path()

from training.dataloader_config import TrainingDataloaderConfig
from training.training_config import TrainingConfig


@dataclass(slots=True)
class CompatibilityCheckResult:
    compatibility: TrainingCompatibilitySummary
    errors: list[TrainingIssue]
    fixes: list[TrainingFixSuggestion]


def collect_missing_special_tokens(tokenizer: Tokenizer, config: TrainingDataloaderConfig) -> list[str]:
    required_tokens: list[str] = []
    if config.add_bos and config.bos_token is not None:
        required_tokens.append(config.bos_token)
    if config.add_eos and config.eos_token is not None:
        required_tokens.append(config.eos_token)
    if not config.drop_last:
        if config.pad_token is not None:
            required_tokens.append(config.pad_token)
        elif config.eos_token is not None:
            required_tokens.append(config.eos_token)

    missing = []
    for token in required_tokens:
        if tokenizer.token_to_id(token) is None:
            missing.append(token)
    return missing


def check_model_tokenizer_compatibility(
    *,
    model_config: dict[str, Any],
    tokenizer_vocab_size: int,
    training_config: TrainingConfig,
    dataloader_config: TrainingDataloaderConfig,
    missing_special_tokens: list[str],
) -> CompatibilityCheckResult:
    compatibility = TrainingCompatibilitySummary(
        model_context_length=int(model_config["context_length"]),
        model_vocab_size=int(model_config["vocab_size"]),
        tokenizer_vocab_size=int(tokenizer_vocab_size),
        seq_len=int(training_config.seq_len),
        scheduler_total_steps=sum(item.steps for item in training_config.lr_scheduler.schedulers),
        max_steps=int(training_config.max_steps),
        missing_special_tokens=missing_special_tokens,
    )
    errors: list[TrainingIssue] = []
    fixes: list[TrainingFixSuggestion] = []

    if int(model_config["vocab_size"]) != int(tokenizer_vocab_size):
        errors.append(
            issue(
                "vocab_size_mismatch",
                f"Model vocab_size ({model_config['vocab_size']}) must match tokenizer vocab_size ({tokenizer_vocab_size}).",
                "$.model_config.vocab_size",
            )
        )

    if training_config.seq_len > int(model_config["context_length"]):
        errors.append(
            issue(
                "seq_len_exceeds_context_length",
                f"seq_len ({training_config.seq_len}) exceeds model context_length ({model_config['context_length']}).",
                "$.training_config.seq_len",
            )
        )
        fixes.append(
            TrainingFixSuggestion(
                code="set_seq_len_to_context_length",
                label="Set seq_len to model context_length",
                description="Clamp sequence length to the selected model's context window.",
                path="training_config.seq_len",
                value=int(model_config["context_length"]),
            )
        )

    for missing_token in missing_special_tokens:
        errors.append(
            issue(
                "missing_special_token",
                f"Tokenizer is missing required special token {missing_token}.",
                "$.dataloader_config",
            )
        )

    return CompatibilityCheckResult(compatibility=compatibility, errors=errors, fixes=fixes)
