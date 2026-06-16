from __future__ import annotations

from pathlib import Path

import torch
from pydantic import ValidationError

from app.training_runs.preflight.assets import tokenizer_display_name
from app.training_runs.preflight.compatibility import (
    check_model_tokenizer_compatibility,
    collect_missing_special_tokens,
)
from app.training_runs.preflight.config_validation import validation_issues
from app.training_runs.preflight.local_files import (
    flatten_data_files,
    has_glob_magic,
    resolve_data_path,
    validate_local_data_files,
)
from app.training_runs.preflight.runtime_summary import default_training_device
from app.training_runs.preflight.scheduler_fixes import (
    add_scheduler_step_fix,
    build_default_scheduler,
    collect_training_config_warnings_and_fixes,
    nearest_divisible_total_batch_size,
    runtime_batch_fixes,
)
from training.dataloader_config import TrainingDataloaderConfig
from training.training_config import TrainingConfig


class FakeTokenizer:
    def __init__(self, known_tokens: set[str]) -> None:
        self.known_tokens = known_tokens

    def token_to_id(self, token: str) -> int | None:
        if token in self.known_tokens:
            return 1
        return None


def valid_training_config_payload(**overrides):
    payload = {
        "max_steps": 10,
        "total_batch_size": 16,
        "seq_len": 8,
        "sample_every": 5,
        "sampler": {"prompts": [{"prompt": "Hello", "max_tokens": 8, "temperature": 0.8, "top_k": 20}]},
        "save_every": 2,
        "optimizer": {"lr": 0.0003, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
        "lr_scheduler": {"type": "sequential", "schedulers": [{"type": "constant", "steps": 10, "factor": 1.0}]},
    }
    payload.update(overrides)
    return payload


def test_validation_issues_humanizes_scheduler_total_error() -> None:
    payload = valid_training_config_payload(max_steps=10)
    payload["lr_scheduler"] = {"type": "sequential", "schedulers": [{"type": "constant", "steps": 5, "factor": 1.0}]}

    try:
        TrainingConfig.model_validate(payload)
    except ValidationError as exc:
        issues = validation_issues("training_config_invalid", "$.training_config", exc)
    else:
        raise AssertionError("Expected invalid scheduler total to raise")

    assert issues[0].code == "training_config_invalid"
    assert issues[0].path == "$.training_config"
    assert issues[0].message == (
        "LR scheduler steps must add up to max_steps. "
        "Use the suggested fix below or edit training_config.lr_scheduler."
    )


def test_scheduler_helpers_preserve_existing_fix_payloads() -> None:
    fixes = []

    add_scheduler_step_fix(
        fixes,
        {"max_steps": 12, "lr_scheduler": {"schedulers": [{"type": "constant", "steps": 3, "factor": 1.0}]}},
    )

    assert fixes[0].code == "match_scheduler_steps_to_max_steps"
    assert fixes[0].value == build_default_scheduler(12)
    assert nearest_divisible_total_batch_size(30, 8) == 32


def test_training_config_warnings_and_runtime_fixes() -> None:
    config = TrainingConfig.model_validate(
            valid_training_config_payload(
                total_batch_size=18,
                sample_every=20,
                save_every=8,
                optimizer={"lr": 0.02, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
        )
    )

    warnings, fixes = collect_training_config_warnings_and_fixes(config)
    runtime_fixes = runtime_batch_fixes(
        "total_batch_size must be divisible by seq_len * world_size.",
        config,
        max_memory_batch_size=4,
    )

    assert [warning.code for warning in warnings] == [
        "save_every_sparse",
        "sample_every_exceeds_max_steps",
        "optimizer_lr_high",
    ]
    assert {fix.code for fix in fixes} == {
        "set_save_every_to_periodic_cadence",
        "set_sample_every_to_run_cadence",
        "set_optimizer_lr_to_starter_safe_value",
    }
    assert any(fix.code == "make_total_batch_size_divisible" for fix in runtime_fixes)


def test_local_file_helpers_validate_nested_missing_paths(tmp_path: Path) -> None:
    existing = tmp_path / "train.txt"
    existing.write_text("hello", encoding="utf-8")
    config = TrainingDataloaderConfig.model_validate(
        {
            "datasets": [
                {
                    "name": "local-text",
                    "data_files": {"train": [str(existing), str(tmp_path / "missing.txt")]},
                }
            ]
        }
    )

    issues = validate_local_data_files(config)

    assert flatten_data_files({"train": ["a", ("b", {"nested": "c"})]}) == ["a", "b", "c"]
    assert has_glob_magic("data/*.txt") is True
    assert [issue.code for issue in issues] == ["local_dataset_file_missing"]
    assert "missing.txt" in issues[0].message


def test_default_shake_dataset_resolves_to_runtime_resource() -> None:
    resolved = resolve_data_path("datasets/shake.txt")

    assert resolved.name == "shake.txt"
    assert resolved.is_file()
    assert "apps/llm-studio/api/datasets/shake.txt" in resolved.as_posix()


def test_compatibility_helpers_report_token_and_shape_issues() -> None:
    dataloader_config = TrainingDataloaderConfig.model_validate(
        {
            "datasets": [{"name": "dataset"}],
            "add_bos": True,
            "bos_token": "<bos>",
            "add_eos": True,
            "eos_token": "<eos>",
            "drop_last": False,
            "pad_token": "<pad>",
        }
    )
    training_config = TrainingConfig.model_validate(valid_training_config_payload(seq_len=8))
    missing_tokens = collect_missing_special_tokens(FakeTokenizer({"<eos>"}), dataloader_config)  # type: ignore[arg-type]

    result = check_model_tokenizer_compatibility(
        model_config={"context_length": 4, "vocab_size": 100},
        tokenizer_vocab_size=101,
        training_config=training_config,
        dataloader_config=dataloader_config,
        missing_special_tokens=missing_tokens,
    )

    assert missing_tokens == ["<bos>", "<pad>"]
    assert result.compatibility.missing_special_tokens == ["<bos>", "<pad>"]
    assert [error.code for error in result.errors] == [
        "vocab_size_mismatch",
        "seq_len_exceeds_context_length",
        "missing_special_token",
        "missing_special_token",
    ]
    assert result.fixes[0].code == "set_seq_len_to_context_length"


def test_asset_and_device_small_helpers(monkeypatch) -> None:
    monkeypatch.delenv("LLM_STUDIO_TRAINING_DEVICE", raising=False)
    monkeypatch.setattr("training.utils.torch.cuda.is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr("training.utils.torch.backends.mps.is_available", lambda: False)

    assert tokenizer_display_name({"name": "  Tokenizer Name  "}, "job123456", None) == "Tokenizer Name"
    assert tokenizer_display_name({}, "job123456", "tokenizer.json") == "tokenizer.json"
    assert default_training_device().type == "cpu"
