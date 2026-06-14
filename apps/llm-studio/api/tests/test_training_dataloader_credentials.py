from __future__ import annotations

from typing import Any

from training.dataloader import TrainingTokenDataset
from training.dataloader_config import TrainingDataloaderConfig
from training.runner import TrainingRunPaths, TrainingStateWriter


class MinimalTokenizer:
    def encode_batch(self, _texts: list[str]) -> list[Any]:
        return []

    def token_to_id(self, _token: str) -> int | None:
        return None


def _config(*, embedded_token: str | None = None) -> TrainingDataloaderConfig:
    dataset: dict[str, Any] = {
        "name": "private-dataset",
        "split": "train",
        "streaming": True,
        "text_columns": ["text"],
    }
    if embedded_token is not None:
        dataset["hf_token"] = embedded_token
    return TrainingDataloaderConfig.model_validate({"datasets": [dataset]})


def test_training_loader_reads_hf_token_from_execution_environment(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    hf_token = "hf_0123456789abcdef0123456789abcdef"
    monkeypatch.setenv("LLM_STUDIO_HF_DATASET_TOKENS", f'[\"{hf_token}\"]')
    monkeypatch.setattr(
        "training.dataloader._load_hf_dataset",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs) or [],
    )

    dataset = TrainingTokenDataset(_config(), MinimalTokenizer(), seq_len=8)
    dataset._load_dataset(0, dataset.config.datasets[0], None)

    assert captured["kwargs"]["token"] == hf_token


def test_training_loader_prefers_legacy_embedded_token_over_environment(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setenv("LLM_STUDIO_HF_DATASET_TOKENS", '["hf_environment"]')
    monkeypatch.setattr(
        "training.dataloader._load_hf_dataset",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs) or [],
    )

    dataset = TrainingTokenDataset(
        _config(embedded_token="hf_legacy_embedded"),
        MinimalTokenizer(),
        seq_len=8,
    )
    dataset._load_dataset(0, dataset.config.datasets[0], None)

    assert captured["kwargs"]["token"] == "hf_legacy_embedded"


def test_training_state_writer_redacts_arbitrary_execution_token(monkeypatch, tmp_path) -> None:
    secret = "legacy-private-token-without-provider-prefix"
    monkeypatch.setenv("LLM_STUDIO_HF_DATASET_TOKENS", f'["{secret}"]')
    paths = TrainingRunPaths.from_output_dir(tmp_path)
    writer = TrainingStateWriter("job123456", paths)

    writer.finalize(
        status="failed",
        state="failed",
        stage="Failed",
        error=f"provider echoed {secret}",
    )

    assert secret not in paths.state_path.read_text(encoding="utf-8")
    assert secret not in paths.metadata_path.read_text(encoding="utf-8")
