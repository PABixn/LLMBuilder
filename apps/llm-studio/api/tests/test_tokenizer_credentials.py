from __future__ import annotations

import json
from pathlib import Path

from app.schemas import load_json
from app.tokenizer_jobs import TrainingJobManager
from app.tokenizer_models import TrainTokenizerRequest
from app.tokenizer_storage import StudioStore

API_ROOT = Path(__file__).resolve().parents[1]


class CapturingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def submit(self, *args: object) -> None:
        self.calls.append(args)


class FailingExecutor:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def submit(self, *_args: object) -> None:
        raise RuntimeError(f"dataset provider echoed {self.secret}")


def test_tokenizer_hf_token_is_only_in_in_memory_execution_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    hf_token = "hf_0123456789abcdef0123456789abcdef"
    monkeypatch.setattr("app.tokenizer_jobs.ensure_free_space", lambda *_args, **_kwargs: 0)
    database_path = tmp_path / "tokenizer.db"
    store = StudioStore(url=f"sqlite:///{database_path}")
    store.initialize()
    executor = CapturingExecutor()
    manager = TrainingJobManager.__new__(TrainingJobManager)
    manager._store = store
    manager._executor = executor
    request = TrainTokenizerRequest.model_validate(
        {
            "tokenizer_config": load_json(API_ROOT / "templates" / "tok_config.json"),
            "dataloader_config": load_json(API_ROOT / "templates" / "dataloader_config.json"),
            "hf_token": hf_token,
            "evaluation_thresholds": [5],
        }
    )

    response = manager.create_job(request)
    stored = store.get_job(response.id)

    assert stored is not None
    assert hf_token not in json.dumps(stored.dataloader_config)
    assert hf_token not in response.model_dump_json()
    assert hf_token.encode() not in database_path.read_bytes()
    assert len(executor.calls) == 1
    execution_config = executor.calls[0][3]
    assert hf_token in execution_config.model_dump_json()


def test_tokenizer_arbitrary_hf_token_is_redacted_from_queue_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    secret = "legacy-private-token-without-provider-prefix"
    monkeypatch.setattr("app.tokenizer_jobs.ensure_free_space", lambda *_args, **_kwargs: 0)
    database_path = tmp_path / "tokenizer.db"
    store = StudioStore(url=f"sqlite:///{database_path}")
    store.initialize()
    manager = TrainingJobManager.__new__(TrainingJobManager)
    manager._store = store
    manager._executor = FailingExecutor(secret)
    request = TrainTokenizerRequest.model_validate(
        {
            "tokenizer_config": load_json(API_ROOT / "templates" / "tok_config.json"),
            "dataloader_config": load_json(API_ROOT / "templates" / "dataloader_config.json"),
            "hf_token": secret,
            "evaluation_thresholds": [5],
        }
    )

    response = manager.create_job(request)
    stored = store.get_job(response.id)

    assert stored is not None
    assert stored.error is not None
    assert secret not in stored.error
    assert secret not in response.model_dump_json()
    assert secret.encode() not in database_path.read_bytes()
    assert manager._known_dataset_hf_tokens(response.id) == ()
