from __future__ import annotations

import copy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import config
from app.schemas import load_json


REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_TEMPLATE_PATH = REPO_ROOT / "model" / "gpt2_config.json"


class FakeProcess:
    def __init__(self, pid: int = 43210) -> None:
        self.pid = pid
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self._returncode = 2


def _load_app_module() -> object:
    import app.main as main_module

    return importlib.reload(main_module)


def _small_model_config(vocab_size: int) -> dict[str, object]:
    return {
        "context_length": 16,
        "vocab_size": vocab_size,
        "n_embd": 8,
        "weight_tying": True,
        "blocks": [
            {
                "components": [
                    {"norm": {"type": "layernorm"}},
                    {"attention": {"n_head": 2, "n_kv_head": 2}},
                    {"norm": {"type": "layernorm"}},
                    {
                        "mlp": {
                            "multiplier": 2,
                            "sequence": [
                                {"linear": {"bias": True}},
                                {"activation": {"type": "relu"}},
                                {"linear": {"bias": True}},
                            ],
                        }
                    },
                ]
            }
        ],
    }


def _training_payload(local_dataset_path: Path) -> dict[str, object]:
    return {
        "training_config": {
            "max_steps": 6,
            "total_batch_size": 32,
            "seq_len": 8,
            "sample_every": 3,
            "sampler": {
                "prompts": [
                    {
                        "prompt": "hello",
                        "max_tokens": 4,
                        "temperature": 0.5,
                        "top_k": 2,
                    }
                ]
            },
            "save_every": 3,
            "optimizer": {
                "lr": 0.001,
                "weight_decay": 0.01,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
            },
            "lr_scheduler": {
                "type": "sequential",
                "schedulers": [
                    {
                        "type": "linear",
                        "steps": 2,
                        "start_factor": 0.2,
                        "end_factor": 1.0,
                    },
                    {
                        "type": "cosine_annealing",
                        "steps": 4,
                        "eta_min": 1e-5,
                    },
                ],
            },
        },
        "dataloader_config": {
            "datasets": [
                {
                    "name": "text",
                    "data_files": {"train": str(local_dataset_path)},
                    "split": "train",
                    "text_columns": ["text"],
                    "weight": 1.0,
                    "streaming": True,
                }
            ],
            "add_eos": True,
            "eos_token": "<|endoftext|>",
            "drop_last": True,
            "mixing": {
                "seed": 42,
                "exhausted_policy": "stop",
            },
        },
    }


def test_training_batch_runtime_plan_uses_largest_valid_auto_micro_batch() -> None:
    from training.training_config import derive_batch_runtime_plan

    auto_plan = derive_batch_runtime_plan(
        total_batch_size=8192,
        seq_len=128,
        max_memory_batch_size=64,
    )
    assert auto_plan.micro_batch_size == 64
    assert auto_plan.grad_accum_steps == 1
    assert auto_plan.tokens_per_micro_step == 8192

    explicit_plan = derive_batch_runtime_plan(
        total_batch_size=8192,
        seq_len=128,
        max_memory_batch_size=64,
        requested_micro_batch_size=8,
    )
    assert explicit_plan.micro_batch_size == 8
    assert explicit_plan.grad_accum_steps == 8


def test_batch_and_lr_recommendation_uses_gpt2_scale_anchor_when_unconstrained() -> None:
    from app.training_recommendations import _model_target_total_batch_size

    target = _model_target_total_batch_size(
        total_parameters=124_000_000,
        seq_len=1024,
        max_memory_micro_batch_size=64,
        max_grad_accum=64,
        dataset_cap_tokens=None,
    )

    assert target == 524288


def test_batch_and_lr_recommendation_keeps_small_models_substantial_when_unconstrained() -> None:
    from app.training_recommendations import _model_target_total_batch_size

    target = _model_target_total_batch_size(
        total_parameters=11_000_000,
        seq_len=256,
        max_memory_micro_batch_size=64,
        max_grad_accum=64,
        dataset_cap_tokens=None,
    )

    assert target == 262144


def test_batch_and_lr_recommendation_prefers_power_of_two_token_batches() -> None:
    from app.training_recommendations import _model_target_total_batch_size

    target = _model_target_total_batch_size(
        total_parameters=123_000_000,
        seq_len=1024,
        max_memory_micro_batch_size=7,
        max_grad_accum=64,
        dataset_cap_tokens=None,
    )

    assert target == 262144
    assert target & (target - 1) == 0


def test_batch_and_lr_recommendation_preserves_canonical_learning_rates() -> None:
    from app.training_recommendations import (
        _DatasetSummary,
        _ModelSummary,
        _ScheduleSummary,
        _recommend_learning_rate,
        _round_learning_rate_to_canonical_mantissa,
    )

    assert _round_learning_rate_to_canonical_mantissa(3.1e-4, lower=6e-5, upper=9e-4) == 3e-4
    assert _round_learning_rate_to_canonical_mantissa(1.7e-4, lower=6e-5, upper=9e-4) == 2e-4

    model_summary = _ModelSummary(
        total_parameters=124_000_000,
        parameter_memory_bytes_bf16=248_000_000,
        estimated_kv_cache_bytes_for_context_fp16=0,
        block_count=12,
        attention_component_count=12,
        max_mlp_multiplier=4.0,
        activation_types=("gelu",),
        norm_types=("layernorm",),
        uses_gqa=False,
        weight_tying=True,
    )
    dataset_summary = _DatasetSummary(
        dataset_count=1,
        local_dataset_count=0,
        streaming_dataset_count=1,
        local_file_count=0,
        local_total_size_bytes=None,
        dominant_dataset_weight=1.0,
        dataset_scale="streaming",
        approx_local_tokens=None,
        step_budget_cap_tokens=None,
        tokenizer_bytes_per_token_assumption=4.0,
    )
    schedule_summary = _ScheduleSummary(
        peak_factor=1.0,
        warmup_fraction=0.1,
        label="Warmup + cosine decay",
    )

    assert (
        _recommend_learning_rate(
            total_batch_size=262_144,
            seq_len=1_024,
            model_summary=model_summary,
            dataset_summary=dataset_summary,
            schedule_summary=schedule_summary,
            variant_multiplier=1.0,
        )
        == 3e-4
    )


def test_batch_and_lr_recommendation_keeps_named_profiles_even_when_values_match() -> None:
    from app.training_recommendations import (
        _BatchCandidate,
        _DatasetSummary,
        _ModelSummary,
        _ScheduleSummary,
        _build_options,
    )
    from training.training_config import TrainingConfig

    training_config = TrainingConfig.model_validate(
        {
            "max_steps": 6,
            "total_batch_size": 32,
            "seq_len": 8,
            "sample_every": 3,
            "sampler": {
                "prompts": [
                    {
                        "prompt": "hello",
                        "max_tokens": 4,
                        "temperature": 0.5,
                        "top_k": 2,
                    }
                ]
            },
            "save_every": 3,
            "optimizer": {
                "lr": 0.001,
                "weight_decay": 0.01,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
            },
            "lr_scheduler": {
                "type": "sequential",
                "schedulers": [
                    {
                        "type": "linear",
                        "steps": 2,
                        "start_factor": 0.2,
                        "end_factor": 1.0,
                    },
                    {
                        "type": "cosine_annealing",
                        "steps": 4,
                        "eta_min": 1e-5,
                    },
                ],
            },
        }
    )
    model_summary = _ModelSummary(
        total_parameters=124_000_000,
        parameter_memory_bytes_bf16=248_000_000,
        estimated_kv_cache_bytes_for_context_fp16=0,
        block_count=12,
        attention_component_count=12,
        max_mlp_multiplier=4.0,
        activation_types=("gelu",),
        norm_types=("layernorm",),
        uses_gqa=False,
        weight_tying=True,
    )
    dataset_summary = _DatasetSummary(
        dataset_count=1,
        local_dataset_count=0,
        streaming_dataset_count=1,
        local_file_count=0,
        local_total_size_bytes=None,
        dominant_dataset_weight=1.0,
        dataset_scale="streaming",
        approx_local_tokens=None,
        step_budget_cap_tokens=None,
        tokenizer_bytes_per_token_assumption=4.0,
    )
    schedule_summary = _ScheduleSummary(
        peak_factor=1.0,
        warmup_fraction=0.1,
        label="Warmup + cosine decay",
    )
    candidate = _BatchCandidate(
        total_batch_size=262_144,
        micro_batch_size=32,
        grad_accum_steps=8,
    )

    options = _build_options(
        training_config=training_config,
        model_summary=model_summary,
        dataset_summary=dataset_summary,
        schedule_summary=schedule_summary,
        current_micro_batch_size=None,
        balanced_candidate=candidate,
        stability_candidate=candidate,
        throughput_candidate=candidate,
    )

    assert [option.key for option in options] == ["balanced", "stability", "throughput"]
    assert [option.learning_rate for option in options] == [3e-4, 2e-4, 4e-4]


def test_training_dataloader_config_accepts_hf_token() -> None:
    from training.dataloader_config import TrainingDataloaderConfig

    config = TrainingDataloaderConfig.model_validate(
        {
            "datasets": [
                {
                    "name": "private-dataset",
                    "split": "train",
                    "streaming": True,
                    "hf_token": "hf_test_token",
                    "text_columns": ["text"],
                }
            ]
        }
    )
    assert config.datasets[0].hf_token == "hf_test_token"


def test_health_and_static_fallback(monkeypatch, tmp_path: Path) -> None:
    web_dist = tmp_path / "web-dist"
    web_dist.mkdir(parents=True, exist_ok=True)
    (web_dist / "index.html").write_text(
        "<html><body><h1>LLM Studio</h1></body></html>",
        encoding="utf-8",
    )
    (web_dist / "asset.txt").write_text("asset-ok", encoding="utf-8")

    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_STUDIO_WEB_DIST_DIR", str(web_dist))
    monkeypatch.setenv("LLM_STUDIO_SERVE_WEB", "1")
    config.reset_settings_cache()

    module = _load_app_module()

    with TestClient(module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"ok": True}

        api_health = client.get("/api/v1/health")
        assert api_health.status_code == 200
        assert api_health.json() == {"ok": True}
        tokenizer_health = client.get("/api/v1/tokenizer/health")
        assert tokenizer_health.status_code == 200
        assert tokenizer_health.json() == {"ok": True}
        training_health = client.get("/api/v1/training/health")
        assert training_health.status_code == 200
        assert training_health.json() == {"ok": True}

        index = client.get("/")
        assert index.status_code == 200
        assert "LLM Studio" in index.text

        asset = client.get("/asset.txt")
        assert asset.status_code == 200
        assert asset.text == "asset-ok"

        spa_fallback = client.get("/deep/link/that/does/not/exist")
        assert spa_fallback.status_code == 200
        assert "LLM Studio" in spa_fallback.text

        missing_api = client.get("/api/v1/unknown-route")
        assert missing_api.status_code == 404


def test_tokenizer_endpoints_validate_and_file_stats(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    sample_file = data_dir / "datasets" / "sample.txt"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    content = "ab\né🙂"
    sample_file.write_text(content, encoding="utf-8")

    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(data_dir))
    config.reset_settings_cache()
    module = _load_app_module()

    with TestClient(module.app) as client:
        templates = client.get("/api/v1/tokenizer/config/templates")
        assert templates.status_code == 200
        template_payload = templates.json()
        assert "tokenizer_config_template" in template_payload
        assert "dataloader_config_template" in template_payload

        tokenizer_validation = client.post(
            "/api/v1/tokenizer/validate/tokenizer",
            json={"config": template_payload["tokenizer_config_template"]},
        )
        assert tokenizer_validation.status_code == 200
        tokenizer_validation_body = tokenizer_validation.json()
        assert tokenizer_validation_body["valid"] is True
        assert isinstance(tokenizer_validation_body["normalized_config"], dict)

        dataloader_validation = client.post(
            "/api/v1/tokenizer/validate/dataloader",
            json={"config": template_payload["dataloader_config_template"]},
        )
        assert dataloader_validation.status_code == 200
        dataloader_validation_body = dataloader_validation.json()
        assert dataloader_validation_body["valid"] is True
        assert isinstance(dataloader_validation_body["normalized_config"], dict)

        file_stats = client.get(
            "/api/v1/tokenizer/files/stats",
            params={"file_path": str(sample_file)},
        )
        assert file_stats.status_code == 200
        stats_body = file_stats.json()
        assert stats_body["file_name"] == sample_file.name
        assert stats_body["size_chars"] == len(content)
        assert stats_body["size_bytes"] == len(content.encode("utf-8"))


def test_training_endpoints_validate_and_preflight(monkeypatch, tmp_path: Path) -> None:
    from app.tokenizer_models import JobStatus
    from app.tokenizer_storage import StoredJob
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    local_dataset = data_dir / "datasets" / "train.txt"
    local_dataset.parent.mkdir(parents=True, exist_ok=True)
    local_dataset.write_text("hello world\nhello again\n", encoding="utf-8")

    tokenizer_output = data_dir / "artifacts" / "tokenizers"
    tokenizer_output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer_output / "tiny-tokenizer.json"
    tokenizer = Tokenizer(WordLevel({"<|endoftext|>": 0, "hello": 1, "world": 2}, unk_token="<|endoftext|>"))
    tokenizer.save(str(tokenizer_path))

    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(data_dir))
    config.reset_settings_cache()
    module = _load_app_module()
    payload = _training_payload(local_dataset)

    with TestClient(module.app) as client:
        store = module.app.state.tokenizer_store
        store.create_job(
            StoredJob(
                id="completed-tokenizer",
                status=JobStatus.completed,
                stage="Completed",
                progress=1.0,
                created_at=datetime.now(timezone.utc),
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
                tokenizer_config={"name": "tiny-tokenizer"},
                dataloader_config={"source": "local"},
                evaluation_thresholds=[5],
                evaluation_text_path="__training_dataset__",
                artifact_file=tokenizer_path.name,
                artifact_path=str(tokenizer_path),
                stats=None,
                error=None,
            )
        )

        project = client.post(
            "/api/v1/projects",
            json={"name": "tiny-model", "model_config": _small_model_config(vocab_size=3)},
        )
        assert project.status_code == 201
        project_id = project.json()["id"]

        templates = client.get("/api/v1/training/config/templates")
        assert templates.status_code == 200
        templates_body = templates.json()
        assert "training_config_template" in templates_body
        assert "dataloader_config_template" in templates_body

        schemas = client.get("/api/v1/training/config/schemas")
        assert schemas.status_code == 200
        schemas_body = schemas.json()
        assert "training_config_schema" in schemas_body
        assert "dataloader_schema" in schemas_body

        validate_training = client.post(
            "/api/v1/training/validate/training-config",
            json={"config": payload["training_config"]},
        )
        assert validate_training.status_code == 200
        assert validate_training.json()["valid"] is True

        validate_dataloader = client.post(
            "/api/v1/training/validate/dataloader",
            json={"config": payload["dataloader_config"]},
        )
        assert validate_dataloader.status_code == 200
        assert validate_dataloader.json()["valid"] is True

        preflight = client.post(
            "/api/v1/training/validate/preflight",
            json={
                "project_id": project_id,
                "tokenizer_job_id": "completed-tokenizer",
                **payload,
            },
        )
        assert preflight.status_code == 200
        preflight_body = preflight.json()
        assert preflight_body["valid"] is True
        assert preflight_body["compatibility"]["tokenizer_vocab_size"] == 3
        assert preflight_body["compatibility"]["model_vocab_size"] == 3
        assert preflight_body["derived_runtime"]["micro_batch_size"] > 0
        assert preflight_body["memory_estimate"]["max_batch_size"] > 0
        recommendation = preflight_body["batch_and_lr_recommendation"]
        assert recommendation is not None
        assert recommendation["recommended_option_key"] == "balanced"
        recommended_option = next(
            option for option in recommendation["options"] if option["key"] == recommendation["recommended_option_key"]
        )
        assert recommended_option["total_batch_size"] % payload["training_config"]["seq_len"] == 0
        assert recommended_option["micro_batch_size"] > 0
        assert recommended_option["grad_accum_steps"] > 0
        assert recommended_option["learning_rate"] > 0
        assert recommendation["signals"]["max_memory_micro_batch_size"] > 0
        assert recommendation["signals"]["local_file_count"] == 1
        assert {issue["code"] for issue in preflight_body["warnings"]} == {"save_every_sparse"}
        fix_codes = {fix["code"] for fix in preflight_body["recommended_fixes"]}
        save_fix = next(
            fix for fix in preflight_body["recommended_fixes"] if fix["code"] == "set_save_every_to_periodic_cadence"
        )
        assert save_fix["value"] == 1
        assert "load_starter_optimizer_defaults" not in fix_codes
        assert "load_starter_scheduler_template" not in fix_codes

        sparse_save_payload = copy.deepcopy(payload)
        sparse_save_payload["training_config"]["save_every"] = 61
        sparse_save_preflight = client.post(
            "/api/v1/training/validate/preflight",
            json={
                "project_id": project_id,
                "tokenizer_job_id": "completed-tokenizer",
                **sparse_save_payload,
            },
        )
        assert sparse_save_preflight.status_code == 200
        sparse_save_body = sparse_save_preflight.json()
        assert "save_every_sparse" in {issue["code"] for issue in sparse_save_body["warnings"]}

        invalid_batch_payload = copy.deepcopy(payload)
        invalid_batch_payload["training_config"]["total_batch_size"] = 30
        invalid_batch_preflight = client.post(
            "/api/v1/training/validate/preflight",
            json={
                "project_id": project_id,
                "tokenizer_job_id": "completed-tokenizer",
                **invalid_batch_payload,
            },
        )
        assert invalid_batch_preflight.status_code == 200
        invalid_batch_body = invalid_batch_preflight.json()
        assert invalid_batch_body["valid"] is False
        assert "invalid_micro_batch_size" in {issue["code"] for issue in invalid_batch_body["errors"]}
        assert invalid_batch_body["batch_and_lr_recommendation"] is not None

        mismatch_project = client.post(
            "/api/v1/projects",
            json={"name": "mismatched-vocab-model", "model_config": _small_model_config(vocab_size=4)},
        )
        assert mismatch_project.status_code == 201
        mismatch_preflight = client.post(
            "/api/v1/training/validate/preflight",
            json={
                "project_id": mismatch_project.json()["id"],
                "tokenizer_job_id": "completed-tokenizer",
                **payload,
            },
        )
        assert mismatch_preflight.status_code == 200
        mismatch_body = mismatch_preflight.json()
        vocab_error = next(
            issue for issue in mismatch_body["errors"] if issue["code"] == "vocab_size_mismatch"
        )
        assert vocab_error["path"] == "$.model_config.vocab_size"

        scheduler_payload = copy.deepcopy(payload)
        scheduler_payload["training_config"]["max_steps"] = 301
        scheduler_preflight = client.post(
            "/api/v1/training/validate/preflight",
            json={
                "project_id": project_id,
                "tokenizer_job_id": "completed-tokenizer",
                **scheduler_payload,
            },
        )
        assert scheduler_preflight.status_code == 200
        scheduler_body = scheduler_preflight.json()
        assert scheduler_body["valid"] is False
        scheduler_error = next(
            issue for issue in scheduler_body["errors"] if issue["code"] == "training_config_invalid"
        )
        assert scheduler_error["message"] == (
            "LR scheduler steps must add up to max_steps. "
            "Use the suggested fix below or edit training_config.lr_scheduler."
        )
        assert scheduler_error["path"] == "$.training_config"
        scheduler_fix = next(
            fix for fix in scheduler_body["recommended_fixes"] if fix["code"] == "match_scheduler_steps_to_max_steps"
        )
        assert sum(item["steps"] for item in scheduler_fix["value"]["schedulers"]) == 301


def test_training_job_endpoints_round_trip(monkeypatch, tmp_path: Path) -> None:
    from app.tokenizer_models import JobStatus
    from app.tokenizer_storage import StoredJob
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    data_dir = tmp_path / "data"
    local_dataset = data_dir / "datasets" / "train.txt"
    local_dataset.parent.mkdir(parents=True, exist_ok=True)
    local_dataset.write_text("hello world\nhello again\n", encoding="utf-8")

    tokenizer_output = data_dir / "artifacts" / "tokenizers"
    tokenizer_output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer_output / "tiny-tokenizer.json"
    tokenizer = Tokenizer(WordLevel({"<|endoftext|>": 0, "hello": 1, "world": 2}, unk_token="<|endoftext|>"))
    tokenizer.save(str(tokenizer_path))

    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(data_dir))
    config.reset_settings_cache()
    module = _load_app_module()
    payload = _training_payload(local_dataset)

    with TestClient(module.app) as client:
        tokenizer_store = module.app.state.tokenizer_store
        tokenizer_store.create_job(
            StoredJob(
                id="completed-tokenizer",
                status=JobStatus.completed,
                stage="Completed",
                progress=1.0,
                created_at=datetime.now(timezone.utc),
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
                tokenizer_config={"name": "tiny-tokenizer"},
                dataloader_config={"source": "local"},
                evaluation_thresholds=[5],
                evaluation_text_path="__training_dataset__",
                artifact_file=tokenizer_path.name,
                artifact_path=str(tokenizer_path),
                stats=None,
                error=None,
            )
        )

        project = client.post(
            "/api/v1/projects",
            json={"name": "tiny-model", "model_config": _small_model_config(vocab_size=3)},
        )
        assert project.status_code == 201
        project_id = project.json()["id"]

        manager = module.app.state.training_jobs

        def fake_spawn_process(**kwargs):
            output_dir = kwargs["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "runtime_state.json").write_text(
                json.dumps(
                    {
                        "job_id": kwargs["job_id"],
                        "status": "running",
                        "state": "training",
                        "stage": "Training step 3 / 6",
                        "progress": 0.5,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "last_step": 3,
                        "max_steps": 6,
                        "elapsed_seconds": 20.5,
                        "eta_seconds": 19.5,
                        "latest_loss": 1.23,
                        "latest_grad_norm": 0.45,
                        "latest_lr": 0.001,
                        "latest_tokens_per_sec": 512.0,
                        "checkpoint_count": 1,
                        "sample_count": 1,
                        "resolved_runtime": {
                            "device": "cpu",
                            "device_type": "cpu",
                            "micro_batch_size": 4,
                            "tokens_per_micro_step": 32,
                            "tokens_per_world_step": 32,
                            "grad_accum_steps": 1,
                            "max_batch_size_from_total": 4,
                            "max_batch_size_from_memory": 8,
                            "max_allowed_batch_size": 4,
                            "ddp": False,
                            "ddp_rank": 0,
                            "ddp_world_size": 1,
                        },
                        "memory_estimate": {"max_batch_size": 8},
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "job_id": kwargs["job_id"],
                        "status": "running",
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "stats.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "step": index,
                            "loss": round(2.0 - index * 0.005, 3),
                            "norm": round(0.8 - index * 0.001, 3),
                            "dt": 0.1,
                            "tok_per_sec": 256 + index,
                            "lr": 0.001,
                        }
                    )
                    + "\n"
                    for index in range(1, 206)
                ),
                encoding="utf-8",
            )
            (output_dir / "samples.jsonl").write_text(
                json.dumps(
                    {
                        "step": 3,
                        "samples": [
                            {"index": 0, "prompt": "hello", "text": "hello world"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "stdout.log").write_text(
                "".join(f"training line {index}\n" for index in range(1, 206)),
                encoding="utf-8",
            )
            (output_dir / "stderr.log").write_text("", encoding="utf-8")
            checkpoint_dir = output_dir / "checkpoints" / "3"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "model-3.pt").write_text("weights", encoding="utf-8")
            (checkpoint_dir / "meta-3.json").write_text("{}", encoding="utf-8")
            return FakeProcess()

        manager._spawn_process = fake_spawn_process

        create = client.post(
            "/api/v1/training/jobs",
            json={
                "name": "run-one",
                "project_id": project_id,
                "tokenizer_job_id": "completed-tokenizer",
                **payload,
            },
        )
        assert create.status_code == 201
        created = create.json()
        job_id = created["id"]
        assert created["name"] == "run-one"
        assert created["project_id"] == project_id
        assert created["tokenizer_job_id"] == "completed-tokenizer"
        assert created["latest_loss"] == 0.975
        assert created["checkpoint_count"] == 1
        assert created["elapsed_seconds"] == 20.5
        assert created["eta_seconds"] == 19.5

        listing = client.get("/api/v1/training/jobs")
        assert listing.status_code == 200
        listing_jobs = listing.json()["jobs"]
        listed = next(job for job in listing_jobs if job["id"] == job_id)
        assert listed["elapsed_seconds"] == 20.5
        assert listed["eta_seconds"] == 19.5

        detail = client.get(f"/api/v1/training/jobs/{job_id}")
        assert detail.status_code == 200
        assert detail.json()["last_step"] == 205
        assert detail.json()["elapsed_seconds"] == 20.5
        assert detail.json()["eta_seconds"] == 19.5

        metrics = client.get(f"/api/v1/training/jobs/{job_id}/metrics")
        assert metrics.status_code == 200
        assert len(metrics.json()["metrics"]) == 205
        assert metrics.json()["metrics"][0]["step"] == 1
        assert metrics.json()["metrics"][-1]["step"] == 205

        tailed_metrics = client.get(f"/api/v1/training/jobs/{job_id}/metrics", params={"limit": 2})
        assert tailed_metrics.status_code == 200
        assert [item["step"] for item in tailed_metrics.json()["metrics"]] == [204, 205]

        samples = client.get(f"/api/v1/training/jobs/{job_id}/samples")
        assert samples.status_code == 200
        assert samples.json()["samples"][0]["samples"][0]["text"] == "hello world"

        logs = client.get(f"/api/v1/training/jobs/{job_id}/logs")
        assert logs.status_code == 200
        assert len(logs.json()["stdout_lines"]) == 205
        assert logs.json()["stdout_lines"][0] == "training line 1"
        assert logs.json()["stdout_lines"][-1] == "training line 205"

        tailed_logs = client.get(f"/api/v1/training/jobs/{job_id}/logs", params={"lines": 2})
        assert tailed_logs.status_code == 200
        assert tailed_logs.json()["stdout_lines"] == ["training line 204", "training line 205"]

        checkpoints = client.get(f"/api/v1/training/jobs/{job_id}/checkpoints")
        assert checkpoints.status_code == 200
        assert checkpoints.json()["checkpoints"][0]["step"] == 3

        stop = client.post(f"/api/v1/training/jobs/{job_id}/stop")
        assert stop.status_code == 200
        assert stop.json()["status"] == "cancelled"

        artifact = client.get(f"/api/v1/training/jobs/{job_id}/artifact")
        assert artifact.status_code == 200
        assert artifact.headers["content-type"] == "application/zip"

        deleted = client.delete(f"/api/v1/training/jobs/{job_id}")
        assert deleted.status_code == 204

        missing = client.get(f"/api/v1/training/jobs/{job_id}")
        assert missing.status_code == 404


def test_validate_model_endpoint_reports_semantic_issues(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(tmp_path / "data"))
    config.reset_settings_cache()
    module = _load_app_module()
    base_config = load_json(MODEL_TEMPLATE_PATH)

    with TestClient(module.app) as client:
        warning_config = copy.deepcopy(base_config)
        warning_config["n_embd"] = 780  # 780 / 12 = 65 (odd head_dim)
        warning_resp = client.post("/api/v1/validate/model", json={"config": warning_config})
        assert warning_resp.status_code == 200
        warning_body = warning_resp.json()
        assert warning_body["valid"] is True
        warning_codes = {issue["code"] for issue in warning_body["warnings"]}
        assert "head_dim_not_even" in warning_codes
        assert warning_body["errors"] == []

        error_config = copy.deepcopy(base_config)
        error_config["blocks"][0]["components"][1]["attention"]["n_kv_head"] = 16
        error_resp = client.post("/api/v1/validate/model", json={"config": error_config})
        assert error_resp.status_code == 200
        error_body = error_resp.json()
        assert error_body["valid"] is False
        error_codes = {issue["code"] for issue in error_body["errors"]}
        assert "n_kv_head_gt_n_head" in error_codes
        assert "n_head_not_divisible_by_n_kv_head" in error_codes


def test_analyze_model_endpoint_instantiates_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(tmp_path / "data"))
    config.reset_settings_cache()
    module = _load_app_module()
    base_config = load_json(MODEL_TEMPLATE_PATH)

    with TestClient(module.app) as client:
        response = client.post("/api/v1/analyze/model", json={"config": base_config})
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is True
        assert body["instantiated"] is True
        assert body["instantiation_error"] is None
        assert body["analysis"]["total_parameters"] > 0
        assert body["analysis"]["block_count"] == len(base_config["blocks"])
        assert body["analysis"]["mlp_activation_step_count"] > 0
        breakdown = body["analysis"]["parameter_breakdown"]
        assert isinstance(breakdown, list)
        assert breakdown
        assert sum(item["parameters"] for item in breakdown) == body["analysis"]["total_parameters"]
        assert (
            sum(item["trainable_parameters"] for item in breakdown)
            == body["analysis"]["trainable_parameters"]
        )
        assert all(item["percentage"] >= 0 for item in breakdown)
        assert all(item["trainable_percentage"] >= 0 for item in breakdown)


def test_projects_endpoints_round_trip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(tmp_path / "data"))
    config.reset_settings_cache()
    module = _load_app_module()
    base_config = load_json(MODEL_TEMPLATE_PATH)

    with TestClient(module.app) as client:
        create = client.post(
            "/api/v1/projects",
            json={"name": "GPT-2 baseline", "model_config": base_config},
        )
        assert create.status_code == 201
        created = create.json()
        project_id = created["id"]
        assert created["name"] == "GPT-2 baseline"
        assert created["valid"] is True
        assert created["model_config"]["n_embd"] == base_config["n_embd"]

        updated_config = copy.deepcopy(base_config)
        updated_config["n_embd"] = 1024
        update = client.put(
            f"/api/v1/projects/{project_id}",
            json={"name": "GPT-2 widened", "model_config": updated_config},
        )
        assert update.status_code == 200
        updated = update.json()
        assert updated["id"] == project_id
        assert updated["name"] == "GPT-2 widened"
        assert updated["model_config"]["n_embd"] == 1024

        listing = client.get("/api/v1/projects")
        assert listing.status_code == 200
        listing_body = listing.json()["projects"]
        listed_ids = [item["id"] for item in listing_body]
        assert project_id in listed_ids
        listed_project = next(item for item in listing_body if item["id"] == project_id)
        assert listed_project["name"] == "GPT-2 widened"

        detail = client.get(f"/api/v1/projects/{project_id}")
        assert detail.status_code == 200
        assert detail.json()["id"] == project_id
        assert detail.json()["model_config"]["n_embd"] == 1024

        artifact = client.get(f"/api/v1/projects/{project_id}/artifact")
        assert artifact.status_code == 200
        assert artifact.json()["n_embd"] == 1024

        deleted = client.delete(f"/api/v1/projects/{project_id}")
        assert deleted.status_code == 204

        missing_detail = client.get(f"/api/v1/projects/{project_id}")
        assert missing_detail.status_code == 404

        missing_artifact = client.get(f"/api/v1/projects/{project_id}/artifact")
        assert missing_artifact.status_code == 404


def test_tokenizer_job_delete_endpoint(monkeypatch, tmp_path: Path) -> None:
    from app.tokenizer_models import JobStatus
    from app.tokenizer_storage import StoredJob

    data_dir = tmp_path / "data"
    tokenizer_output = data_dir / "artifacts" / "tokenizers"
    tokenizer_output.mkdir(parents=True, exist_ok=True)
    artifact_path = tokenizer_output / "demo-tokenizer.json"
    artifact_path.write_text("{\"vocab\": []}", encoding="utf-8")

    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(data_dir))
    config.reset_settings_cache()
    module = _load_app_module()
    now = datetime.now(timezone.utc)

    with TestClient(module.app) as client:
        store = module.app.state.tokenizer_store
        store.create_job(
            StoredJob(
                id="completed-job",
                status=JobStatus.completed,
                stage="Completed",
                progress=1.0,
                created_at=now,
                started_at=now,
                finished_at=now,
                tokenizer_config={"name": "demo"},
                dataloader_config={"source": "local"},
                evaluation_thresholds=[5],
                evaluation_text_path="__training_dataset__",
                artifact_file=artifact_path.name,
                artifact_path=str(artifact_path),
                stats=None,
                error=None,
            )
        )
        store.create_job(
            StoredJob(
                id="running-job",
                status=JobStatus.running,
                stage="Training tokenizer",
                progress=0.4,
                created_at=now,
                started_at=now,
                finished_at=None,
                tokenizer_config={"name": "running"},
                dataloader_config={"source": "local"},
                evaluation_thresholds=[5],
                evaluation_text_path="__training_dataset__",
                artifact_file=None,
                artifact_path=None,
                stats=None,
                error=None,
            )
        )

        delete_running = client.delete("/api/v1/tokenizer/jobs/running-job")
        assert delete_running.status_code == 409

        delete_completed = client.delete("/api/v1/tokenizer/jobs/completed-job")
        assert delete_completed.status_code == 204
        assert not artifact_path.exists()

        missing = client.get("/api/v1/tokenizer/jobs/completed-job")
        assert missing.status_code == 404
