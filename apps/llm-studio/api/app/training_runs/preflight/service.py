from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from tokenizers import Tokenizer

from ...config import get_settings
from ...dataset_credentials import strip_hf_tokens
from ...tokenizer_storage import StudioStore as TokenizerStudioStore
from ..schemas import (
    DerivedRuntimeSummary,
    TrainingAssetRef,
    TrainingBatchLrRecommendation,
    TrainingCompatibilitySummary,
    TrainingFixSuggestion,
    TrainingIssue,
    TrainingPreflightRequest,
    TrainingPreflightResponse,
)
from .assets import (
    load_config_dict,
    load_project_asset,
    load_tokenizer_asset,
    require_tokenizer_artifact_path,
)
from .compatibility import check_model_tokenizer_compatibility, collect_missing_special_tokens
from .config_validation import issue, validation_issues
from .local_files import validate_local_data_files
from .recommendations import build_batch_and_lr_recommendation
from .runtime_summary import build_runtime_summary
from .scheduler_fixes import add_scheduler_step_fix, collect_training_config_warnings_and_fixes

from training.dataloader_config import TrainingDataloaderConfig
from training.training_config import TrainingConfig


@dataclass(slots=True)
class ResolvedPreflightContext:
    valid: bool
    model_project: TrainingAssetRef
    tokenizer: TrainingAssetRef
    model_config: dict[str, Any]
    normalized_training_config: dict[str, Any]
    normalized_dataloader_config: dict[str, Any]
    warnings: list[TrainingIssue]
    errors: list[TrainingIssue]
    recommended_fixes: list[TrainingFixSuggestion]
    compatibility: TrainingCompatibilitySummary | None
    derived_runtime: DerivedRuntimeSummary | None
    memory_estimate: dict[str, Any] | None
    batch_and_lr_recommendation: TrainingBatchLrRecommendation | None


class TrainingPreflightService:
    def __init__(self, *, tokenizer_store: TokenizerStudioStore) -> None:
        self._tokenizer_store = tokenizer_store

    def build_preflight(self, request: TrainingPreflightRequest) -> TrainingPreflightResponse:
        return self.response_from_context(self.resolve_context(request))

    def response_from_context(
        self,
        context: ResolvedPreflightContext,
    ) -> TrainingPreflightResponse:
        return TrainingPreflightResponse(
            valid=context.valid,
            model_project=context.model_project,
            tokenizer=context.tokenizer,
            normalized_training_config=context.normalized_training_config,
            normalized_dataloader_config=context.normalized_dataloader_config,
            warnings=context.warnings,
            errors=context.errors,
            recommended_fixes=context.recommended_fixes,
            compatibility=context.compatibility,
            derived_runtime=context.derived_runtime,
            memory_estimate=context.memory_estimate,
            batch_and_lr_recommendation=context.batch_and_lr_recommendation,
        )

    def resolve_context(self, request: TrainingPreflightRequest) -> ResolvedPreflightContext:
        warnings: list[TrainingIssue] = []
        errors: list[TrainingIssue] = []
        fixes: list[TrainingFixSuggestion] = []

        project_ref, model_config = self.load_project_asset(request.project_id)
        tokenizer_ref, tokenizer_path = self.load_tokenizer_asset(request.tokenizer_job_id)

        try:
            parsed_training_config = TrainingConfig.model_validate(request.training_config)
            normalized_training_config = parsed_training_config.model_dump(mode="json")
        except ValidationError as exc:
            parsed_training_config = None
            normalized_training_config = request.training_config
            errors.extend(validation_issues("training_config_invalid", "$.training_config", exc))
        except Exception as exc:
            parsed_training_config = None
            normalized_training_config = request.training_config
            errors.append(issue("training_config_invalid", str(exc), "$.training_config"))

        try:
            parsed_dataloader_config = TrainingDataloaderConfig.model_validate(request.dataloader_config)
            normalized_dataloader_config = parsed_dataloader_config.model_dump(mode="json")
        except ValidationError as exc:
            parsed_dataloader_config = None
            normalized_dataloader_config = request.dataloader_config
            errors.extend(validation_issues("dataloader_config_invalid", "$.dataloader_config", exc))
        except Exception as exc:
            parsed_dataloader_config = None
            normalized_dataloader_config = request.dataloader_config
            errors.append(issue("dataloader_config_invalid", str(exc), "$.dataloader_config"))
        normalized_dataloader_config = strip_hf_tokens(normalized_dataloader_config)

        compatibility = None
        derived_runtime = None
        memory_estimate = None
        batch_and_lr_recommendation = None

        add_scheduler_step_fix(fixes, request.training_config)

        if parsed_training_config is not None and parsed_dataloader_config is not None:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer_vocab_size = tokenizer.get_vocab_size()
            missing_special_tokens = collect_missing_special_tokens(tokenizer, parsed_dataloader_config)
            compatibility_result = check_model_tokenizer_compatibility(
                model_config=model_config,
                tokenizer_vocab_size=tokenizer_vocab_size,
                training_config=parsed_training_config,
                dataloader_config=parsed_dataloader_config,
                missing_special_tokens=missing_special_tokens,
            )
            compatibility = compatibility_result.compatibility
            errors.extend(compatibility_result.errors)
            fixes.extend(compatibility_result.fixes)

            errors.extend(validate_local_data_files(parsed_dataloader_config))
            scheduler_warnings, scheduler_fixes = collect_training_config_warnings_and_fixes(parsed_training_config)
            warnings.extend(scheduler_warnings)
            fixes.extend(scheduler_fixes)

            if parsed_training_config.seq_len <= int(model_config["context_length"]):
                loaded_model_config = load_config_dict(model_config)
                runtime_result = build_runtime_summary(
                    model_config=loaded_model_config,
                    training_config=parsed_training_config,
                    dataloader_config=parsed_dataloader_config,
                )
                errors.extend(runtime_result.errors)
                fixes.extend(runtime_result.fixes)
                memory_estimate = runtime_result.memory_estimate
                derived_runtime = runtime_result.derived_runtime

                tokenizer_job = self._tokenizer_store.get_job(request.tokenizer_job_id)
                batch_and_lr_recommendation = build_batch_and_lr_recommendation(
                    model_config=loaded_model_config,
                    model=runtime_result.model,
                    training_config=parsed_training_config,
                    dataloader_config=parsed_dataloader_config,
                    tokenizer_stats=tokenizer_job.stats if tokenizer_job is not None else None,
                    memory_estimate=runtime_result.recommendation_estimate,
                    current_batch_plan=runtime_result.batch_plan,
                )

        return ResolvedPreflightContext(
            valid=not errors,
            model_project=project_ref,
            tokenizer=tokenizer_ref,
            model_config=model_config,
            normalized_training_config=normalized_training_config,
            normalized_dataloader_config=normalized_dataloader_config,
            warnings=warnings,
            errors=errors,
            recommended_fixes=fixes,
            compatibility=compatibility,
            derived_runtime=derived_runtime,
            memory_estimate=memory_estimate,
            batch_and_lr_recommendation=batch_and_lr_recommendation,
        )

    def load_project_asset(self, project_id: str) -> tuple[TrainingAssetRef, dict[str, Any]]:
        return load_project_asset(project_id, projects_dir=get_settings().projects_dir)

    def load_tokenizer_asset(self, tokenizer_job_id: str) -> tuple[TrainingAssetRef, Path]:
        return load_tokenizer_asset(tokenizer_job_id, tokenizer_store=self._tokenizer_store)

    def require_tokenizer_artifact_path(self, tokenizer_job_id: str) -> Path:
        return require_tokenizer_artifact_path(tokenizer_job_id, tokenizer_store=self._tokenizer_store)
