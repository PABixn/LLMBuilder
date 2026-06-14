from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer

from model.loader import load_config
from model.model import ConfigurableGPT
from training.checkpoint_manager import CheckpointManager
from training.dataloader import TrainingDataLoader
from training.dataloader_config import load_training_dataloader_config
from training.logger import Logger
from training.lr_scheduler import build_lr_scheduler
from training.memory_estimator import MemoryEstimator
from training.training_config import derive_batch_runtime_plan, load_training_config
from training.utils import get_init

HF_DATASET_TOKENS_ENV = "LLM_STUDIO_HF_DATASET_TOKENS"


class CancellationRequested(RuntimeError):
    pass


@dataclass(slots=True)
class TrainingRunArgs:
    job_id: str
    model_config_path: Path
    tokenizer_path: Path
    training_config_path: Path
    dataloader_config_path: Path
    output_dir: Path


@dataclass(slots=True)
class TrainingRunPaths:
    output_dir: Path
    metadata_path: Path
    state_path: Path
    stats_path: Path
    samples_path: Path
    checkpoints_dir: Path
    artifact_manifest_path: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "TrainingRunPaths":
        output_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            output_dir=output_dir,
            metadata_path=output_dir / "metadata.json",
            state_path=output_dir / "runtime_state.json",
            stats_path=output_dir / "stats.jsonl",
            samples_path=output_dir / "samples.jsonl",
            checkpoints_dir=output_dir / "checkpoints",
            artifact_manifest_path=output_dir / "artifact_manifest.json",
        )


class TrainingStateWriter:
    def __init__(self, job_id: str, paths: TrainingRunPaths) -> None:
        self.job_id = job_id
        self.paths = paths
        self._known_secrets = execution_known_secrets()
        now = utc_now()
        self.state: dict[str, Any] = {
            "job_id": job_id,
            "status": "pending",
            "state": "queued",
            "stage": "Queued",
            "progress": 0.0,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "last_step": 0,
            "max_steps": 0,
            "latest_loss": None,
            "latest_grad_norm": None,
            "latest_lr": None,
            "latest_tokens_per_sec": None,
            "checkpoint_count": 0,
            "sample_count": 0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "resolved_runtime": None,
            "memory_estimate": None,
            "error": None,
        }
        self.metadata: dict[str, Any] = {
            "job_id": job_id,
            "created_at": now,
            "status": "pending",
            "error": None,
        }
        self._write_state()
        self._write_metadata()

    def initialize_inputs(self, *, inputs: dict[str, Any]) -> None:
        self.metadata["inputs"] = inputs
        self._write_metadata()
        self._write_artifact_manifest()

    def mark_started(self) -> None:
        now = utc_now()
        self.state["started_at"] = now
        self.state["updated_at"] = now
        self.metadata["started_at"] = now
        self.metadata["status"] = "running"
        self.state["status"] = "running"
        self._write_state()
        self._write_metadata()

    def update(
        self,
        *,
        status: str | None = None,
        state: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
        error: str | None = None,
        **fields: Any,
    ) -> None:
        if status is not None:
            self.state["status"] = status
            self.metadata["status"] = status
        if state is not None:
            self.state["state"] = state
        if stage is not None:
            self.state["stage"] = stage
        if progress is not None:
            self.state["progress"] = max(0.0, min(float(progress), 1.0))
        if error is not None:
            sanitized_error = redact_execution_secrets(error, secrets=self._known_secrets)
            self.state["error"] = sanitized_error
            self.metadata["error"] = sanitized_error
        for key, value in fields.items():
            self.state[key] = value
        self.state["updated_at"] = utc_now()
        self._write_state()
        self._write_metadata()
        self._write_artifact_manifest()

    def record_runtime(self, payload: dict[str, Any]) -> None:
        self.state["resolved_runtime"] = payload
        self.metadata["resolved_runtime"] = payload
        self._write_state()
        self._write_metadata()

    def record_memory_estimate(self, payload: dict[str, Any]) -> None:
        self.state["memory_estimate"] = payload
        self.metadata["memory_estimate"] = payload
        self._write_state()
        self._write_metadata()

    def record_step(
        self,
        payload: dict[str, Any],
        *,
        max_steps: int,
        started_monotonic: float,
    ) -> None:
        step = int(payload["step"])
        completed_steps = step + 1
        elapsed_seconds = max(time.monotonic() - started_monotonic, 0.0)
        steps_per_second = (completed_steps / elapsed_seconds) if elapsed_seconds > 0 else 0.0
        remaining_steps = max(max_steps - completed_steps, 0)
        eta_seconds = (remaining_steps / steps_per_second) if steps_per_second > 0 else None
        self.update(
            status="running",
            state="training",
            stage=f"Training step {completed_steps:,} / {max_steps:,}",
            progress=phase_progress(completed_steps / max_steps, 0.24, 0.96),
            last_step=completed_steps,
            max_steps=max_steps,
            latest_loss=float(payload["loss"]),
            latest_grad_norm=float(payload["norm"]),
            latest_lr=float(payload["lr"]),
            latest_tokens_per_sec=float(payload["tok_per_sec"]),
            elapsed_seconds=elapsed_seconds,
            eta_seconds=eta_seconds,
        )

    def record_sample(self, payload: dict[str, Any]) -> None:
        self.update(sample_count=self.state["sample_count"] + 1, last_sample=payload)

    def record_checkpoint(self, payload: dict[str, Any]) -> None:
        self.update(
            checkpoint_count=self.state["checkpoint_count"] + 1,
            last_checkpoint=payload,
        )

    def finalize(self, *, status: str, state: str, stage: str, error: str | None = None) -> None:
        finished_at = utc_now()
        self.metadata["finished_at"] = finished_at
        self.state["finished_at"] = finished_at
        self.update(
            status=status,
            state=state,
            stage=stage,
            progress=1.0 if status in {"completed", "failed", "cancelled"} else self.state["progress"],
            error=error,
        )

    def _write_state(self) -> None:
        atomic_write_json(self.paths.state_path, self.state)

    def _write_metadata(self) -> None:
        atomic_write_json(self.paths.metadata_path, self.metadata)

    def _write_artifact_manifest(self) -> None:
        manifest = build_artifact_manifest(self.paths.output_dir, self.state)
        atomic_write_json(self.paths.artifact_manifest_path, manifest)


def run_training_job(
    args: TrainingRunArgs,
    *,
    writer: TrainingStateWriter | None = None,
) -> dict[str, Any]:
    paths = TrainingRunPaths.from_output_dir(args.output_dir)
    resolved_writer = writer or TrainingStateWriter(args.job_id, paths)
    if writer is None:
        resolved_writer.initialize_inputs(
            inputs={
                "model_config_path": str(args.model_config_path),
                "tokenizer_path": str(args.tokenizer_path),
                "training_config_path": str(args.training_config_path),
                "dataloader_config_path": str(args.dataloader_config_path),
                "output_dir": str(args.output_dir),
            }
        )
    install_signal_handlers()

    training_config = load_training_config(args.training_config_path)
    dataloader_config = load_training_dataloader_config(args.dataloader_config_path)
    model_config = load_config(args.model_config_path)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    eos_token_id = (
        tokenizer.token_to_id(dataloader_config.eos_token)
        if dataloader_config.add_eos and dataloader_config.eos_token is not None
        else None
    )

    resolved_writer.mark_started()
    resolved_writer.update(status="running", state="preflight", stage="Loading configs", progress=0.02)

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, autocast_ctx, synchronize = get_init()
    is_cuda = device_type == "cuda"
    master_process = ddp_rank == 0

    resolved_writer.update(
        status="running",
        state="initializing_model",
        stage="Initializing model",
        progress=0.08,
    )

    orig_model = ConfigurableGPT(model_config)
    orig_model = orig_model.to(device)
    compiled_model = maybe_compile_model(orig_model, is_cuda=is_cuda)

    if ddp:
        model = DDP(compiled_model, device_ids=[ddp_local_rank] if is_cuda else None)
    else:
        model = compiled_model

    optimizer = orig_model.setup_optimizer(
        lr=training_config.optimizer.lr,
        weight_decay=training_config.optimizer.weight_decay,
        betas=training_config.optimizer.betas,
        eps=training_config.optimizer.eps,
    )

    resolved_writer.update(
        status="running",
        state="estimating_memory",
        stage="Estimating memory",
        progress=0.12,
    )

    memory_estimator = MemoryEstimator(
        model=orig_model,
        optimizer=optimizer,
        device=device,
        token_dtype=dataloader_config.token_dtype,
    )
    memory_estimate = memory_estimator.estimate(
        seq_len=training_config.seq_len,
        batch_size=None,
    )

    batch_plan = derive_batch_runtime_plan(
        total_batch_size=training_config.total_batch_size,
        seq_len=training_config.seq_len,
        max_memory_batch_size=memory_estimate.max_batch_size,
        world_size=ddp_world_size,
        requested_micro_batch_size=training_config.micro_batch_size,
    )
    configured_memory_estimate = memory_estimator.estimate(
        seq_len=training_config.seq_len,
        batch_size=batch_plan.micro_batch_size,
    )

    runtime_summary = {
        "device": str(device),
        "device_type": device_type,
        "ddp": ddp,
        "ddp_rank": ddp_rank,
        "ddp_world_size": ddp_world_size,
        "micro_batch_size": batch_plan.micro_batch_size,
        "tokens_per_micro_step": batch_plan.tokens_per_micro_step,
        "tokens_per_world_step": batch_plan.tokens_per_world_step,
        "grad_accum_steps": batch_plan.grad_accum_steps,
        "max_batch_size_from_total": batch_plan.max_batch_size_from_total,
        "max_batch_size_from_memory": batch_plan.max_batch_size_from_memory,
        "max_allowed_batch_size": batch_plan.max_allowed_batch_size,
    }
    resolved_writer.record_runtime(runtime_summary)

    logger = Logger(
        stats_file_path=paths.stats_path,
        samples_file_path=paths.samples_path,
        on_step=lambda payload: resolved_writer.record_step(
            payload,
            max_steps=training_config.max_steps,
            started_monotonic=started_monotonic,
        ),
        on_sample=resolved_writer.record_sample,
        on_memory_estimate=resolved_writer.record_memory_estimate,
    )
    logger.memory_estimate(configured_memory_estimate)

    checkpoint_manager = CheckpointManager(paths.checkpoints_dir)

    resolved_writer.update(
        status="running",
        state="building_dataloader",
        stage="Building dataloader",
        progress=0.18,
    )

    train_loader = TrainingDataLoader(
        config=dataloader_config,
        tokenizer=tokenizer,
        batch_size=batch_plan.micro_batch_size,
        seq_len=training_config.seq_len,
    )

    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if model_config.vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"Model vocab_size ({model_config.vocab_size}) does not match tokenizer vocab_size ({tokenizer_vocab_size})."
        )

    scheduler = build_lr_scheduler(optimizer, training_config.lr_scheduler)
    generation_model = model.module if isinstance(model, DDP) else model

    started_monotonic = time.monotonic()
    resolved_writer.update(
        status="running",
        state="training",
        stage=f"Training step 0 / {training_config.max_steps:,}",
        progress=0.24,
        max_steps=training_config.max_steps,
    )

    for step in range(training_config.max_steps):
        raise_if_cancelled()
        synchronize()
        step_started = time.time()
        loss_accum = 0.0

        for micro_step in range(batch_plan.grad_accum_steps):
            raise_if_cancelled()
            x, y = train_loader.next_batch()
            x = x.to(device=device, non_blocking=is_cuda)
            y = y.to(device=device, non_blocking=is_cuda)

            synchronize_ctx = model.no_sync() if ddp and micro_step < batch_plan.grad_accum_steps - 1 else nullcontext()
            with synchronize_ctx:
                with autocast_ctx:
                    loss = model(x, y)

            loss = loss / batch_plan.grad_accum_steps
            loss_accum += float(loss.detach().item())
            loss.backward()

        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), 1.0)
        grad_norm = float(grad_norm_tensor.item())

        optimizer.step()
        adamw_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        model.zero_grad(set_to_none=True)

        synchronize()
        dt = max(time.time() - step_started, 1e-9)
        tok_per_sec = float(training_config.total_batch_size / dt)

        if master_process:
            logger.step(step, loss_accum, grad_norm, dt, tok_per_sec, adamw_lr)

            completed_steps = step + 1
            if 0 < completed_steps < training_config.max_steps and completed_steps % training_config.save_every == 0:
                resolved_writer.update(
                    status="running",
                    state="checkpointing",
                    stage=f"Saving checkpoint at step {completed_steps:,}",
                    progress=phase_progress(completed_steps / training_config.max_steps, 0.24, 0.96),
                )
                checkpoint = checkpoint_manager.save(
                    step=completed_steps,
                    model_data=orig_model.state_dict(),
                    optimizer_data=optimizer.state_dict(),
                    meta_data={
                        "step": completed_steps,
                        "batch_size": batch_plan.micro_batch_size,
                        "seq_len": training_config.seq_len,
                        "model_config": model_config.model_dump(mode="json"),
                        "runtime": runtime_summary,
                    },
                )
                resolved_writer.record_checkpoint(checkpoint)
                resolved_writer.update(
                    status="running",
                    state="training",
                    stage=f"Training step {completed_steps:,} / {training_config.max_steps:,}",
                    progress=phase_progress(completed_steps / training_config.max_steps, 0.24, 0.96),
                )

            if completed_steps > 0 and completed_steps % training_config.sample_every == 0:
                resolved_writer.update(
                    status="running",
                    state="sampling",
                    stage=f"Generating samples at step {completed_steps:,}",
                    progress=phase_progress(completed_steps / training_config.max_steps, 0.24, 0.96),
                )
                generation_model.eval()
                samples: list[str] = []
                prompts: list[str] = []

                for prompt_config in training_config.sampler.prompts:
                    prompts.append(prompt_config.prompt)
                    prompt_tokens = tokenizer.encode(prompt_config.prompt).ids

                    with autocast_ctx:
                        new_tokens = list(
                            generation_model.generate(
                                tokens=prompt_tokens,
                                max_tokens=prompt_config.max_tokens,
                                temperature=prompt_config.temperature,
                                top_k=prompt_config.top_k,
                                stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
                            )
                        )

                    full_tokens = prompt_tokens + new_tokens
                    decoded_tokens = tokenizer.decode(full_tokens, skip_special_tokens=False)
                    samples.append(decoded_tokens)

                logger.sample(completed_steps, samples, prompts=prompts)
                generation_model.train()
                resolved_writer.update(
                    status="running",
                    state="training",
                    stage=f"Training step {completed_steps:,} / {training_config.max_steps:,}",
                    progress=phase_progress(completed_steps / training_config.max_steps, 0.24, 0.96),
                )

    resolved_writer.update(status="running", state="finalizing", stage="Saving final checkpoint", progress=0.98)
    final_checkpoint = checkpoint_manager.save(
        step=training_config.max_steps,
        model_data=orig_model.state_dict(),
        optimizer_data=optimizer.state_dict(),
        meta_data={
            "step": training_config.max_steps,
            "batch_size": batch_plan.micro_batch_size,
            "seq_len": training_config.seq_len,
            "model_config": model_config.model_dump(mode="json"),
            "runtime": runtime_summary,
        },
    )
    resolved_writer.record_checkpoint(final_checkpoint)
    resolved_writer.finalize(status="completed", state="completed", stage="Completed")
    return {
        "status": "completed",
        "runtime": runtime_summary,
        "memory_estimate": configured_memory_estimate.to_dict(),
    }


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _cancel_signal_handler)
    signal.signal(signal.SIGINT, _cancel_signal_handler)


def maybe_compile_model(model: torch.nn.Module, *, is_cuda: bool) -> torch.nn.Module:
    if not is_cuda:
        return model

    if not torch_compile_enabled():
        print(
            "torch.compile disabled: set LLM_STUDIO_TORCH_COMPILE=1 to opt in; running training in eager mode.",
            flush=True,
        )
        return model

    compiler = find_c_compiler()
    if compiler is None:
        print("torch.compile disabled: no C compiler found on PATH; running training in eager mode.", flush=True)
        return model

    try:
        compiled = torch.compile(model, dynamic=False)
    except Exception as exc:
        error = redact_execution_secrets(f"{type(exc).__name__}: {exc}")
        print(f"torch.compile disabled: {error}; running training in eager mode.", flush=True)
        return model

    print(f"torch.compile enabled with C compiler: {compiler}", flush=True)
    return compiled


def torch_compile_enabled() -> bool:
    value = os.getenv("LLM_STUDIO_TORCH_COMPILE", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def find_c_compiler() -> str | None:
    configured = os.getenv("CC")
    if configured:
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        configured_path = Path(configured)
        if configured_path.exists():
            return str(configured_path)

    for candidate in ("cc", "gcc", "clang"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def phase_progress(fraction: float, progress_start: float, progress_end: float) -> float:
    clamped = max(0.0, min(float(fraction), 1.0))
    return progress_start + (progress_end - progress_start) * clamped


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execution_known_secrets() -> tuple[str, ...]:
    raw = os.getenv(HF_DATASET_TOKENS_ENV)
    if not raw:
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(payload, list):
        return ()
    return tuple(
        sorted(
            {
                item.strip()
                for item in payload
                if isinstance(item, str) and item.strip()
            },
            key=len,
            reverse=True,
        )
    )


def redact_execution_secrets(
    value: str,
    *,
    secrets: tuple[str, ...] | None = None,
) -> str:
    redacted = value
    for secret in secrets if secrets is not None else execution_known_secrets():
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def build_artifact_manifest(output_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_size = 0
    for candidate in sorted(output_dir.rglob("*")):
        if not candidate.is_file():
            continue
        relative_path = candidate.relative_to(output_dir)
        size_bytes = candidate.stat().st_size
        total_size += size_bytes
        files.append(
            {
                "path": str(relative_path),
                "size_bytes": size_bytes,
            }
        )
    return {
        "job_id": state.get("job_id"),
        "status": state.get("status"),
        "checkpoint_count": state.get("checkpoint_count", 0),
        "sample_count": state.get("sample_count", 0),
        "total_size_bytes": total_size,
        "files": files,
    }


def raise_if_cancelled() -> None:
    # The signal handler raises directly, but this helper makes cancellation checkpoints explicit.
    return None


def _cancel_signal_handler(signum: int, _frame: Any) -> None:
    raise CancellationRequested(f"Received signal {signum}")


def parse_args(argv: list[str] | None = None) -> TrainingRunArgs:
    parser = argparse.ArgumentParser(description="Run a job-scoped LLM training process.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--model-config-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--training-config-path", required=True)
    parser.add_argument("--dataloader-config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parsed = parser.parse_args(argv)
    return TrainingRunArgs(
        job_id=parsed.job_id,
        model_config_path=Path(parsed.model_config_path),
        tokenizer_path=Path(parsed.tokenizer_path),
        training_config_path=Path(parsed.training_config_path),
        dataloader_config_path=Path(parsed.dataloader_config_path),
        output_dir=Path(parsed.output_dir),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = TrainingRunPaths.from_output_dir(args.output_dir)
    writer = TrainingStateWriter(args.job_id, paths)
    writer.initialize_inputs(
        inputs={
            "model_config_path": str(args.model_config_path),
            "tokenizer_path": str(args.tokenizer_path),
            "training_config_path": str(args.training_config_path),
            "dataloader_config_path": str(args.dataloader_config_path),
            "output_dir": str(args.output_dir),
        }
    )
    try:
        run_training_job(args, writer=writer)
        return 0
    except CancellationRequested as exc:
        writer.finalize(status="cancelled", state="cancelled", stage="Cancelled", error=str(exc))
        return 2
    except Exception as exc:
        writer.finalize(
            status="failed",
            state="failed",
            stage="Failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        print(redact_execution_secrets(traceback.format_exc()))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
