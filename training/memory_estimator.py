from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from model.model import (
    CausalSelfAttention,
    ConfigurableGPT,
    ConfigurableMLP,
    LearnableRMSNorm,
    StaticRMSNorm,
)

_MIB = 1024 ** 2
_GIB = 1024 ** 3


@dataclass(slots=True)
class DeviceMemorySnapshot:
    device: torch.device
    free_bytes: int
    total_bytes: int
    allocated_bytes: int
    reserved_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StepMemoryBreakdown:
    model_parameter_bytes: int
    model_buffer_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    ddp_bucket_bytes: int
    rope_cache_bytes: int
    fixed_workspace_bytes: int
    fixed_misc_bytes: int
    input_target_bytes_per_token: int
    activation_bytes_per_token: int
    attention_bytes_per_token: int
    logits_bytes_per_token: int
    dynamic_misc_bytes_per_token: int

    @property
    def fixed_bytes(self) -> int:
        return (
            self.gradient_bytes
            + self.optimizer_state_bytes
            + self.ddp_bucket_bytes
            + self.rope_cache_bytes
            + self.fixed_workspace_bytes
            + self.fixed_misc_bytes
        )

    @property
    def dynamic_bytes_per_token(self) -> int:
        return (
            self.input_target_bytes_per_token
            + self.activation_bytes_per_token
            + self.attention_bytes_per_token
            + self.logits_bytes_per_token
            + self.dynamic_misc_bytes_per_token
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["fixed_bytes"] = self.fixed_bytes
        out["dynamic_bytes_per_token"] = self.dynamic_bytes_per_token
        return out


@dataclass(slots=True)
class StepSizeEstimate:
    device: torch.device
    seq_len: int
    attention_backend: str
    snapshot: DeviceMemorySnapshot
    breakdown: StepMemoryBreakdown
    safety_margin: float
    usable_free_bytes: int
    max_step_tokens: int
    max_batch_size: int
    max_total_batch_tokens: int
    estimated_peak_additional_bytes: int
    configured_step_tokens: int | None = None
    configured_peak_additional_bytes: int | None = None
    configured_fits: bool | None = None

    @property
    def model_resident_bytes(self) -> int:
        return self.breakdown.model_parameter_bytes + self.breakdown.model_buffer_bytes

    @property
    def estimated_peak_total_bytes(self) -> int:
        return self.model_resident_bytes + self.estimated_peak_additional_bytes

    @property
    def configured_peak_total_bytes(self) -> int | None:
        if self.configured_peak_additional_bytes is None:
            return None
        return self.model_resident_bytes + self.configured_peak_additional_bytes

    def to_dict(self) -> dict[str, Any]:
        out = {
            "device": str(self.device),
            "seq_len": self.seq_len,
            "attention_backend": self.attention_backend,
            "safety_margin": self.safety_margin,
            "usable_free_bytes": self.usable_free_bytes,
            "model_resident_bytes": self.model_resident_bytes,
            "max_step_tokens": self.max_step_tokens,
            "max_batch_size": self.max_batch_size,
            "max_total_batch_tokens": self.max_total_batch_tokens,
            "estimated_peak_additional_bytes": self.estimated_peak_additional_bytes,
            "estimated_peak_total_bytes": self.estimated_peak_total_bytes,
            "configured_step_tokens": self.configured_step_tokens,
            "configured_peak_additional_bytes": self.configured_peak_additional_bytes,
            "configured_peak_total_bytes": self.configured_peak_total_bytes,
            "configured_fits": self.configured_fits,
            "snapshot": self.snapshot.to_dict(),
            "breakdown": self.breakdown.to_dict(),
        }
        return out


class MemoryEstimator:
    """
    Simple class-based API for training memory estimation.

    Example:
        estimator = MemoryEstimator(
            model,
            optimizer=optimizer,
            device=device,
            gradient_accumulation_steps=grad_accum_steps,
            world_size=world_size,
        )
        estimate = estimator.estimate(seq_len=seq_len, batch_size=batch_size)
        print(estimator.format(estimate))
    """

    def __init__(
        self,
        model: ConfigurableGPT,
        *,
        optimizer: Optimizer | None = None,
        device: torch.device | str | None = None,
        gradient_accumulation_steps: int = 1,
        world_size: int = 1,
        attention_backend: Literal["auto", "flash", "math"] = "auto",
        safety_margin: float = 0.9,
        logits_backward_factor: float = 2.0,
        dynamic_overhead_factor: float = 0.15,
        fixed_workspace_bytes: int = 256 * _MIB,
        fixed_misc_bytes: int = 128 * _MIB,
        ddp_bucket_factor: float | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.world_size = world_size
        self.attention_backend = attention_backend
        self.safety_margin = safety_margin
        self.logits_backward_factor = logits_backward_factor
        self.dynamic_overhead_factor = dynamic_overhead_factor
        self.fixed_workspace_bytes = fixed_workspace_bytes
        self.fixed_misc_bytes = fixed_misc_bytes
        self.ddp_bucket_factor = ddp_bucket_factor

    def estimate(self, *, seq_len: int, batch_size: int | None = None) -> StepSizeEstimate:
        return estimate_max_step_tokens(
            model=self.model,
            optimizer=self.optimizer,
            seq_len=seq_len,
            batch_size=batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            world_size=self.world_size,
            device=self.device,
            attention_backend=self.attention_backend,
            safety_margin=self.safety_margin,
            logits_backward_factor=self.logits_backward_factor,
            dynamic_overhead_factor=self.dynamic_overhead_factor,
            fixed_workspace_bytes=self.fixed_workspace_bytes,
            fixed_misc_bytes=self.fixed_misc_bytes,
            ddp_bucket_factor=self.ddp_bucket_factor,
        )

    def report(self, *, seq_len: int, batch_size: int | None = None) -> str:
        return format_step_size_estimate(self.estimate(seq_len=seq_len, batch_size=batch_size))

    @staticmethod
    def format(estimate: StepSizeEstimate) -> str:
        return format_step_size_estimate(estimate)


def format_step_size_estimate(estimate: StepSizeEstimate) -> str:
    lines = [
        "Memory step-size estimate",
        f"  device: {estimate.device}",
        f"  attention backend: {estimate.attention_backend}",
        f"  free memory now: {_format_bytes(estimate.snapshot.free_bytes)} / {_format_bytes(estimate.snapshot.total_bytes)}",
        f"  usable free memory (margin {estimate.safety_margin:.0%}): {_format_bytes(estimate.usable_free_bytes)}",
        f"  model resident memory (params + buffers): {_format_bytes(estimate.model_resident_bytes)}",
        f"  fixed additional memory: {_format_bytes(estimate.breakdown.fixed_bytes)}",
        f"  dynamic memory per token: {_format_bytes(estimate.breakdown.dynamic_bytes_per_token)}",
        f"  max micro-step tokens (batch_size * seq_len): {estimate.max_step_tokens:,}",
        f"  max batch_size at seq_len={estimate.seq_len}: {estimate.max_batch_size:,}",
        f"  max total tokens per optimizer step (at current grad_accum * world_size): {estimate.max_total_batch_tokens:,}",
        f"  estimated peak total training memory at max step: {_format_bytes(estimate.estimated_peak_total_bytes)}",
    ]
    if estimate.configured_step_tokens is not None:
        state = "fits" if estimate.configured_fits else "exceeds"
        lines.append(
            f"  configured step ({estimate.configured_step_tokens:,} tokens) {state} estimate"
        )
        if estimate.configured_peak_additional_bytes is not None:
            lines.append(
                f"  configured peak additional memory: {_format_bytes(estimate.configured_peak_additional_bytes)}"
            )
        if estimate.configured_peak_total_bytes is not None:
            lines.append(
                f"  configured peak total training memory: {_format_bytes(estimate.configured_peak_total_bytes)}"
            )
    return "\n".join(lines)


def estimate_max_step_tokens(
    model: ConfigurableGPT,
    *,
    optimizer: Optimizer | None,
    seq_len: int,
    batch_size: int | None = None,
    gradient_accumulation_steps: int = 1,
    world_size: int = 1,
    device: torch.device | str | None = None,
    attention_backend: Literal["auto", "flash", "math"] = "auto",
    safety_margin: float = 0.9,
    logits_backward_factor: float = 2.0,
    dynamic_overhead_factor: float = 0.15,
    fixed_workspace_bytes: int = 256 * _MIB,
    fixed_misc_bytes: int = 128 * _MIB,
    ddp_bucket_factor: float | None = None,
) -> StepSizeEstimate:
    """
    Estimate the largest micro-step token count (batch_size * seq_len) that should fit in memory.

    The estimate combines:
    - Live free device memory
    - Fixed training-time allocations (grads, optimizer states, DDP buckets, rope cache, workspace)
    - Dynamic token-dependent allocations (inputs/targets, activations, attention bookkeeping, logits)

    Returned numbers are intentionally conservative and should be treated as upper bounds with safety margin.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if not (0 < safety_margin <= 1):
        raise ValueError("safety_margin must be in (0, 1]")
    if logits_backward_factor <= 0:
        raise ValueError("logits_backward_factor must be > 0")
    if dynamic_overhead_factor < 0:
        raise ValueError("dynamic_overhead_factor must be >= 0")
    if fixed_workspace_bytes < 0 or fixed_misc_bytes < 0:
        raise ValueError("fixed overhead bytes must be >= 0")
    if gradient_accumulation_steps <= 0 or world_size <= 0:
        raise ValueError("gradient_accumulation_steps and world_size must be > 0")
    if seq_len > int(model.config.context_length):
        raise ValueError(
            f"seq_len ({seq_len}) exceeds model context_length ({model.config.context_length})"
        )

    resolved_device = _resolve_device(model, device)
    snapshot = get_device_memory_snapshot(resolved_device)
    resolved_backend = _resolve_attention_backend(resolved_device, attention_backend)

    params = list(model.parameters())
    if not params:
        raise ValueError("model has no parameters")

    model_parameter_bytes = sum(p.numel() * p.element_size() for p in params)
    model_buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    trainable_params = [p for p in params if p.requires_grad]
    trainable_numel = sum(p.numel() for p in trainable_params)
    trainable_dtype_bytes = max((p.element_size() for p in trainable_params), default=4)

    gradient_bytes = trainable_numel * trainable_dtype_bytes
    optimizer_state_bytes = _estimate_optimizer_state_bytes(
        optimizer=optimizer,
        trainable_numel=trainable_numel,
        trainable_dtype_bytes=trainable_dtype_bytes,
    )

    if ddp_bucket_factor is None:
        ddp_bucket_factor = 1.0 if dist.is_available() and dist.is_initialized() else 0.0
    ddp_bucket_bytes = int(gradient_bytes * max(ddp_bucket_factor, 0.0))

    activation_dtype = _resolve_activation_dtype(resolved_device)
    activation_dtype_bytes = _dtype_bytes(activation_dtype)

    saved_activation_elements_per_token, attention_elements_per_token = _estimate_activation_elements_per_token(
        model=model,
        seq_len=seq_len,
        attention_backend=resolved_backend,
    )

    activation_bytes_per_token = int(math.ceil(saved_activation_elements_per_token * activation_dtype_bytes))
    attention_bytes_per_token = int(math.ceil(attention_elements_per_token * activation_dtype_bytes))

    input_target_bytes_per_token = 2 * _dtype_bytes(torch.long)

    vocab_size = int(model.config.vocab_size)
    logits_bytes_per_token = int(math.ceil(vocab_size * _dtype_bytes(torch.float32) * logits_backward_factor))

    core_dynamic_per_token = (
        input_target_bytes_per_token
        + activation_bytes_per_token
        + attention_bytes_per_token
        + logits_bytes_per_token
    )
    dynamic_misc_bytes_per_token = int(math.ceil(core_dynamic_per_token * dynamic_overhead_factor))

    rope_cache_bytes = _estimate_rope_cache_bytes(model, seq_len)

    breakdown = StepMemoryBreakdown(
        model_parameter_bytes=model_parameter_bytes,
        model_buffer_bytes=model_buffer_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        ddp_bucket_bytes=ddp_bucket_bytes,
        rope_cache_bytes=rope_cache_bytes,
        fixed_workspace_bytes=fixed_workspace_bytes,
        fixed_misc_bytes=fixed_misc_bytes,
        input_target_bytes_per_token=input_target_bytes_per_token,
        activation_bytes_per_token=activation_bytes_per_token,
        attention_bytes_per_token=attention_bytes_per_token,
        logits_bytes_per_token=logits_bytes_per_token,
        dynamic_misc_bytes_per_token=dynamic_misc_bytes_per_token,
    )

    usable_free_bytes = int(snapshot.free_bytes * safety_margin)
    dynamic_budget = usable_free_bytes - breakdown.fixed_bytes

    if dynamic_budget <= 0:
        max_step_tokens = 0
    else:
        max_step_tokens = dynamic_budget // max(1, breakdown.dynamic_bytes_per_token)

    max_batch_size = max_step_tokens // seq_len
    max_step_tokens = max_batch_size * seq_len

    estimated_peak_additional_bytes = breakdown.fixed_bytes + max_step_tokens * breakdown.dynamic_bytes_per_token
    max_total_batch_tokens = max_step_tokens * gradient_accumulation_steps * world_size

    configured_step_tokens = None
    configured_peak_additional_bytes = None
    configured_fits = None

    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        configured_step_tokens = batch_size * seq_len
        configured_peak_additional_bytes = (
            breakdown.fixed_bytes + configured_step_tokens * breakdown.dynamic_bytes_per_token
        )
        configured_fits = configured_peak_additional_bytes <= usable_free_bytes

    return StepSizeEstimate(
        device=resolved_device,
        seq_len=seq_len,
        attention_backend=resolved_backend,
        snapshot=snapshot,
        breakdown=breakdown,
        safety_margin=safety_margin,
        usable_free_bytes=usable_free_bytes,
        max_step_tokens=max_step_tokens,
        max_batch_size=max_batch_size,
        max_total_batch_tokens=max_total_batch_tokens,
        estimated_peak_additional_bytes=estimated_peak_additional_bytes,
        configured_step_tokens=configured_step_tokens,
        configured_peak_additional_bytes=configured_peak_additional_bytes,
        configured_fits=configured_fits,
    )


def get_device_memory_snapshot(device: torch.device | str | None = None) -> DeviceMemorySnapshot:
    resolved_device = torch.device(device) if device is not None else _infer_default_device()

    if resolved_device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(resolved_device)
        allocated_bytes = torch.cuda.memory_allocated(resolved_device)
        reserved_bytes = torch.cuda.memory_reserved(resolved_device)
        return DeviceMemorySnapshot(
            device=resolved_device,
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            allocated_bytes=int(allocated_bytes),
            reserved_bytes=int(reserved_bytes),
        )

    if resolved_device.type == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
        total_bytes = int(getattr(torch.mps, "recommended_max_memory", lambda: 0)())
        allocated_bytes = int(getattr(torch.mps, "current_allocated_memory", lambda: 0)())
        driver_bytes = int(getattr(torch.mps, "driver_allocated_memory", lambda: allocated_bytes)())
        free_bytes = max(total_bytes - driver_bytes, 0) if total_bytes > 0 else max(total_bytes - allocated_bytes, 0)
        return DeviceMemorySnapshot(
            device=resolved_device,
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            allocated_bytes=int(allocated_bytes),
            reserved_bytes=int(driver_bytes),
        )

    total_bytes, free_bytes = _cpu_memory_info()
    used_bytes = max(total_bytes - free_bytes, 0)
    return DeviceMemorySnapshot(
        device=resolved_device,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        allocated_bytes=used_bytes,
        reserved_bytes=used_bytes,
    )


def _resolve_device(model: nn.Module, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return _infer_default_device()


def _infer_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_attention_backend(
    device: torch.device,
    attention_backend: Literal["auto", "flash", "math"],
) -> str:
    if attention_backend != "auto":
        return attention_backend

    if device.type != "cuda" or not hasattr(torch.backends, "cuda"):
        return "math"

    flash_enabled = bool(getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)())
    mem_eff_enabled = bool(getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: False)())

    if flash_enabled or mem_eff_enabled:
        return "flash"

    return "math"


def _resolve_activation_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _estimate_optimizer_state_bytes(
    *,
    optimizer: Optimizer | None,
    trainable_numel: int,
    trainable_dtype_bytes: int,
) -> int:
    if optimizer is None:
        return 0

    existing_state_bytes = _optimizer_existing_state_bytes(optimizer)
    if existing_state_bytes > 0:
        return existing_state_bytes

    cls_name = optimizer.__class__.__name__.lower()

    if "adam" in cls_name:
        # exp_avg + exp_avg_sq; use fp32 floor for conservative estimates.
        state_dtype_bytes = max(trainable_dtype_bytes, 4)
        master_weights_bytes = trainable_numel * (4 if trainable_dtype_bytes < 4 else 0)
        return 2 * trainable_numel * state_dtype_bytes + master_weights_bytes

    if "sgd" in cls_name:
        has_momentum = any(group.get("momentum", 0.0) > 0.0 for group in optimizer.param_groups)
        if has_momentum:
            return trainable_numel * trainable_dtype_bytes
        return 0

    if "adagrad" in cls_name:
        return trainable_numel * max(trainable_dtype_bytes, 4)

    return 0


def _optimizer_existing_state_bytes(optimizer: Optimizer) -> int:
    state_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                state_bytes += value.numel() * value.element_size()
    return state_bytes


def _estimate_activation_elements_per_token(
    *,
    model: ConfigurableGPT,
    seq_len: int,
    attention_backend: str,
) -> tuple[float, float]:
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    n_embd = int(model.config.n_embd)

    saved_elements = 0.0
    attention_elements = 0.0

    # Embedding output and top-level norms retain hidden-state tensors for backward.
    saved_elements += float(n_embd)  # token embedding output
    saved_elements += float(n_embd)  # in_norm input

    for block in model.transformer.h:
        for layer in block.layer:
            if isinstance(layer, CausalSelfAttention):
                block_saved, block_attention = _estimate_attention_elements_per_token(
                    layer=layer,
                    seq_len=seq_len,
                    attention_backend=attention_backend,
                )
                saved_elements += block_saved
                attention_elements += block_attention
            elif isinstance(layer, ConfigurableMLP):
                saved_elements += _estimate_mlp_elements_per_token(layer=layer, input_dim=n_embd)
            elif isinstance(layer, (nn.LayerNorm, LearnableRMSNorm, StaticRMSNorm)):
                saved_elements += float(n_embd)
            else:
                # Unknown ops inside a residual block are treated as hidden-size activations.
                saved_elements += float(n_embd)

    saved_elements += float(n_embd)  # out_norm input

    return saved_elements, attention_elements


def _estimate_attention_elements_per_token(
    *,
    layer: CausalSelfAttention,
    seq_len: int,
    attention_backend: str,
) -> tuple[float, float]:
    dim = float(layer.n_embd)
    kv_ratio = float(layer.n_kv_head) / float(layer.n_head)

    # Inputs saved by q/k/v/proj linears.
    linear_inputs = 4.0 * dim

    # Saved intermediates from RoPE, q/k norm and SDPA backward inputs.
    q = dim
    k = dim * kv_ratio
    v = dim * kv_ratio
    rope_and_norm = (q + k) + (q + k)
    sdpa_inputs = q + k + v
    proj_input = dim

    saved_elements = linear_inputs + rope_and_norm + sdpa_inputs + proj_input

    heads = float(layer.n_head)
    if attention_backend == "math":
        # Full attention matrix + softmax bookkeeping: O(T^2) memory.
        attention_elements = 2.0 * heads * float(seq_len)
    else:
        # Flash / memory-efficient attention keeps O(T) stats.
        attention_elements = 4.0 * heads

    return saved_elements, attention_elements


def _estimate_mlp_elements_per_token(*, layer: ConfigurableMLP, input_dim: int) -> float:
    current_dim = float(input_dim)
    saved_elements = 0.0

    for sublayer in layer.layer:
        if isinstance(sublayer, nn.Linear):
            saved_elements += float(sublayer.in_features)
            current_dim = float(sublayer.out_features)
        elif isinstance(sublayer, (nn.LayerNorm, LearnableRMSNorm, StaticRMSNorm)):
            saved_elements += current_dim
        elif isinstance(
            sublayer,
            (
                nn.GELU,
                nn.ReLU,
                nn.SiLU,
                nn.Tanh,
                nn.Sigmoid,
                nn.LeakyReLU,
                nn.ELU,
                nn.Softplus,
            ),
        ):
            saved_elements += current_dim
        else:
            saved_elements += current_dim

    return saved_elements


def _estimate_rope_cache_bytes(model: ConfigurableGPT, seq_len: int) -> int:
    rope_dtype_bytes = _dtype_bytes(torch.bfloat16)
    total = 0

    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            total += seq_len * module.head_dim * rope_dtype_bytes

    return total


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _cpu_memory_info() -> tuple[int, int]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass

    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                meminfo = handle.read()
            total_kib = _parse_meminfo_kib(meminfo, "MemTotal")
            avail_kib = _parse_meminfo_kib(meminfo, "MemAvailable")
            return total_kib * 1024, avail_kib * 1024
        except Exception:
            pass

    if sys.platform == "darwin":
        try:
            vm_stat_output = subprocess.check_output(["vm_stat"], text=True)

            page_size_match = re.search(r"page size of (\d+) bytes", vm_stat_output)
            if not page_size_match:
                return 0, 0
            page_size = int(page_size_match.group(1))

            pages = _parse_vm_stat_pages(vm_stat_output)
            # On macOS, inactive + speculative are typically reclaimable.
            available_pages = (
                pages.get("Pages free", 0)
                + pages.get("Pages inactive", 0)
                + pages.get("Pages speculative", 0)
            )
            total_bytes = 0
            try:
                total_bytes = int(os.sysconf("SC_PHYS_PAGES")) * page_size
            except Exception:
                total_bytes = available_pages * page_size
            return total_bytes, available_pages * page_size
        except Exception:
            pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return total_pages * page_size, avail_pages * page_size
    except Exception:
        return 0, 0


def _parse_meminfo_kib(text: str, key: str) -> int:
    match = re.search(rf"^{re.escape(key)}:\s+(\d+)\s+kB$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"{key} not found in /proc/meminfo")
    return int(match.group(1))


def _parse_vm_stat_pages(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        num = "".join(ch for ch in value if ch.isdigit())
        if not num:
            continue
        out[key.strip()] = int(num)
    return out


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= _GIB:
        return f"{num_bytes / _GIB:.2f} GiB"
    return f"{num_bytes / _MIB:.2f} MiB"
