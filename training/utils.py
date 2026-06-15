import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

TRAINING_DEVICE_ENV = "LLM_STUDIO_TRAINING_DEVICE"
INFERENCE_DEVICE_ENV = "LLM_STUDIO_INFERENCE_DEVICE"
SUPPORTED_DEVICE_TYPES = {"cpu", "cuda", "mps"}


def get_init():
    device_type = autodetect_device_type()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

    return (
        ddp,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        device,
        device_type,
        autocast_ctx,
        synchronize,
    )


def autodetect_device_type():
    device_type = resolve_training_device_type()
    print(f"Detected device type: {device_type}")
    return device_type


def resolve_training_device_type() -> str:
    return _resolve_device_type(TRAINING_DEVICE_ENV)


def resolve_inference_device_type() -> str:
    return _resolve_device_type(INFERENCE_DEVICE_ENV)


def _resolve_device_type(override_env: str) -> str:
    requested = os.environ.get(override_env, "").strip().lower()
    if requested:
        if requested not in SUPPORTED_DEVICE_TYPES:
            supported = ", ".join(sorted(SUPPORTED_DEVICE_TYPES))
            raise RuntimeError(
                f"{override_env} must be one of: {supported}; got {requested!r}."
            )
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"{override_env}=cuda was requested, but CUDA is unavailable."
            )
        if requested == "mps" and not _mps_is_available():
            raise RuntimeError(
                f"{override_env}=mps was requested, but MPS is unavailable."
            )
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if _mps_is_available():
        return "mps"
    return "cpu"


def _mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])

        ddp_rank = int(os.environ.get("RANK"))
        ddp_local_rank = int(os.environ.get("LOCAL_RANK"))
        ddp_world_size = int(os.environ.get("WORLD_SIZE"))

        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def compute_init(device_type="cuda"):
    if device_type == "cuda":
        torch.set_float32_matmul_precision("medium")

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device


def is_ddp():
    return int(os.environ.get("RANK", -1)) != -1
