from contextlib import nullcontext

import torch
import torch.distributed as dist
import os

def get_init():
    device_type = autodetect_device_type()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, autocast_ctx, synchronize

def autodetect_device_type():
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    print(f"Detected device type: {device_type}")

    return device_type

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