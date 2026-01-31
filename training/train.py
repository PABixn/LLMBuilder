from tokenizers.tokenizers import Tokenizer

from model.model import ConfigurableGPT
from training.dataloader import TrainingDataLoader
from training.dataloader_config import load_training_dataloader_config
from training.lr_scheduler import build_lr_scheduler
from training_config import load_training_config
from utils import get_init
from model.loader import load_config
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    config = load_training_config("training/training_config.json")

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, autocast_ctx, synchronize = get_init()

    batch_size = 32

    #Gradient accumulation
    tokens_per_pass = batch_size * config.seq_len
    world_tokens_per_pass = tokens_per_pass * ddp_world_size

    assert config.total_batch_size % world_tokens_per_pass == 0

    grad_accum_steps = config.total_batch_size // world_tokens_per_pass
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    #Tokenier
    tokenizer = Tokenizer.from_file("trained_tokenizer.json")

    #Initialize model
    model_config = load_config("model/gpt2_config.json")
    orig_model = ConfigurableGPT(model_config)

    compiled_model = torch.compile(orig_model, dynamic=False)

    if ddp:
        model = DDP(compiled_model, device_ids=[ddp_local_rank] if device_type == "cuda" else None)
    else:
        model = compiled_model

    #Optimizer
    optimizer = orig_model.setup_optimizer(lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay, betas=config.optimizer.betas, eps=config.optimizer.eps)

    #LR Schedulers
    scheduler = build_lr_scheduler(optimizer, config.lr_scheduler)

    #Dataloaders
    train_loader_config = load_training_dataloader_config("training/dataloader_config.json")
    train_loader = TrainingDataLoader(config=train_loader_config, tokenizer=tokenizer, batch_size=batch_size)



if __name__ == "__main__":
    main()



