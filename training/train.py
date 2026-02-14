import time
from contextlib import nullcontext

from tokenizers.tokenizers import Tokenizer

from model.model import ConfigurableGPT
from training.checkpoint_manager import CheckpointManager
from training.dataloader import TrainingDataLoader
from training.dataloader_config import load_training_dataloader_config
from training.logger import Logger
from training.lr_scheduler import build_lr_scheduler
from training.training_config import load_training_config
from training.utils import get_init
from model.loader import load_config
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    config = load_training_config("training/training_config.json")

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, autocast_ctx, synchronize = get_init()

    master_process = ddp_rank == 0

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
    orig_model = orig_model.to(device)

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
    train_loader = TrainingDataLoader(
        config=train_loader_config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=config.seq_len,
    )

    #Logger and checkpoing manager
    logger = Logger(stats_file_path="stats.jsonl")
    checkpoint_manager = CheckpointManager()

    #Checks
    assert model_config.vocab_size == tokenizer.get_vocab_size()

    for step in range(config.max_steps):
        synchronize()
        t0 = time.time()

        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            synchronize_ctx = model.no_sync() if ddp and micro_step < grad_accum_steps - 1 else nullcontext()
            with synchronize_ctx:
                with autocast_ctx:
                    loss = model(x, y)

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), 1.0)
        grad_norm = grad_norm_tensor.item()

        optimizer.step()

        adamw_lr = optimizer.param_groups[0]['lr']

        scheduler.step()

        model.zero_grad(set_to_none=True)

        synchronize()
        t1 = time.time()
        dt = t1 - t0

        tok_per_sec = int(config.total_batch_size / dt)

        if master_process:
            #Log step
            logger.step(step, loss_accum.item(), grad_norm, dt, tok_per_sec, adamw_lr)

            if 0 < step < config.max_steps - 1 and step % config.save_every == 0:
                checkpoint_manager.save(
                    step=step,
                    model_data=orig_model.state_dict(),
                    optimizer_data=optimizer.state_dict(),
                    meta_data={
                        "step": step,
                        "batch_size": batch_size,
                        "seq_len": config.seq_len,
                        "model_config": model_config.model_dump_json()
                    })

            #Sample from the model
            if step > 0 and step % config.sample_every == 0:
                model.eval()

                samples = []

                for idx, obj in enumerate(config.sampler.prompts):
                    prompt_tokens = tokenizer.encode(obj.prompt).ids

                    with autocast_ctx:
                        new_tokens = list(
                            model.generate(
                                tokens=prompt_tokens,
                                max_tokens=obj.max_tokens,
                                temperature=obj.temperature,
                                top_k=obj.top_k,
                            )
                        )

                    full_tokens = prompt_tokens + new_tokens
                    decoded_tokens = tokenizer.decode(full_tokens, skip_special_tokens=False)
                    samples.append(decoded_tokens)

                logger.sample(step, samples)

                model.train()

    checkpoint_manager.save(
        step=config.max_steps,
        model_data=orig_model.state_dict(),
        optimizer_data=optimizer.state_dict(),
        meta_data={
            "step": config.max_steps,
            "batch_size": batch_size,
            "seq_len": config.seq_len,
            "model_config": model_config.model_dump_json()
        })

if __name__ == "__main__":
    main()

