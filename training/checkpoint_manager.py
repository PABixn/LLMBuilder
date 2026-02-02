import json
import os
import torch

class CheckpointManager:
    def __init__(self, checkpoints_dir: str = "checkpoints"):
        self.checkpoints_dir = checkpoints_dir

    def save(self, model_data, optimizer_data, meta_data, step):
        final_dir = os.path.join(self.checkpoints_dir, str(step))
        os.makedirs(final_dir, exist_ok=True)

        model_path = os.path.join(final_dir, f"model-{step}.pt")
        torch.save(model_data, model_path)

        meta_path = os.path.join(final_dir, f"meta-{step}.pt")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta_data, indent=4))

        if optimizer_data is not None:
            optimizer_path = os.path.join(final_dir, f"optimizer-{step}.pt")
            torch.save(optimizer_data, optimizer_path)
