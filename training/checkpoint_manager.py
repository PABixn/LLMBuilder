from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, checkpoints_dir: str | Path) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        model_data: Any,
        optimizer_data: Any,
        meta_data: dict[str, Any],
        step: int,
    ) -> dict[str, Any]:
        final_dir = self.checkpoints_dir / str(step)
        final_dir.mkdir(parents=True, exist_ok=True)

        model_path = final_dir / f"model-{step}.pt"
        torch.save(model_data, model_path)

        meta_path = final_dir / f"meta-{step}.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta_data, handle, indent=2)
            handle.write("\n")

        optimizer_path: Path | None = None
        if optimizer_data is not None:
            optimizer_path = final_dir / f"optimizer-{step}.pt"
            torch.save(optimizer_data, optimizer_path)

        files = [
            str(model_path),
            str(meta_path),
        ]
        if optimizer_path is not None:
            files.append(str(optimizer_path))

        return {
            "step": int(step),
            "directory": str(final_dir),
            "files": files,
        }
