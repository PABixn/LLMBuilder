from __future__ import annotations

from pathlib import Path

from training.runner import TrainingRunArgs, run_training_job


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "training"
    args = TrainingRunArgs(
        job_id="manual-training-run",
        model_config_path=repo_root / "model" / "gpt2_config.json",
        tokenizer_path=repo_root / "trained_tokenizer.json",
        training_config_path=repo_root / "training" / "training_config.json",
        dataloader_config_path=repo_root / "training" / "dataloader_config.json",
        output_dir=output_dir,
    )
    run_training_job(args)


if __name__ == "__main__":
    main()
