import json
from typing import List

from training.memory_estimator import StepSizeEstimate, MemoryEstimator


class Logger:
    def __init__(self, stats_file_path: str = "stats.jsonl", samples_file_path: str = "samples.jsonl"):
        self.stats_file_path = stats_file_path
        self.samples_file_path = samples_file_path

        pass

    def step(self, step: int, loss: float, norm: float, dt: float, tok_per_sec: float, lr: float):
        print(f"step: {step} | loss: {loss:.4f} | norm: {norm:.2f} | dt: {dt*1000:.2f} | tok/sec {tok_per_sec:.2f} | lr: {lr:.4e}")

        with open(self.stats_file_path, "a", encoding="utf-8") as f:
            stats = {"step": step, "loss": loss, "norm": norm, "dt": dt, "tok_per_sec": tok_per_sec, "lr": lr}
            f.write(json.dumps(stats) + "\n")
            f.flush()

    def sample(self, step: int, samples: List[str]):
        for idx, sample in enumerate(samples):
            print(f"Sample {idx}: {sample} \n \n")

        with open(self.samples_file_path, "a", encoding="utf-8") as f:
            texts = {"step": step, "samples": samples}
            f.write(json.dumps(texts) + "\n")
            f.flush()

    def memory_estimate(self, estimate: StepSizeEstimate):
        print(MemoryEstimator.format(estimate))