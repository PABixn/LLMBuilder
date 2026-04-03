from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from training.memory_estimator import MemoryEstimator, StepSizeEstimate

StepCallback = Callable[[dict[str, Any]], None]
SampleCallback = Callable[[dict[str, Any]], None]
MemoryEstimateCallback = Callable[[dict[str, Any]], None]


class Logger:
    def __init__(
        self,
        *,
        stats_file_path: str | Path = "stats.jsonl",
        samples_file_path: str | Path = "samples.jsonl",
        on_step: StepCallback | None = None,
        on_sample: SampleCallback | None = None,
        on_memory_estimate: MemoryEstimateCallback | None = None,
    ) -> None:
        self.stats_file_path = Path(stats_file_path)
        self.samples_file_path = Path(samples_file_path)
        self.on_step = on_step
        self.on_sample = on_sample
        self.on_memory_estimate = on_memory_estimate

        self.stats_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.samples_file_path.parent.mkdir(parents=True, exist_ok=True)

    def step(
        self,
        step: int,
        loss: float,
        norm: float,
        dt: float,
        tok_per_sec: float,
        lr: float,
    ) -> dict[str, Any]:
        payload = {
            "step": int(step),
            "loss": float(loss),
            "norm": float(norm),
            "dt": float(dt),
            "tok_per_sec": float(tok_per_sec),
            "lr": float(lr),
        }

        print(
            "step: "
            f"{payload['step']} | loss: {payload['loss']:.4f} | norm: {payload['norm']:.2f} "
            f"| dt: {payload['dt'] * 1000:.2f} | tok/sec {payload['tok_per_sec']:.2f} "
            f"| lr: {payload['lr']:.4e}"
        )

        self._append_jsonl(self.stats_file_path, payload)
        if self.on_step is not None:
            self.on_step(payload)
        return payload

    def sample(
        self,
        step: int,
        samples: Sequence[str],
        *,
        prompts: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        entries = []
        for idx, sample in enumerate(samples):
            prompt = prompts[idx] if prompts is not None and idx < len(prompts) else None
            print(f"Sample {idx}: {sample}\n")
            entries.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "text": sample,
                }
            )

        payload = {
            "step": int(step),
            "samples": entries,
        }

        self._append_jsonl(self.samples_file_path, payload)
        if self.on_sample is not None:
            self.on_sample(payload)
        return payload

    def memory_estimate(self, estimate: StepSizeEstimate) -> dict[str, Any]:
        payload = estimate.to_dict()
        print(MemoryEstimator.format(estimate))
        if self.on_memory_estimate is not None:
            self.on_memory_estimate(payload)
        return payload

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
            handle.flush()
