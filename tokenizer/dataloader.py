from __future__ import annotations

import multiprocessing
import random
import re
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from datasets import load_dataset

from tokenizer.dataloader_config import (
    DataloaderConfig,
    DatasetSpec,
    MixingConfig,
    NormalizationConfig,
    ShuffleConfig,
    TextBudgetConfig,
)

_WHITESPACE_RE = re.compile(r"\s+")


class TextNormalizer:
    def __init__(self, config: NormalizationConfig) -> None:
        self.config = config

    def __call__(self, text: str) -> Optional[str]:
        if self.config.normalize_newlines:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if self.config.collapse_whitespace:
            text = _WHITESPACE_RE.sub(" ", text)
        if self.config.strip:
            text = text.strip()
        if self.config.lowercase:
            text = text.lower()
        if self.config.drop_empty and not text:
            return None
        return text


class TextBudget:
    def __init__(self, config: TextBudgetConfig) -> None:
        self.limit = config.limit
        self.unit = config.unit
        self.behavior = config.behavior
        self.used = 0

    def remaining(self) -> int:
        return max(self.limit - self.used, 0)

    def consume(self, text: str) -> Tuple[Optional[str], bool]:
        length = measure_text(text, self.unit)
        remaining = self.remaining()
        if remaining <= 0:
            return None, True
        if length <= remaining:
            self.used += length
            return text, False
        if self.behavior == "stop":
            return None, True
        truncated = truncate_text(text, self.unit, remaining)
        if truncated:
            self.used += measure_text(truncated, self.unit)
            return truncated, True
        return None, True


class SmoothWeightedRoundRobin:
    def __init__(self, weights: Sequence[float], seed: Optional[int]) -> None:
        if not weights:
            raise ValueError("weights must be a non-empty sequence")
        self.weights = [float(w) for w in weights]
        self.total = sum(self.weights)
        if self.total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.current = [0.0 for _ in self.weights]
        self.rng = random.Random(seed) if seed is not None else None
        if self.rng is not None:
            for idx in range(len(self.current)):
                self.current[idx] = self.rng.random() * self.total

    def next_index(self) -> int:
        best_idx = 0
        best_val = None
        for idx, weight in enumerate(self.weights):
            self.current[idx] += weight
            val = self.current[idx]
            if best_val is None or val > best_val:
                best_val = val
                best_idx = idx
            elif val == best_val and self.rng is not None:
                if self.rng.random() < 0.5:
                    best_idx = idx
        self.current[best_idx] -= self.total
        return best_idx


def measure_text(text: str, unit: str) -> int:
    if unit == "chars":
        return len(text)
    if unit == "bytes":
        return len(text.encode("utf-8"))
    raise ValueError(f"Unknown text unit: {unit}")


def truncate_text(text: str, unit: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if unit == "chars":
        return text[:max_len]
    if unit == "bytes":
        data = text.encode("utf-8")[:max_len]
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unknown text unit: {unit}")


def split_text(text: str, unit: str, chunk_size: int) -> Iterable[str]:
    if chunk_size <= 0:
        return
    if unit == "chars":
        for idx in range(0, len(text), chunk_size):
            yield text[idx : idx + chunk_size]
        return
    if unit == "bytes":
        data = text.encode("utf-8")
        offset = 0
        while offset < len(data):
            window = data[offset : offset + chunk_size]
            prefix = window.decode("utf-8", errors="ignore")
            if not prefix:
                offset += 1
                continue
            consumed = len(prefix.encode("utf-8"))
            yield prefix
            offset += consumed
        return
    raise ValueError(f"Unknown text unit: {unit}")


def chunk_stream(
    text_iter: Iterable[str],
    chunk_size: int,
    unit: str,
    drop_last: bool,
) -> Iterable[str]:
    buffer = ""
    buffer_len = 0
    for text in text_iter:
        if not text:
            continue
        text_len = measure_text(text, unit)
        if buffer_len == 0 and text_len >= chunk_size:
            for part in split_text(text, unit, chunk_size):
                part_len = measure_text(part, unit)
                if part_len == chunk_size:
                    yield part
                elif part_len > 0:
                    buffer = part
                    buffer_len = part_len
            continue
        if buffer_len + text_len <= chunk_size:
            buffer += text
            buffer_len += text_len
            continue
        if buffer_len > 0:
            yield buffer
            buffer = ""
            buffer_len = 0
        if text_len >= chunk_size:
            for part in split_text(text, unit, chunk_size):
                part_len = measure_text(part, unit)
                if part_len == chunk_size:
                    yield part
                elif part_len > 0:
                    buffer = part
                    buffer_len = part_len
        else:
            buffer = text
            buffer_len = text_len
    if buffer_len > 0 and not drop_last:
        yield buffer


def _shuffle_iterable(
    iterable: Iterable,
    buffer_size: int,
    seed: Optional[int],
) -> Iterable:
    if buffer_size <= 1:
        yield from iterable
        return
    rng = random.Random(seed)
    iterator = iter(iterable)
    buffer: List = []
    for _ in range(buffer_size):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            rng.shuffle(buffer)
            yield from buffer
            return
    for item in iterator:
        idx = rng.randrange(buffer_size)
        yield buffer[idx]
        buffer[idx] = item
    rng.shuffle(buffer)
    yield from buffer


def extract_text(example: dict, columns: Sequence[str], joiner: str) -> Optional[str]:
    parts: List[str] = []
    for col in columns:
        if col not in example:
            continue
        value = example[col]
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            parts.append(joiner.join(str(item) for item in value))
        else:
            parts.append(str(value))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return joiner.join(parts)


class StreamingTextDataset(IterableDataset):
    def __init__(self, config: DataloaderConfig) -> None:
        super().__init__()
        self.config = config
        self._epoch = multiprocessing.Value("i", 0)
        self._shuffle_seed_fallbacks: Optional[List[int]] = None
        self._init_shuffle_seed_fallbacks()

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _current_epoch(self) -> int:
        return self._epoch.value

    def __iter__(self) -> Iterator[str]:
        worker_info = get_worker_info()
        datasets = [
            self._load_dataset(idx, spec, worker_info)
            for idx, spec in enumerate(self.config.datasets)
        ]
        mixed_iter = self._interleave(datasets, self.config.datasets, self.config.mixing)
        if self.config.chunking is not None:
            mixed_iter = chunk_stream(
                mixed_iter,
                self.config.chunking.chunk_size,
                self.config.budget.unit,
                self.config.chunking.drop_last,
            )
        budget = TextBudget(self.config.budget)
        for text in mixed_iter:
            output, stop = budget.consume(text)
            if output:
                yield output
            if stop:
                break

    def _resolve_shuffle_config(self, spec: DatasetSpec) -> Optional[ShuffleConfig]:
        if spec.shuffle is False:
            return None
        if spec.shuffle is not None:
            return spec.shuffle
        if self.config.shuffle is False:
            return None
        return self.config.shuffle

    def _init_shuffle_seed_fallbacks(self) -> None:
        if self._shuffle_seed_fallbacks is not None:
            return
        needs_fallback = False
        for spec in self.config.datasets:
            shuffle_config = self._resolve_shuffle_config(spec)
            if shuffle_config is not None and shuffle_config.seed is None:
                needs_fallback = True
                break
        if not needs_fallback:
            return
        rng = random.SystemRandom()
        base_seed = rng.randrange(2**32)
        rng = random.Random(base_seed)
        self._shuffle_seed_fallbacks = [
            rng.randrange(2**32) for _ in self.config.datasets
        ]

    def _effective_shuffle_seed(self, seed: Optional[int], dataset_index: int) -> Optional[int]:
        if seed is None:
            if self._shuffle_seed_fallbacks is None:
                return None
            seed = self._shuffle_seed_fallbacks[dataset_index]
        if seed is None:
            return None
        return seed + self._current_epoch()

    def _shuffle_dataset(
        self,
        dataset: Iterable,
        shuffle_config: ShuffleConfig,
        dataset_index: int,
    ) -> Iterable:
        seed = self._effective_shuffle_seed(shuffle_config.seed, dataset_index)
        if hasattr(dataset, "shuffle"):
            try:
                return dataset.shuffle(seed=seed, buffer_size=shuffle_config.buffer_size)
            except Exception:
                pass
        return _shuffle_iterable(dataset, shuffle_config.buffer_size, seed)

    def _load_dataset(self, dataset_index: int, spec: DatasetSpec, worker_info) -> Iterable:
        args = [spec.name]
        if spec.config:
            args.append(spec.config)
        kwargs = {
            "split": spec.split,
            "streaming": True,
        }
        if spec.data_files is not None:
            kwargs["data_files"] = spec.data_files
        if spec.columns is not None:
            kwargs["columns"] = spec.columns
        if spec.filters is not None:
            kwargs["filters"] = [tuple(filt) for filt in spec.filters]
        dataset = load_dataset(*args, **kwargs)
        if hasattr(dataset, "features"):
            for column in spec.text_columns:
                if column not in dataset.features:
                    raise ValueError(
                        f"Dataset '{spec.name}' does not contain text column '{column}'"
                    )
        shuffle_config = self._resolve_shuffle_config(spec)
        if shuffle_config is not None:
            dataset = self._shuffle_dataset(dataset, shuffle_config, dataset_index)
        if worker_info is not None:
            dataset = self._shard_dataset(dataset, worker_info.id, worker_info.num_workers)
        return dataset

    def _shard_dataset(self, dataset: Iterable, worker_id: int, num_workers: int) -> Iterable:
        if hasattr(dataset, "shard"):
            try:
                return dataset.shard(num_shards=num_workers, index=worker_id)
            except Exception:
                pass
        return _strided_iterable(dataset, worker_id, num_workers)

    def _iter_text_records(self, dataset: Iterable, spec: DatasetSpec) -> Iterable[str]:
        normalizer = TextNormalizer(spec.normalization or self.config.normalization)
        separator = spec.record_separator
        if separator is None:
            separator = self.config.record_separator
        for example in dataset:
            text = extract_text(example, spec.text_columns, spec.text_joiner)
            if text is None:
                continue
            text = normalizer(text)
            if text is None:
                continue
            if separator:
                text = text + separator
            yield text

    def _interleave(
        self,
        datasets: List[Iterable],
        specs: List[DatasetSpec],
        mixing: MixingConfig,
    ) -> Iterable[str]:
        active_datasets = list(datasets)
        active_specs = list(specs)
        iterators = [
            iter(self._iter_text_records(ds, spec))
            for ds, spec in zip(active_datasets, active_specs)
        ]
        weights = [spec.weight for spec in active_specs]
        sampler = SmoothWeightedRoundRobin(weights, mixing.seed)
        while iterators:
            idx = sampler.next_index()
            try:
                yield next(iterators[idx])
                continue
            except StopIteration:
                if mixing.exhausted_policy == "stop":
                    return
                if mixing.exhausted_policy == "repeat":
                    iterators[idx] = iter(
                        self._iter_text_records(active_datasets[idx], active_specs[idx])
                    )
                    try:
                        yield next(iterators[idx])
                        continue
                    except StopIteration:
                        pass
                iterators.pop(idx)
                active_datasets.pop(idx)
                active_specs.pop(idx)
                weights.pop(idx)
                if not iterators:
                    return
                sampler = SmoothWeightedRoundRobin(weights, mixing.seed)


def _strided_iterable(dataset: Iterable, worker_id: int, num_workers: int) -> Iterable:
    for idx, sample in enumerate(dataset):
        if idx % num_workers == worker_id:
            yield sample


def build_dataset(config: DataloaderConfig | dict | str) -> StreamingTextDataset:
    if isinstance(config, DataloaderConfig):
        resolved = config
    elif isinstance(config, str):
        from dataloader_config import load_dataloader_config

        resolved = load_dataloader_config(config)
    else:
        resolved = DataloaderConfig.model_validate(config)
    return StreamingTextDataset(resolved)


def create_tokenizer_dataloader(
    config: DataloaderConfig | dict | str,
    batch_size: Optional[int] = 1,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: Optional[bool] = None,
    **kwargs,
) -> DataLoader:
    dataset = build_dataset(config)
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    dataloader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        **kwargs,
    )
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **dataloader_kwargs)
