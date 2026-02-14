from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from datasets import load_dataset, load_from_disk

from training.dataloader_config import (
    DatasetSpec,
    MixingConfig,
    NormalizationConfig,
    ShuffleConfig,
    TrainingDataloaderConfig,
    load_training_dataloader_config,
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


def _get_column_names(dataset: Iterable) -> Optional[List[str]]:
    if hasattr(dataset, "column_names"):
        cols = getattr(dataset, "column_names")
        if cols:
            return list(cols)
    if hasattr(dataset, "features"):
        features = getattr(dataset, "features")
        if features:
            return list(features)
    return None


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "uint8": torch.uint8,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported token_dtype '{name}'")
    return mapping[name]


@dataclass
class _DatasetState:
    dataset: Iterable
    spec: DatasetSpec
    tokenized: bool
    index: int


class TrainingTokenDataset(IterableDataset):
    def __init__(
        self,
        config: TrainingDataloaderConfig,
        tokenizer,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if not hasattr(tokenizer, "encode_batch") or not hasattr(tokenizer, "token_to_id"):
            raise ValueError("tokenizer must expose encode_batch() and token_to_id()")
        self._epoch = multiprocessing.Value("i", 0)
        self._shuffle_seed_fallbacks: Optional[List[int]] = None
        self._init_shuffle_seed_fallbacks()
        self._bos_id = self._resolve_token_id(self.config.bos_token, "bos_token")
        self._eos_id = self._resolve_token_id(self.config.eos_token, "eos_token")
        self._pad_id = self._resolve_token_id(self.config.pad_token, "pad_token")
        if self.config.add_bos and self._bos_id is None:
            raise ValueError("bos_token was not found in tokenizer vocab")
        if self.config.add_eos and self._eos_id is None:
            raise ValueError("eos_token was not found in tokenizer vocab")
        if not self.config.drop_last and self._pad_id is None:
            if self._eos_id is not None:
                self._pad_id = self._eos_id
            else:
                raise ValueError("pad_token (or eos_token) was not found in tokenizer vocab")
        self._dtype = _resolve_dtype(self.config.token_dtype)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _current_epoch(self) -> int:
        return self._epoch.value

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        states = [
            self._load_dataset(idx, spec, worker_info)
            for idx, spec in enumerate(self.config.datasets)
        ]
        mixed_iter = self._interleave(states, self.config.mixing)
        yield from self._pack_tokens(mixed_iter)

    def _resolve_token_id(self, token: Optional[str], label: str) -> Optional[int]:
        if token is None:
            return None
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"{label} '{token}' not found in tokenizer vocab")
        return int(token_id)

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
            except TypeError:
                try:
                    return dataset.shuffle(seed=seed)
                except Exception:
                    pass
            except Exception:
                pass
        return _shuffle_iterable(dataset, shuffle_config.buffer_size, seed)

    def _load_dataset(
        self,
        dataset_index: int,
        spec: DatasetSpec,
        worker_info,
    ) -> _DatasetState:
        args = [spec.name]
        if spec.config:
            args.append(spec.config)
        kwargs = {
            "split": spec.split,
            "streaming": spec.streaming,
        }
        if spec.data_files is not None:
            kwargs["data_files"] = spec.data_files
        if spec.columns is not None:
            kwargs["columns"] = spec.columns
        if spec.filters is not None:
            kwargs["filters"] = [tuple(filt) for filt in spec.filters]
        if self.config.cache_dir is not None:
            kwargs["cache_dir"] = self.config.cache_dir
        dataset = load_dataset(*args, **kwargs)
        if hasattr(dataset, "features"):
            for column in spec.text_columns:
                if column not in dataset.features:
                    raise ValueError(
                        f"Dataset '{spec.name}' does not contain text column '{column}'"
                    )
        tokenized = False
        if not spec.streaming and self.config.cache_dir:
            cache_path = self._token_cache_path(spec, dataset_index)
            if cache_path.exists():
                dataset = load_from_disk(cache_path)
                tokenized = True
            else:
                tokenized_dataset = self._tokenize_dataset(dataset, spec)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tokenized_dataset.save_to_disk(cache_path)
                dataset = load_from_disk(cache_path)
                tokenized = True
        shuffle_config = self._resolve_shuffle_config(spec)
        if shuffle_config is not None:
            dataset = self._shuffle_dataset(dataset, shuffle_config, dataset_index)
        if self.config.node_split:
            dataset = self._split_by_node(dataset)
        if worker_info is not None:
            dataset = self._shard_dataset(dataset, worker_info.id, worker_info.num_workers)
        return _DatasetState(dataset=dataset, spec=spec, tokenized=tokenized, index=dataset_index)

    def _split_by_node(self, dataset: Iterable) -> Iterable:
        from datasets.distributed import split_dataset_by_node

        rank = self.config.node_rank
        world_size = self.config.node_world_size
        if rank is None or world_size is None:
            env_rank = os.getenv("RANK")
            env_world = os.getenv("WORLD_SIZE")
            if env_rank is not None and env_world is not None:
                rank = int(env_rank)
                world_size = int(env_world)
        if rank is None or world_size is None:
            return dataset
        return split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def _shard_dataset(self, dataset: Iterable, worker_id: int, num_workers: int) -> Iterable:
        if hasattr(dataset, "shard"):
            try:
                return dataset.shard(num_shards=num_workers, index=worker_id)
            except Exception:
                pass
        return _strided_iterable(dataset, worker_id, num_workers)

    def _token_cache_path(self, spec: DatasetSpec, dataset_index: int) -> Path:
        cache_root = Path(self.config.cache_dir)
        slug = _slugify(f"{dataset_index}-{spec.name}-{spec.config or 'default'}-{spec.split}")
        cache_key = self._token_cache_key(spec)
        return cache_root / f"tokenized-{slug}-{cache_key}"

    def _token_cache_key(self, spec: DatasetSpec) -> str:
        normalizer = spec.normalization or self.config.normalization
        separator = spec.record_separator
        if separator is None:
            separator = self.config.record_separator
        payload = {
            "spec": spec.model_dump(),
            "normalization": normalizer.model_dump(),
            "record_separator": separator,
            "tokenizer": self._tokenizer_cache_fingerprint(),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:12]

    def _tokenizer_cache_fingerprint(self) -> str:
        try:
            return hashlib.sha1(self.tokenizer.to_str().encode("utf-8")).hexdigest()
        except Exception:
            try:
                vocab = self.tokenizer.get_vocab()
                return hashlib.sha1(json.dumps(vocab, sort_keys=True).encode("utf-8")).hexdigest()
            except Exception:
                return str(getattr(self.tokenizer, "get_vocab_size", lambda: "unknown")())

    def _tokenize_dataset(self, dataset: Iterable, spec: DatasetSpec) -> Iterable:
        normalizer = TextNormalizer(spec.normalization or self.config.normalization)
        separator = spec.record_separator
        if separator is None:
            separator = self.config.record_separator
        remove_columns = _get_column_names(dataset)
        batch_size = self.config.pretokenize_batch_size

        def build_text_batch(examples: dict) -> dict:
            batch_len = 0
            for value in examples.values():
                batch_len = len(value)
                break
            if batch_len == 0:
                return {"text": []}
            texts: List[Optional[str]] = []
            for idx in range(batch_len):
                parts: List[str] = []
                for col in spec.text_columns:
                    if col not in examples:
                        continue
                    value = examples[col][idx]
                    if value is None:
                        continue
                    if isinstance(value, (list, tuple)):
                        parts.append(spec.text_joiner.join(str(item) for item in value))
                    else:
                        parts.append(str(value))
                if not parts:
                    texts.append(None)
                    continue
                if len(parts) == 1:
                    text = parts[0]
                else:
                    text = spec.text_joiner.join(parts)
                text = normalizer(text)
                if text is None:
                    texts.append(None)
                    continue
                if separator:
                    text = text + separator
                texts.append(text)
            return {"text": texts}

        def filter_empty(example: dict) -> bool:
            text = example.get("text")
            return text is not None and text != ""

        def tokenize_batch(examples: dict) -> dict:
            texts = examples.get("text", [])
            encodings = self.tokenizer.encode_batch(texts)
            return {"tokens": [encoding.ids for encoding in encodings]}

        map_kwargs = dict(batched=True, batch_size=batch_size)
        if remove_columns is not None:
            map_kwargs["remove_columns"] = remove_columns
        dataset = dataset.map(build_text_batch, **map_kwargs)
        dataset = dataset.filter(filter_empty)
        dataset = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=["text"],
        )
        return dataset

    def _iter_token_records(self, state: _DatasetState) -> Iterable[List[int]]:
        dataset = state.dataset
        if not state.tokenized:
            dataset = self._tokenize_dataset(dataset, state.spec)
        for example in dataset:
            tokens = example.get("tokens")
            if tokens is None:
                continue
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            else:
                tokens = list(tokens)
            if not tokens:
                continue
            if self.config.add_bos:
                tokens = [self._bos_id] + tokens
            if self.config.add_eos:
                tokens = tokens + [self._eos_id]
            yield tokens

    def _interleave(
        self,
        states: List[_DatasetState],
        mixing: MixingConfig,
    ) -> Iterable[List[int]]:
        active_states = list(states)
        iterators = [iter(self._iter_token_records(state)) for state in active_states]
        weights = [state.spec.weight for state in active_states]
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
                    iterators[idx] = iter(self._iter_token_records(active_states[idx]))
                    try:
                        yield next(iterators[idx])
                        continue
                    except StopIteration:
                        pass
                iterators.pop(idx)
                active_states.pop(idx)
                weights.pop(idx)
                if not iterators:
                    return
                sampler = SmoothWeightedRoundRobin(weights, mixing.seed)

    def _pack_tokens(
        self, token_iter: Iterable[List[int]]
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        seq_len = self.seq_len
        pad_id = self._pad_id
        buffer: List[int] = []
        for tokens in token_iter:
            if not tokens:
                continue
            buffer.extend(tokens)
            while len(buffer) >= seq_len + 1:
                window = buffer[: seq_len + 1]
                inputs = torch.tensor(window[:-1], dtype=self._dtype)
                targets = torch.tensor(window[1:], dtype=self._dtype)
                yield inputs, targets
                buffer = buffer[seq_len:]
        if buffer and not self.config.drop_last:
            if len(buffer) < seq_len + 1:
                pad_needed = seq_len + 1 - len(buffer)
                buffer = buffer + [pad_id] * pad_needed
            window = buffer[: seq_len + 1]
            inputs = torch.tensor(window[:-1], dtype=self._dtype)
            targets = torch.tensor(window[1:], dtype=self._dtype)
            yield inputs, targets


def _strided_iterable(dataset: Iterable, worker_id: int, num_workers: int) -> Iterable:
    for idx, sample in enumerate(dataset):
        if idx % num_workers == worker_id:
            yield sample


def build_training_dataset(
    config: TrainingDataloaderConfig | dict | str,
    tokenizer,
    seq_len: int,
) -> TrainingTokenDataset:
    if isinstance(config, TrainingDataloaderConfig):
        resolved = config
    elif isinstance(config, str):
        resolved = load_training_dataloader_config(config)
    else:
        resolved = TrainingDataloaderConfig.model_validate(config)
    return TrainingTokenDataset(resolved, tokenizer, seq_len=seq_len)


class TrainingDataLoader:
    def __init__(
        self,
        config: TrainingDataloaderConfig | dict | str,
        tokenizer,
        batch_size: int,
        seq_len: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: Optional[bool] = None,
        **kwargs,
    ) -> None:
        self.dataset = build_training_dataset(config, tokenizer, seq_len=seq_len)
        if persistent_workers is None:
            persistent_workers = num_workers > 0
        dataloader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **kwargs,
        )
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
        self.dataloader = DataLoader(self.dataset, **dataloader_kwargs)
        self.epoch = 0
        self.dataset.set_epoch(self.epoch)
        self._iterator = iter(self.dataloader)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(self.epoch)
        self._iterator = iter(self.dataloader)

    def reset(self) -> None:
        self.set_epoch(self.epoch + 1)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self._iterator)
        except StopIteration:
            self.reset()
            return next(self._iterator)

    def __iter__(self):
        return iter(self.dataloader)
