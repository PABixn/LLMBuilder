from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tokenizers import Tokenizer
from tokenizers.decoders import (
    ByteLevel as ByteLevelDecoder,
    Metaspace as MetaspaceDecoder,
    WordPiece as WordpieceDecoder,
)
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.pre_tokenizers import (
    ByteLevel as ByteLevelPreTokenizer,
    Metaspace as MetaspacePreTokenizer,
    Whitespace as WhitespacePreTokenizer,
)
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

from tokenizer.dataloader import build_dataset
from tokenizer.dataloader_config import DataloaderConfig
from tokenizer.loader import TokenizerConfig


@dataclass
class TokenizerStats:
    num_records: int = 0
    num_chars: int = 0
    num_tokens: int = 0
    token_per_char: float = 0.0
    chars_per_token: float = 0.0
    avg_chars_per_record: float = 0.0
    avg_tokens_per_record: float = 0.0
    vocab_size: int = 0
    num_used_tokens: int = 0
    num_unused_tokens: int = 0
    rare_tokens: Dict[int, int] = field(default_factory=dict)
    rare_token_fraction: Dict[int, float] = field(default_factory=dict)


def _normalize_thresholds(thresholds: Sequence[int]) -> list[int]:
    normalized = sorted(set(int(value) for value in thresholds))
    if not normalized:
        raise ValueError("thresholds must include at least one value")
    if any(value <= 0 for value in normalized):
        raise ValueError("thresholds values must all be positive")
    return normalized


def _iter_text_file(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line


class ConfigurableTokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config

        self.tokenizer = self.get_tokenizer()
        self.trainer = self.get_trainer()

    def train_from_file(self, filepaths: List[str]):
        self.tokenizer.train(filepaths, self.trainer)

        self.tokenizer.save(self.config.name + ".json")

    def train_from_dataset(self, dataloader_config: DataloaderConfig):
        dataset = build_dataset(dataloader_config)
        self.tokenizer.train_from_iterator(dataset, self.trainer)
        self.tokenizer.save(self.config.name + ".json")

    @staticmethod
    def eval_tokenizer_on_file(
        thresholds: Sequence[int],
        tokenizer: Tokenizer,
        text_path: str | Path = "datasets/shake.txt",
        batch_size: int = 64,
    ) -> TokenizerStats:
        path = Path(text_path)
        return ConfigurableTokenizer.eval_tokenizer_on_iterator(
            thresholds=thresholds,
            tokenizer=tokenizer,
            text_iter=_iter_text_file(path),
            batch_size=batch_size,
        )

    @staticmethod
    def eval_tokenizer_on_dataset(
        thresholds: Sequence[int],
        tokenizer: Tokenizer,
        dataloader_config: DataloaderConfig | dict | str,
        batch_size: int = 64,
    ) -> TokenizerStats:
        dataset = build_dataset(dataloader_config)
        return ConfigurableTokenizer.eval_tokenizer_on_iterator(
            thresholds=thresholds,
            tokenizer=tokenizer,
            text_iter=dataset,
            batch_size=batch_size,
        )

    @staticmethod
    def eval_tokenizer_on_iterator(
        thresholds: Sequence[int],
        tokenizer: Tokenizer,
        text_iter: Iterable[str],
        batch_size: int = 64,
    ) -> TokenizerStats:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        normalized_thresholds = _normalize_thresholds(thresholds)
        token_frequencies: Counter[int] = Counter()

        num_records = 0
        num_chars = 0
        num_tokens = 0
        batch: list[str] = []

        def flush_batch() -> None:
            nonlocal num_tokens
            if not batch:
                return
            encodings = tokenizer.encode_batch(batch)
            for encoding in encodings:
                ids = encoding.ids
                num_tokens += len(ids)
                token_frequencies.update(ids)
            batch.clear()

        for raw_text in text_iter:
            text = raw_text if isinstance(raw_text, str) else str(raw_text)
            num_records += 1
            num_chars += len(text)
            batch.append(text)
            if len(batch) >= batch_size:
                flush_batch()
        flush_batch()

        num_used_tokens = len(token_frequencies)
        vocab_size = tokenizer.get_vocab_size()

        rare_tokens: dict[int, int] = {}
        rare_token_fraction: dict[int, float] = {}
        for threshold in normalized_thresholds:
            rare_count = sum(
                1 for frequency in token_frequencies.values() if frequency < threshold
            )
            rare_tokens[threshold] = rare_count
            rare_token_fraction[threshold] = (
                rare_count / num_used_tokens if num_used_tokens > 0 else 0.0
            )

        token_per_char = (num_tokens / num_chars) if num_chars > 0 else 0.0
        chars_per_token = (num_chars / num_tokens) if num_tokens > 0 else 0.0
        avg_chars_per_record = (num_chars / num_records) if num_records > 0 else 0.0
        avg_tokens_per_record = (num_tokens / num_records) if num_records > 0 else 0.0

        return TokenizerStats(
            num_records=num_records,
            num_chars=num_chars,
            num_tokens=num_tokens,
            token_per_char=token_per_char,
            chars_per_token=chars_per_token,
            avg_chars_per_record=avg_chars_per_record,
            avg_tokens_per_record=avg_tokens_per_record,
            vocab_size=vocab_size,
            num_used_tokens=num_used_tokens,
            num_unused_tokens=max(vocab_size - num_used_tokens, 0),
            rare_tokens=rare_tokens,
            rare_token_fraction=rare_token_fraction,
        )

    def get_tokenizer(self):
        tok = None

        if self.config.tokenizer_type == "bpe":
            if self.config.byte_fallback:
                tok = Tokenizer(BPE(byte_fallback=True))
            else:
                tok = Tokenizer(BPE(byte_fallback=False, unk_token=self.config.unk_token))

        elif self.config.tokenizer_type == "wordpiece":
            tok = Tokenizer(WordPiece(unk_token=self.config.unk_token))

        elif self.config.tokenizer_type == "unigram":
            tok = Tokenizer(Unigram())
        else:
            raise ValueError(f"Unknown tokenizer type {self.config.tokenizer_type}")

        tok.pre_tokenizer = self._get_pre_tokenizer()
        tok.decoder = self._get_decoder()
        tok.add_special_tokens(self.config.special_tokens)

        return tok

    def get_trainer(self):
        if self.config.tokenizer_type == "bpe":
            return BpeTrainer(
                vocab_size=self.config.vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=self.config.special_tokens,
            )
        elif self.config.tokenizer_type == "wordpiece":
            return WordPieceTrainer(
                vocab_size=self.config.vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=self.config.special_tokens,
            )
        elif self.config.tokenizer_type == "unigram":
            return UnigramTrainer(
                vocab_size=self.config.vocab_size,
                special_tokens=self.config.special_tokens,
            )
        else:
            raise ValueError(f"Unknown trainer type {self.config.tokenizer_type}")

    def _get_pre_tokenizer(self):
        if self.config.pre_tokenizer == "byte_level":
            return ByteLevelPreTokenizer()
        elif self.config.pre_tokenizer == "whitespace":
            return WhitespacePreTokenizer()
        elif self.config.pre_tokenizer == "metaspace":
            return MetaspacePreTokenizer()
        else:
            raise ValueError(f"Unknown pre_tokenizer type {self.config.pre_tokenizer}")

    def _get_decoder(self):
        if self.config.decoder == "byte_level":
            return ByteLevelDecoder()
        elif self.config.decoder == "wordpiece":
            return WordpieceDecoder()
        elif self.config.decoder == "metaspace":
            return MetaspaceDecoder()
        else:
            raise ValueError(f"Unknown decoder type {self.config.decoder}")
