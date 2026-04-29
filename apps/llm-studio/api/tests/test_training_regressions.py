from __future__ import annotations

import math
from pathlib import Path

import torch

from model.loader import LLMConfig
from model.model import CausalSelfAttention, ConfigurableGPT
from tokenizer.dataloader import build_dataset
from training.dataloader import build_training_dataset


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _CharTokenizer:
    def __init__(self) -> None:
        self._vocab = {"<|endoftext|>": 0}
        self.id_to_token = {0: "<|endoftext|>"}

    def token_to_id(self, token: str) -> int | None:
        return self._vocab.get(token)

    def encode_batch(self, texts: list[str]) -> list[_Encoding]:
        encodings: list[_Encoding] = []
        for text in texts:
            ids: list[int] = []
            for char in text:
                token_id = self._vocab.get(char)
                if token_id is None:
                    token_id = len(self._vocab)
                    self._vocab[char] = token_id
                    self.id_to_token[token_id] = char
                ids.append(token_id)
            encodings.append(_Encoding(ids))
        return encodings

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        chars: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token[int(token_id)]
            if skip_special_tokens and token == "<|endoftext|>":
                continue
            chars.append(token)
        return "".join(chars)


def _tiny_model_config(vocab_size: int) -> dict[str, object]:
    return {
        "context_length": 8,
        "vocab_size": vocab_size,
        "n_embd": 8,
        "weight_tying": True,
        "blocks": [
            {
                "components": [
                    {"norm": {"type": "layernorm"}},
                    {"attention": {"n_head": 2, "n_kv_head": 2}},
                    {"norm": {"type": "layernorm"}},
                    {
                        "mlp": {
                            "multiplier": 2,
                            "sequence": [
                                {"linear": {"bias": True}},
                                {"activation": {"type": "relu"}},
                                {"linear": {"bias": True}},
                            ],
                        }
                    },
                ]
            }
        ],
    }


def test_tokenizer_local_text_dataset_reads_each_file_as_one_document(tmp_path: Path) -> None:
    local_dataset = tmp_path / "shake.txt"
    content = "First line.\nSecond line.\n\nThird line.\n"
    local_dataset.write_text(content, encoding="utf-8")

    dataset = build_dataset(
        {
            "datasets": [
                {
                    "name": "text",
                    "data_files": {"train": str(local_dataset)},
                    "split": "train",
                    "text_columns": ["text"],
                    "weight": 1.0,
                }
            ],
            "budget": {"limit": 10_000, "unit": "chars", "behavior": "stop"},
            "mixing": {"seed": 42, "exhausted_policy": "stop"},
            "normalization": {
                "normalize_newlines": True,
                "collapse_whitespace": False,
                "strip": False,
                "lowercase": False,
                "drop_empty": True,
            },
            "record_separator": "",
        }
    )

    assert list(dataset) == [content]


def test_training_local_text_dataset_adds_eos_once_per_file(tmp_path: Path) -> None:
    local_dataset = tmp_path / "shake.txt"
    local_dataset.write_text("a\nb\nc", encoding="utf-8")
    tokenizer = _CharTokenizer()

    dataset = build_training_dataset(
        {
            "datasets": [
                {
                    "name": "text",
                    "data_files": {"train": str(local_dataset)},
                    "split": "train",
                    "text_columns": ["text"],
                    "weight": 1.0,
                    "streaming": True,
                }
            ],
            "add_eos": True,
            "eos_token": "<|endoftext|>",
            "drop_last": True,
            "mixing": {"seed": 42, "exhausted_policy": "stop"},
            "normalization": {
                "normalize_newlines": True,
                "collapse_whitespace": False,
                "strip": False,
                "lowercase": False,
                "drop_empty": True,
            },
            "record_separator": "",
        },
        tokenizer,
        seq_len=1,
    )

    windows = list(dataset)
    assert windows

    token_stream = [int(windows[0][0].item())]
    token_stream.extend(int(target.item()) for _, target in windows)

    eos_id = tokenizer.token_to_id("<|endoftext|>")
    assert eos_id is not None
    assert token_stream.count(eos_id) == 1
    assert token_stream[-1] == eos_id


def test_training_debug_preview_shows_local_text_record_and_windows(tmp_path: Path) -> None:
    local_dataset = tmp_path / "shake.txt"
    local_dataset.write_text("a\nb\nc", encoding="utf-8")
    tokenizer = _CharTokenizer()

    dataset = build_training_dataset(
        {
            "datasets": [
                {
                    "name": "text",
                    "data_files": {"train": str(local_dataset)},
                    "split": "train",
                    "text_columns": ["text"],
                    "weight": 1.0,
                    "streaming": True,
                }
            ],
            "add_eos": True,
            "eos_token": "<|endoftext|>",
            "drop_last": True,
            "mixing": {"seed": 42, "exhausted_policy": "stop"},
            "normalization": {
                "normalize_newlines": True,
                "collapse_whitespace": False,
                "strip": False,
                "lowercase": False,
                "drop_empty": True,
            },
            "record_separator": "",
        },
        tokenizer,
        seq_len=2,
    )

    preview = dataset.debug_preview(max_token_records=2, max_windows=2, preview_token_count=8)

    assert preview["eos_token_id"] == 0
    assert len(preview["datasets"]) == 1
    dataset_preview = preview["datasets"][0]
    assert dataset_preview["is_local_text"] is True
    assert dataset_preview["resolved_file_count"] == 1
    assert len(dataset_preview["token_records"]) == 1
    first_record = dataset_preview["token_records"][0]
    assert first_record["eos_count"] == 1
    assert first_record["decoded_head"] == "a\nb\nc<|endoftext|>"
    assert len(preview["packed_windows"]) == 2
    assert preview["packed_windows"][0]["input"]["decoded_head"] == "a\n"


def test_model_generate_stops_before_emitting_stop_token() -> None:
    model = ConfigurableGPT(LLMConfig.model_validate(_tiny_model_config(vocab_size=4)))
    emitted_tokens = iter([2, 0, 3])

    def fake_forward(idx, targets=None, kv_cache=None, loss_reduction="mean"):
        logits = torch.full((1, idx.size(1), 4), -1e9, dtype=torch.float32, device=idx.device)
        logits[0, -1, next(emitted_tokens)] = 1e9
        return logits

    model.forward = fake_forward  # type: ignore[method-assign]

    generated = list(
        model.generate(
            tokens=[1],
            max_tokens=3,
            temperature=0.0,
            top_k=None,
            stop_token_ids=[0],
        )
    )

    assert generated == [2]


def test_cuda_memory_snapshot_accepts_unindexed_cuda_device(monkeypatch) -> None:
    from training import memory_estimator

    calls: dict[str, torch.device] = {}

    def fake_mem_get_info(device: torch.device) -> tuple[int, int]:
        calls["mem_get_info_device"] = device
        return 10, 20

    def fake_memory_allocated(device: torch.device) -> int:
        calls["memory_allocated_device"] = device
        return 3

    def fake_memory_reserved(device: torch.device) -> int:
        calls["memory_reserved_device"] = device
        return 4

    monkeypatch.setattr(memory_estimator.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(memory_estimator.torch.cuda, "mem_get_info", fake_mem_get_info)
    monkeypatch.setattr(memory_estimator.torch.cuda, "memory_allocated", fake_memory_allocated)
    monkeypatch.setattr(memory_estimator.torch.cuda, "memory_reserved", fake_memory_reserved)

    snapshot = memory_estimator.get_device_memory_snapshot(torch.device("cuda"))

    assert snapshot.device == torch.device("cuda:0")
    assert snapshot.free_bytes == 10
    assert snapshot.total_bytes == 20
    assert snapshot.allocated_bytes == 3
    assert snapshot.reserved_bytes == 4
    assert calls == {
        "mem_get_info_device": torch.device("cuda:0"),
        "memory_allocated_device": torch.device("cuda:0"),
        "memory_reserved_device": torch.device("cuda:0"),
    }


def test_runner_skips_torch_compile_without_c_compiler(monkeypatch, capsys) -> None:
    from training import runner

    model = torch.nn.Linear(1, 1)

    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)

    def fail_compile(*_args, **_kwargs):
        raise AssertionError("torch.compile should not be called without a C compiler")

    monkeypatch.setattr(runner.torch, "compile", fail_compile)

    assert runner.maybe_compile_model(model, is_cuda=True) is model
    assert "no C compiler" in capsys.readouterr().out


def test_model_initial_loss_stays_near_log_vocab_for_tied_and_untied_heads() -> None:
    vocab_size = 2000
    idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    expected_loss = math.log(vocab_size)

    for weight_tying in (True, False):
        torch.manual_seed(0)
        model = ConfigurableGPT(
            LLMConfig.model_validate(
                {
                    "context_length": 16,
                    "vocab_size": vocab_size,
                    "n_embd": 64,
                    "weight_tying": weight_tying,
                    "blocks": [],
                }
            )
        )

        with torch.no_grad():
            loss = float(model(idx, targets))

        assert abs(loss - expected_loss) < 0.5


def test_block_preactivation_feeds_attention_branch_without_overwriting_residual() -> None:
    model = ConfigurableGPT(
        LLMConfig.model_validate(
            {
                "context_length": 4,
                "vocab_size": 16,
                "n_embd": 4,
                "weight_tying": False,
                "blocks": [
                    {
                        "components": [
                            {"activation": {"type": "tanh"}},
                            {"attention": {"n_head": 1, "n_kv_head": 1}},
                        ]
                    }
                ],
            }
        )
    )

    block = model.transformer.h[0]
    attention = next(layer for layer in block.layer if isinstance(layer, CausalSelfAttention))
    attention.forward = lambda x, kv_cache=None: x  # type: ignore[method-assign]

    x = torch.tensor([[[0.5, -0.25, 1.0, -1.5]]], dtype=torch.float32)
    out = block(x.clone(), None)
    expected = x + torch.tanh(x)

    assert torch.allclose(out, expected, atol=1e-6)
