from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import model.model as model_module

from model.loader import Attention, LLMConfig
from model.model import CausalSelfAttention, ConfigurableGPT
from tokenizer.dataloader import build_dataset
from training.dataloader import build_training_dataset
from training.dataloader_config import load_training_dataloader_config
from training import memory_estimator
from training import utils as training_utils


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


def test_cpu_memory_info_uses_native_windows_fallback_without_psutil(monkeypatch) -> None:
    monkeypatch.setattr(
        memory_estimator,
        "_psutil_memory_info",
        lambda: (_ for _ in ()).throw(ImportError("psutil unavailable")),
    )
    monkeypatch.setattr(memory_estimator.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(memory_estimator.sys, "platform", "win32")
    monkeypatch.setattr(memory_estimator, "_windows_memory_info", lambda: (16_000, 8_000))

    assert memory_estimator._cpu_memory_info() == (16_000, 8_000)


def test_local_compute_device_overrides_are_shared_and_validated(monkeypatch) -> None:
    monkeypatch.setattr(training_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv(training_utils.TRAINING_DEVICE_ENV, "cpu")
    assert training_utils.resolve_training_device_type() == "cpu"
    monkeypatch.setenv(training_utils.INFERENCE_DEVICE_ENV, "cpu")
    assert training_utils.resolve_inference_device_type() == "cpu"

    monkeypatch.setenv(training_utils.TRAINING_DEVICE_ENV, "invalid")
    with pytest.raises(RuntimeError, match="must be one of"):
        training_utils.resolve_training_device_type()

    monkeypatch.setattr(training_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv(training_utils.TRAINING_DEVICE_ENV, "cuda")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        training_utils.resolve_training_device_type()


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


def test_training_local_text_paths_loaded_from_runpod_bundle_resolve_under_job_root(tmp_path: Path) -> None:
    job_root = tmp_path / "jobs" / "job-123"
    inputs_dir = job_root / "inputs"
    local_dataset = inputs_dir / "local_files" / "000-text" / "00000-shake.txt"
    local_dataset.parent.mkdir(parents=True, exist_ok=True)
    local_dataset.write_text("remote bundle text", encoding="utf-8")
    dataloader_config_path = inputs_dir / "dataloader_config.json"
    dataloader_config_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "text",
                        "data_files": {"train": "inputs/local_files/000-text/00000-shake.txt"},
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
            }
        ),
        encoding="utf-8",
    )

    config = load_training_dataloader_config(dataloader_config_path)
    tokenizer = _CharTokenizer()
    dataset = build_training_dataset(config, tokenizer, seq_len=1)

    windows = list(dataset)
    assert windows
    token_stream = [int(windows[0][0].item())]
    for _, targets in windows:
        token_stream.append(int(targets.item()))

    assert tokenizer.decode(token_stream) == "remote bundle text<|endoftext|>"


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


def test_runner_skips_torch_compile_by_default(monkeypatch, capsys) -> None:
    from training import runner

    model = torch.nn.Linear(1, 1)

    monkeypatch.delenv("LLM_STUDIO_TORCH_COMPILE", raising=False)

    def fail_compile(*_args, **_kwargs):
        raise AssertionError("torch.compile should be opt-in")

    monkeypatch.setattr(runner.torch, "compile", fail_compile)

    assert runner.maybe_compile_model(model, is_cuda=True) is model
    assert "set LLM_STUDIO_TORCH_COMPILE=1" in capsys.readouterr().out


def test_runner_skips_torch_compile_without_c_compiler_when_opted_in(monkeypatch, capsys) -> None:
    from training import runner

    model = torch.nn.Linear(1, 1)

    monkeypatch.setenv("LLM_STUDIO_TORCH_COMPILE", "1")
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


def test_attention_uses_pytorch_24_compatible_gqa_fallback(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_scaled_dot_product_attention(q, k, v, *, attn_mask=None, is_causal=False, **kwargs):
        calls["kwargs"] = kwargs
        calls["q_heads"] = q.size(1)
        calls["k_heads"] = k.size(1)
        calls["v_heads"] = v.size(1)
        calls["is_causal"] = is_causal
        return torch.zeros_like(q)

    monkeypatch.setattr(model_module.F, "scaled_dot_product_attention", fake_scaled_dot_product_attention)

    attention = CausalSelfAttention(0, 8, Attention(n_head=4, n_kv_head=2))
    x = torch.randn(1, 3, 8)
    out = attention(x, None)

    assert out.shape == x.shape
    assert calls["kwargs"] == {}
    assert calls["q_heads"] == 4
    assert calls["k_heads"] == 4
    assert calls["v_heads"] == 4
    assert calls["is_causal"] is True
