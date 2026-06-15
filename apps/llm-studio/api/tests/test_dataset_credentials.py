from __future__ import annotations

import json

from app.dataset_credentials import (
    encode_dataset_hf_tokens,
    inject_dataset_hf_tokens,
    split_dataset_hf_tokens,
    strip_hf_tokens,
)


def test_dataset_credentials_split_and_inject_without_mutating_input() -> None:
    original = {
        "datasets": [
            {"name": "private-a", "hf_token": " hf_embedded "},
            {"name": "private-b"},
            "legacy-invalid-entry",
        ],
        "nested": {"hf-token": "also-secret", "safe": True},
    }

    sanitized, tokens = split_dataset_hf_tokens(original, fallback_token=" hf_fallback ")

    assert tokens == ["hf_embedded", "hf_fallback", "hf_fallback"]
    assert sanitized["datasets"] == [
        {"name": "private-a"},
        {"name": "private-b"},
        "legacy-invalid-entry",
    ]
    assert original["datasets"][0]["hf_token"] == " hf_embedded "
    assert inject_dataset_hf_tokens(sanitized, tokens)["datasets"][:2] == [
        {"name": "private-a", "hf_token": "hf_embedded"},
        {"name": "private-b", "hf_token": "hf_fallback"},
    ]


def test_strip_hf_tokens_recursively_handles_normalized_key_variants() -> None:
    sanitized = strip_hf_tokens(
        {
            "hf_token": "one",
            "nested": [{"HF-TOKEN": "two", "safe": "value"}],
            "tuple": ({"hf token": "three"},),
        }
    )

    assert sanitized == {
        "nested": [{"safe": "value"}],
        "tuple": [{}],
    }


def test_encode_dataset_hf_tokens_only_emits_non_empty_credentials() -> None:
    assert encode_dataset_hf_tokens([None, "  "]) is None
    assert json.loads(encode_dataset_hf_tokens([None, " hf_private "]) or "") == [
        None,
        "hf_private",
    ]
