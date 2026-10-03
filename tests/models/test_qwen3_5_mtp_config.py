# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Qwen3.5 MTP speculative decoding: config overrides and
draft logits."""

import types
from typing import Any

import pytest
import torch
from transformers import AutoConfig, PreTrainedConfig

from vllm.config.speculative import SpeculativeConfig

_CHECKPOINTS = {
    "qwen3_5": "Qwen/Qwen3.8-27B",
    "qwen3_5_moe": "Qwen/Qwen3.6-35B-A3B",
}


def _mtp_config(model_type: str) -> PreTrainedConfig:
    """Create a top-level MTP configuration with mtp_num_hidden_layers."""
    kwargs: dict[str, Any] = {
        "model_type": model_type,
        "architectures": ["SomeArch"],
        "mtp_num_hidden_layers": 1,
    }
    return PreTrainedConfig(**kwargs)


def _multimodal_wrapper_mtp_config(
    model_type: str, mtp_layers: int = 1
) -> PreTrainedConfig:
    """Download a multimodal wrapper config via AutoConfig.from_pretrained
    and configure mtp_num_hidden_layers in text_config.

    Uses real-world Hugging Face Hub checkpoints:
    - Dense (qwen3_5):     Qwen/Qwen3.8-27B
    - MoE   (qwen3_5_moe): Qwen/Qwen3.6-35B-A3B
    """
    repo = _CHECKPOINTS[model_type]
    config: PreTrainedConfig = AutoConfig.from_pretrained(repo)
    text_config = config.get_text_config()
    text_config.mtp_num_hidden_layers = mtp_layers
    return config


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
        # Text-only config variants must map to the same MTP architectures.
        ("qwen3_5_text", "Qwen3_5MTP"),
        ("qwen3_5_moe_text", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_recognizes_text_only_types(
    model_type: str, expected_arch: str
) -> None:
    """Verify that text-only config variants map to the expected MTP architectures."""
    cfg = SpeculativeConfig.hf_config_override(_mtp_config(model_type))
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 1


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_extracts_n_predict_from_multimodal_wrapper(
    model_type: str, expected_arch: str
) -> None:
    """Verify that multimodal wrapper checkpoints with mtp_num_hidden_layers
    in text_config resolve n_predict and architecture correctly."""
    cfg = SpeculativeConfig.hf_config_override(
        _multimodal_wrapper_mtp_config(model_type, mtp_layers=2)
    )
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 2


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_top_level_precedence_over_nested_text_config(
    model_type: str, expected_arch: str
) -> None:
    """Verify that an explicit top-level mtp_num_hidden_layers takes precedence
    over a nested text_config value."""
    cfg = _multimodal_wrapper_mtp_config(model_type, mtp_layers=2)
    cfg.mtp_num_hidden_layers = 3
    overridden = SpeculativeConfig.hf_config_override(cfg)
    assert overridden.model_type == "qwen3_5_mtp"
    assert overridden.architectures == [expected_arch]
    assert overridden.n_predict == 3


@pytest.mark.parametrize(
    "model_id,expected_arch",
    [
        ("Qwen/Qwen3.8-27B", "Qwen3_5MTP"),
        ("Qwen/Qwen3.6-35B-A3B", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_downloads_real_hf_hub_configs(
    model_id: str, expected_arch: str
) -> None:
    """Verify that unmodified real-world checkpoints downloaded via
    AutoConfig.from_pretrained resolve n_predict=1 from text_config."""
    hf_config = AutoConfig.from_pretrained(model_id)
    cfg = SpeculativeConfig.hf_config_override(hf_config)
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 1


def test_mtp_draft_vocab_prefix_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    """VLLM_MTP_DRAFT_VOCAB_SIZE keeps draft logits for the first N token ids
    and gives every other id -inf, so drafts only propose prefix ids."""
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    monkeypatch.setenv("VLLM_MTP_DRAFT_VOCAB_SIZE", "8")
    torch.manual_seed(0)
    hidden = torch.randn(3, 16)
    weight = torch.randn(32, 16)
    model = types.SimpleNamespace(
        lm_head=types.SimpleNamespace(weight=weight, tp_size=1),
        logits_processor=types.SimpleNamespace(org_vocab_size=32),
    )
    model._compute_prefix_logits = types.MethodType(
        Qwen3_5MTP._compute_prefix_logits, model
    )

    logits = Qwen3_5MTP.compute_logits(model, hidden)

    assert logits.shape == (3, 32)
    torch.testing.assert_close(logits[:, :8], hidden @ weight[:8].T)
    assert torch.isneginf(logits[:, 8:]).all()
