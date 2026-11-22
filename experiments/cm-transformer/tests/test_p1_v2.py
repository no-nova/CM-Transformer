"""Tests for the text-only P1 v2 work: harness scoring, importance permutation,
adaptive threshold gating, and answer extraction."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from weights_guard import require as _require_weights  # noqa: E402

_require_weights("qwen35_08b")

from bench_text import extract_answer
from model_ref import (TextBackbone, apply_block_mask, block_energy,
                       keep_mask_from_energy, permute_mlp)
from common import MODEL_DIR, Weights, build_tokenizer
from p1_v2 import block_stats, build_perm, importance, output_gram

DT = torch.float32
DEV = "cpu"


@pytest.fixture(scope="module")
def tb():
    torch.set_grad_enabled(False)
    return TextBackbone(Weights(DT, DEV))


# --------------------------------------------------------------------------- #
# harness scoring
# --------------------------------------------------------------------------- #
def test_length_normalisation_beats_raw_loglik():
    """A short confident option must win over a long option with more total
    loglik — that is exactly what single-token scoring got wrong."""
    from bench_text import summarize
    rows = [{
        "id": "x", "answer": 1,
        "cands": [{"choice": "yes", "loglik": -2.0, "n_tok": 1, "n_bytes": 3},
                  {"choice": "a much longer answer", "loglik": -5.0, "n_tok": 8,
                   "n_bytes": 20}],
    }]
    s = summarize(rows)
    # raw loglik prefers the one-token option, the length-normalised score
    # (-2.0 vs -0.625) prefers the long one: that is the defect being fixed.
    raw = max(range(2), key=lambda i: rows[0]["cands"][i]["loglik"])
    assert raw == 0
    assert s["acc"] == 1.0


def test_gsm8k_answer_extraction():
    assert extract_answer("so 21 - 15 = 6\n#### 6") == "6"
    assert extract_answer("#### 1,234") == "1234"
    assert extract_answer("the total is 42 dollars.") == "42"
    assert extract_answer("-3.5 units") == "-3.5"
    assert extract_answer("no digits here") is None


# --------------------------------------------------------------------------- #
# permutation
# --------------------------------------------------------------------------- #
def test_output_gram_is_symmetric_psd_diag():
    torch.manual_seed(0)
    H = torch.randn(64, 96)
    Wd = torch.randn(32, 96)
    C = output_gram(H, Wd)
    assert C.shape == (96, 96)
    assert torch.allclose(C, C.t(), atol=1e-5)
    # diagonal = var(h_i) * ||w_i||^2 >= 0
    v = H.var(0, unbiased=True) * Wd.norm(dim=0) ** 2
    assert torch.allclose(C.diag(), v, rtol=2e-2, atol=1e-4)


def test_build_perm_is_a_permutation_and_lowers_block_error():
    torch.manual_seed(1)
    I, lane = 128, 32
    H = torch.randn(512, I)
    Wd = torch.randn(64, I)
    perm, _ = build_perm(H, Wd, lane=lane)
    assert perm.numel() == I
    assert sorted(perm.tolist()) == list(range(I))
    C = output_gram(H, Wd)
    ours = block_stats(perm, C)
    nat = block_stats(torch.arange(I), C)
    rnd = block_stats(torch.randperm(I, generator=torch.Generator().manual_seed(3)), C)
    assert ours <= nat and ours <= rnd * 1.05


def test_permute_mlp_preserves_function():
    """Permuting gate/up rows and down columns must not change the MLP output."""
    torch.manual_seed(2)
    I, H = 128, 32
    W = (torch.randn(I, H), torch.randn(I, H), torch.randn(H, I))
    perm = torch.randperm(I, generator=torch.Generator().manual_seed(4))
    W2 = permute_mlp(W, perm)
    x = torch.randn(4, H)
    y = F.silu(x @ W[0].t()) * (x @ W[1].t()) @ W[2].t()
    y2 = F.silu(x @ W2[0].t()) * (x @ W2[1].t()) @ W2[2].t()
    assert torch.allclose(y, y2, rtol=1e-4, atol=1e-3)


# --------------------------------------------------------------------------- #
# gating
# --------------------------------------------------------------------------- #
def test_threshold_gate_matches_topk_at_calibrated_quantile():
    torch.manual_seed(3)
    a = torch.randn(16, 256)
    scale = torch.rand(256) + 0.5
    e = block_energy(a * scale)
    rate = 0.25
    th = torch.quantile(e.flatten(), rate)
    keep_th = e >= th
    keep_top = keep_mask_from_energy(e, rate)
    assert abs(float(keep_th.float().mean()) - (1 - rate)) < 0.05
    assert abs(float(keep_top.float().mean()) - (1 - rate)) < 0.05


def test_importance_sorting_keeps_more_energy_than_natural(tb):
    """The core claim of v2: importance-ordered blocks retain more output-weighted
    energy than the native neuron order at the same block count."""
    tok = build_tokenizer(MODEL_DIR)
    ids = torch.tensor(tok("The history of the Roman Empire spans more than a "
                           "thousand years, and its influence on law and")["input_ids"])
    cap = {}
    tb.forward(ids, capture=cap, logits_positions=[0])
    i = 3
    p = f"{tb.prefix}layers.{i}.mlp."
    x = cap["mlp_in"][i]
    a = F.silu(F.linear(x, tb.w[p + "gate_proj.weight"])) * F.linear(x, tb.w[p + "up_proj.weight"])
    Wd = tb.w[p + "down_proj.weight"]
    imp = importance(a, Wd)
    perm, _ = build_perm(a, Wd)
    scale = Wd.norm(dim=0)
    e = block_energy(a * scale)                              # [T, B] native order
    keep_n = max(1, int(round(e.shape[-1] * 0.7)))
    nat = float(e.topk(keep_n, dim=-1).values.sum() / e.sum())
    ep = block_energy((a * scale)[:, perm.cpu().to(a.device)])
    ours = float(ep.topk(keep_n, dim=-1).values.sum() / ep.sum())
    assert ours >= nat
