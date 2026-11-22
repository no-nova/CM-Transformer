"""Tests for the P1 text capability harness: incremental decoding, embeds path,
N-D block gating, and the small metric helpers used by the capability sweep."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from weights_guard import require as _require_weights  # noqa: E402

_require_weights("qwen35_08b")

from common import MODEL_DIR, RESULTS_DIR, Weights, build_tokenizer
from model_ref import (TextBackbone, apply_block_mask, block_energy,
                       keep_mask_from_energy, new_state, permute_mlp)
from run_text_eval import first_divergence, rouge_l
import vlm_eval as V

DT = torch.float32
DEV = "cpu"


@pytest.fixture(scope="module")
def tb():
    torch.set_grad_enabled(False)
    return TextBackbone(Weights(DT, DEV))


@pytest.fixture(scope="module")
def tok():
    return build_tokenizer(MODEL_DIR)


def test_incremental_decode_matches_full_forward(tb, tok):
    """Prefill + single-token steps must reproduce the full teacher-forced pass."""
    ids = torch.tensor(tok("The capital of France is Paris, and the")["input_ids"])
    full = tb.forward(ids)["logits"]
    st = new_state()
    pre = tb.forward(ids[:6].unsqueeze(0), state=st, pos0=0)["logits"][-1]
    got = [pre]
    for t in range(6, ids.numel()):
        got.append(tb.forward(ids[t:t + 1].unsqueeze(0), state=st, pos0=t)["logits"][-1])
    got = torch.stack(got)
    assert torch.allclose(got, full[5:], atol=2e-3, rtol=2e-4)
    assert torch.equal(got.argmax(-1), full[5:].argmax(-1))


def test_embeds_path_equals_ids_path(tb, tok):
    ids = torch.tensor(tok("Machine learning models")["input_ids"])
    emb = F.embedding(ids, tb.w[tb.prefix + "embed_tokens.weight"])
    a = tb.forward(ids)["logits"]
    b = tb.forward(embeds=emb)["logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_block_energy_and_mask_are_nd(tb):
    a = torch.randn(2, 5, 256)
    e = block_energy(a, 32)
    assert e.shape == (2, 5, 8)
    m = keep_mask_from_energy(e, 0.25)
    assert m.shape == (2, 5, 8)
    assert int(m.sum(-1).min()) == 6 and int(m.sum(-1).max()) == 6
    masked = apply_block_mask(a, m, 32)
    kept = masked.reshape(2, 5, 8, 32)
    # every fully-kept block is untouched, every skipped block is exactly zero
    assert torch.equal(kept[m], a.reshape(2, 5, 8, 32)[m])
    assert float(kept[~m].abs().max()) == 0.0


def test_permute_mlp_is_isomorphic(tb, tok):
    ids = torch.tensor(tok("In 1969, humans first")["input_ids"])
    base = tb.forward(ids)["logits"]
    perm = torch.randperm(3584, generator=torch.Generator().manual_seed(7))
    ov = {}
    for i in range(tb.cfg.layers):
        pre = f"{tb.prefix}layers.{i}.mlp."
        ov[i] = permute_mlp((tb.w[pre + "gate_proj.weight"], tb.w[pre + "up_proj.weight"],
                             tb.w[pre + "down_proj.weight"]), perm)
    alt = tb.forward(ids, mlp_override=ov)["logits"]
    assert float((alt - base).abs().max()) < 1e-3
    assert torch.equal(alt.argmax(-1), base.argmax(-1))


def test_rouge_l_and_first_divergence():
    assert rouge_l("a b c", "a b c") == 1.0
    assert rouge_l("a b c d", "a b") == pytest.approx(2 * (2 / 4) * 1.0 / (2 / 4 + 1.0))
    assert rouge_l("", "") == 1.0
    assert first_divergence([1, 2, 3], [1, 2, 3]) == 3
    assert first_divergence([1, 9, 3], [1, 2, 3]) == 1


def test_vlm_template_places_image_tokens(tok):
    n = 49
    ids = V.template_ids(tok, n, "Describe this video.")
    assert ids.count(V.IMG_TOK) == n
    start = len(tok("<|im_start|>user\n<|vision_start|>", add_special_tokens=False)["input_ids"])
    assert ids[start:start + n] == [V.IMG_TOK] * n


def test_capability_results_sane():
    """Whatever capability_*.json exists must satisfy the protocol invariants."""
    files = sorted(RESULTS_DIR.glob("capability_*.json"))
    if not files:
        pytest.skip("no capability results yet")
    for f in files:
        d = json.loads(f.read_text())
        rows = d["rows"]
        m0 = rows[0]
        assert m0["variant"] == "M0" and m0["rate"] == 0.0
        if "ppl" in m0:
            assert 3.0 < m0["ppl"] < 200.0, (f.name, m0["ppl"])
        for r in rows[1:]:
            if r["rate"] == 0.0:
                assert abs(r.get("delta_ppl", 0.0)) < 1e-3
            if "delta_ppl" in r:
                assert r["delta_ppl"] > -0.05      # skipping cannot help perplexity
