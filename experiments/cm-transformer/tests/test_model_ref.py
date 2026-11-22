import json
import sys
from pathlib import Path

import pytest
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from weights_guard import require as _require_weights  # noqa: E402

_require_weights("qwen35_08b")

from common import MODEL_DIR, RESULTS_DIR, SAFETENSORS, Weights, rms_norm  # noqa: E402
from model_ref import (TextBackbone, TextCfg, VisionCfg, VisionEncoder,  # noqa: E402
                       apply_block_mask, block_energy, gelu_tanh, keep_mask_from_energy,
                       layer_norm, permute_mlp, silu)

CONFIG = json.loads((MODEL_DIR / "config.json").read_text())


@pytest.fixture(scope="module")
def cfg_pair():
    return VisionCfg(), TextCfg()


# ---------------------------------------------------------------- architecture
def test_config_matches_paper_description(cfg_pair):
    v, t = cfg_pair
    assert (v.depth, v.hidden, v.heads, v.inter) == (12, 768, 12, 3072)
    assert (v.patch, v.temporal, v.merge) == (16, 2, 2)
    assert v.pos_embed_n == 2304 and v.out_hidden == 1024
    assert (t.layers, t.hidden, t.inter) == (24, 1024, 3584)
    assert t.heads == 8 and t.kv_heads == 2 and t.head_dim == 256
    tc = CONFIG["text_config"]
    assert tc["num_hidden_layers"] == 24 and tc["tie_word_embeddings"] is True
    assert tc["attn_output_gate"] is True and tc["linear_conv_kernel_dim"] == 4
    assert t.vocab == 248320 and CONFIG["text_config"]["hidden_act"] == "silu"


def test_layer_types_are_3_linear_then_full():
    lt = CONFIG["text_config"]["layer_types"]
    assert len(lt) == 24
    full = [i for i, x in enumerate(lt) if x == "full_attention"]
    assert full == [3, 7, 11, 15, 19, 23]
    assert sum(1 for x in lt if x == "linear_attention") == 18


def test_checkpoint_shapes(cfg_pair):
    v, t = cfg_pair
    w = Weights()
    vp = "model.visual."
    assert tuple(w[vp + "patch_embed.proj.weight"].shape) == (768, 3, 2, 16, 16)
    assert tuple(w[vp + "blocks.0.attn.qkv.weight"].shape) == (2304, 768)
    assert tuple(w[vp + "blocks.11.mlp.linear_fc2.weight"].shape) == (768, 3072)
    assert tuple(w[vp + "merger.linear_fc1.weight"].shape) == (3072, 3072)
    assert tuple(w[vp + "merger.linear_fc2.weight"].shape) == (1024, 3072)
    tp = "model.language_model."
    assert tuple(w[tp + "layers.3.mlp.gate_proj.weight"].shape) == (3584, 1024)
    assert tuple(w[tp + "layers.3.mlp.down_proj.weight"].shape) == (1024, 3584)
    assert tuple(w[tp + "layers.3.self_attn.q_proj.weight"].shape) == (4096, 1024)
    assert tuple(w[tp + "layers.3.self_attn.o_proj.weight"].shape) == (1024, 2048)
    assert tuple(w[tp + "layers.0.linear_attn.in_proj_qkv.weight"].shape) == (6144, 1024)
    assert tuple(w[tp + "layers.0.linear_attn.conv1d.weight"].shape) == (6144, 1, 4)
    assert str(w[tp + "layers.0.linear_attn.A_log"].dtype) == "torch.float32"


# ------------------------------------------------------------------- numerics
def test_rms_norm_matches_reference():
    """Qwen3.5 RMSNorm is zero-centred: y = x/rms(x) * (1 + w)."""
    x = torch.randn(4, 8, dtype=torch.float32)
    g = torch.rand(8) + 0.5
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + g)
    assert torch.allclose(rms_norm(x, g, 1e-6), ref, atol=1e-6)


def test_swiglu_equivalence_with_permutation():
    """Exact functional isomorphism of a 32-lane permutation (zero skipping)."""
    torch.manual_seed(0)
    I, D, T = 128, 32, 5
    g, u = torch.randn(I, D), torch.randn(I, D)
    bg, bu = torch.randn(I), torch.randn(I)
    d = torch.randn(D, I)
    x = torch.randn(T, D)
    perm = torch.randperm(I)
    y0 = (silu(x @ g.T + bg) * (x @ u.T + bu)) @ d.T
    y1 = (silu(x @ g[perm].T + bg[perm]) * (x @ u[perm].T + bu[perm])) @ d[:, perm].T
    tol = 1e-5 * y0.abs().max()
    assert (y0 - y1).abs().max() < tol
    gp, up, dp = permute_mlp((g, u, d), perm)
    y2 = (silu(x @ gp.T + bg[perm]) * (x @ up.T + bu[perm])) @ dp.T
    assert (y0 - y2).abs().max() < tol


def test_block_energy_and_mask_semantics():
    a = torch.randn(3, 320)
    a[:, 32:] = 0.0                      # only the first 32-lane block has energy
    e = block_energy(a)
    assert tuple(e.shape) == (3, 10)
    assert torch.allclose(e[:, 1:], torch.zeros(3, 9))
    keep = keep_mask_from_energy(e, 0.5)
    assert int(keep.sum(-1).min()) == 5            # round(10 * (1 - 0.5))
    kept = apply_block_mask(a, keep)
    assert torch.allclose(kept, a)                 # dropping empty blocks is lossless
    lo = apply_block_mask(a, keep_mask_from_energy(e, 0.9))
    assert (lo.abs().sum(-1) <= a.abs().sum(-1) + 1e-6).all()


def test_vision_patch_embed_has_zero_halo():
    """A change confined to one 16x16 tile must leave every other patch token
    bit-identical: L_halo = 0 for the stride-16 patch embed."""
    enc = VisionEncoder(Weights(torch.float32))
    base = torch.zeros(1, 2, 3, 224, 224)
    pert = base.clone()
    pert[0, :, :, 32:48, 64:80] = 1.0            # tile (row 2, col 4) of the 14x14 grid
    with torch.no_grad():
        h0 = enc.embed(base)[0]
        h1 = enc.embed(pert)[0]
    diff = (h0 - h1).abs().sum(-1).reshape(196)
    changed = torch.nonzero(diff > 0, as_tuple=False).flatten().tolist()
    # tokens are stored in spatial-merge order (block_row, block_col, in_row, in_col)
    r, c = 2, 4
    m = enc.cfg.merge
    bw, i_r, i_c = 14 // m, r % m, c % m
    expected = [(r // m) * bw + (c // m)] * 0  # placeholder, computed below
    expect = ((r // m) * (14 // m) + (c // m)) * (m * m) + i_r * m + i_c
    assert changed == [expect], (changed, expect)


# --------------------------------------------------------------------- models
def test_vision_forward_shapes_and_determinism():
    enc = VisionEncoder(Weights(torch.float32))
    frames = torch.zeros(1, 2, 3, 224, 224)
    with torch.no_grad():
        a = enc.forward(frames)
        b = enc.forward(frames)
    assert tuple(a.shape) == (49, 1024)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()


def test_text_forward_is_deterministic_and_causal():
    tb = TextBackbone(Weights(torch.float32))
    ids = torch.tensor([10, 20, 30, 40, 50, 60])
    with torch.no_grad():
        l1 = tb.forward(ids)["logits"]
        l2 = tb.forward(ids)["logits"]
        l3 = tb.forward(torch.cat([ids, torch.tensor([70])]))["logits"]
    assert torch.equal(l1, l2)
    assert tuple(l1.shape) == (6, 248320)
    # changing a later token must not move earlier logits
    d = (l1[:-1] - l3[:l1.shape[0] - 1]).abs().max()
    assert d < 1e-3 * l1.abs().max()
    assert torch.isfinite(l1).all()


def test_partial_rope_touches_only_first_quarter():
    tb = TextBackbone(Weights(torch.float32))
    q = torch.ones(4, 8, 256)
    k = torch.ones(4, 2, 256)
    q2, k2 = tb.rope(q.clone(), k.clone())
    assert torch.allclose(q2[..., 64:], q[..., 64:])      # untouched tail
    assert not torch.allclose(q2[..., :64], q[..., :64])  # rotated head
    assert torch.allclose(k2[..., 64:], k[..., 64:])
