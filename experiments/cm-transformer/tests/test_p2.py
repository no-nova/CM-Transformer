"""P2 tests: zero-halo dirty-token detection, streaming compaction, guards."""
import json
import sys
from pathlib import Path

import pytest
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from weights_guard import require as _require_weights  # noqa: E402

_require_weights("qwen35_08b")

from common import RESULTS_DIR, Weights  # noqa: E402
from model_ref import VisionEncoder  # noqa: E402
import p2_stream as P  # noqa: E402

GRID, NTOK, PATCH = P.GRID, P.NTOK, P.PATCH


def static_frames(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(3, 224, 224, generator=g)
    return base[None].repeat(n, 1, 1, 1) * 2 - 1


def moving_square(n=8, seed=0):
    frames = torch.zeros(n, 3, 224, 224)
    for t in range(n):
        y, x = 32 + 16 * t, 32 + 16 * t
        frames[t, :, y:y + 32, x:x + 32] = 1.0
    return frames


# ------------------------------------------------------------------ detection
def test_per_patch_rel_l2_identity_is_zero():
    f = static_frames(2)
    rel = P.per_patch_rel_l2(f[0], f[1])
    assert tuple(rel.shape) == (NTOK,)
    assert torch.allclose(rel, torch.zeros(NTOK), atol=1e-7)


def test_per_patch_rel_l2_localises_a_single_tile():
    f = static_frames(2)
    f2 = f[1].clone()
    f2[:, 32:48, 64:80] += 0.5
    rel = P.per_patch_rel_l2(f[0], f2)
    hot = torch.nonzero(rel > 1e-4, as_tuple=False).flatten().tolist()
    assert hot == [2 * GRID + 4]


def test_dilate_grows_a_patch_mask_by_one_ring():
    m = torch.zeros(NTOK, dtype=torch.bool)
    m[7 * GRID + 7] = True                      # centre of the 14x14 grid
    assert int(P.dilate(m, 0).sum()) == 1
    assert int(P.dilate(m, 1).sum()) == 9
    assert int(P.dilate(m, 2).sum()) == 25
    corner = torch.zeros(NTOK, dtype=torch.bool)
    corner[0] = True
    assert int(P.dilate(corner, 1).sum()) == 4   # clipped at the border


def test_detect_dirty_static_input_only_flushes_frame_zero():
    frames = static_frames(6)
    dirty, scores, flush = P.detect_dirty(frames)
    assert flush[0] and not any(flush[1:])
    assert bool(dirty[0].all())
    assert all(int(d.sum()) == 0 for d in dirty[1:])


def test_detect_dirty_flags_motion_and_flushes_on_scene_cut():
    frames = moving_square(8)
    frames[5] = torch.randn(3, 224, 224)         # hard cut at frame 5
    dirty, scores, flush = P.detect_dirty(frames)
    assert flush[5]
    assert bool(dirty[5].all())
    assert scores[5] > max(scores[1:5])


def test_age_guard_refreshes_periodically():
    clean = [torch.zeros(NTOK, dtype=torch.bool) for _ in range(12)]
    out = P.apply_age_guard(clean, max_age=2)
    refreshed = [i for i, d in enumerate(out) if bool(d.any())]
    assert refreshed == [2, 5, 8, 11]    # refresh once the age exceeds max_age
    assert all(int(d.sum()) in (0, NTOK) for d in out)   # all-or-nothing refresh


def test_drift_guard_tracks_cumulative_change():
    still = static_frames(5)
    no = P.apply_drift_guard(still, [torch.zeros(NTOK, dtype=torch.bool)] * 5, 0.1)
    assert not any(bool(d.any()) for d in no)
    busy = moving_square(5) * 0.5
    yes = P.apply_drift_guard(busy, [torch.zeros(NTOK, dtype=torch.bool)] * 5, 0.1)
    assert any(bool(d.any()) for d in yes)


# ------------------------------------------------------------------ streaming
@pytest.fixture(scope="module")
def enc():
    return VisionEncoder(Weights(torch.float32))


def test_gathered_patch_tokens_match_the_dense_embedding(enc):
    frames = static_frames(1)[0]
    step = torch.stack([frames, frames])
    sv = P.StreamingVision(enc)
    idx = torch.tensor([5, 40, 191])
    d = torch.zeros(NTOK, dtype=torch.bool)
    d[idx] = True
    with torch.no_grad():
        dense = enc.embed(step[None])[0]
        got = sv.embed_dirty(step, idx)
    assert got.shape == (3, 768)
    assert (got - dense[idx]).abs().max() < 1e-5


def test_all_dirty_frame_is_bit_exact_against_dense(enc):
    frames = static_frames(2)
    step = torch.stack([frames[1], frames[1]])
    sv = P.StreamingVision(enc)
    with torch.no_grad():
        ref = enc.forward(step[None])
        merged, _ = sv.step(step, torch.ones(NTOK, dtype=torch.bool))
    assert torch.equal(merged, ref)


def test_streaming_carries_state_and_stays_finite(enc):
    frames = static_frames(3)
    sv = P.StreamingVision(enc)
    with torch.no_grad():
        for t in range(3):
            step = torch.stack([frames[t], frames[t]])
            merged, h = sv.step(step, torch.ones(NTOK, dtype=torch.bool) if t == 0
                                else torch.zeros(NTOK, dtype=torch.bool))
    assert tuple(merged.shape) == (49, 1024)
    assert torch.isfinite(merged).all()
    assert sv.K[0] is not None and tuple(sv.K[0].shape) == (NTOK, 12, 64)


# --------------------------------------------------------------- MAC algebra
def test_mac_accounting_is_self_consistent():
    total = (P.DENSE_MAC["patch_embed"]
             + 12 * (P.DENSE_MAC["qkv"] + P.DENSE_MAC["attn_qk_av"] + P.DENSE_MAC["proj"]
                     + P.DENSE_MAC["mlp"])
             + P.DENSE_MAC["merger"])
    assert total == P.DENSE_TOTAL
    assert 0.6 < P.MLP_SHARE < 0.7
    assert P.DENSE_MAC["mlp"] == 2 * NTOK * 768 * 3072


# ---------------------------------------------------------- recorded numbers
def _load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(p.read_text())


def test_recorded_flush_frames_are_exact():
    d = _load("p2_cuda.json")
    for reg, rows in d["regimes"].items():
        for pol, prow in rows["policies"].items():
            for key, v in prow["variants"].items():
                if v["skip_rate"] != 0.0:
                    continue          # P1 block skipping is the only loss on a flushed frame
                assert v["flush_frame_max_abs"] == 0.0, (reg, pol, key)
                assert v["flush_frame_rel_l2"] == 0.0, (reg, pol, key)


def test_recorded_zero_halo_dirty_ratio_is_flat_over_layers():
    d = _load("p2_cuda.json")
    for reg, rows in d["regimes"].items():
        percolation = rows["halo_percolation_3x3_stage"]
        assert percolation[0] == pytest.approx(rows["mean_dirty_ratio"], abs=1e-6)
        assert all(b >= a - 1e-9 for a, b in zip(percolation, percolation[1:]))
        assert percolation[-1] > 3 * percolation[0]        # what a 3x3 CNN would suffer


def test_recorded_p2_beats_dense_on_slow_regimes():
    d = _load("p2_cuda.json")
    for reg in ("V3", "V5", "V6"):
        v = d["regimes"][reg]["policies"]["tau"]["variants"]["0.0|isomorphic"]
        assert v["wallclock_speedup"] > 1.3
        assert v["mac_reduction"] > 0.8
