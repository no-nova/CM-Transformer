"""Cross-check `model_ref.py` against the official `transformers` implementation.

    python3 verify_official.py [--vision] [--layers]

Requires transformers>=5.23 (which ships `models/qwen3_5`). Writes
`results/verify_official.json` and prints per-layer max-abs / cosine.

The reference implementation is our instrumented copy of Qwen3.5 (plain weight
dict, no HF modules) used by the P1/P2 experiments; this script proves it now
matches the official forward pass, which is the prerequisite for any capability
number reported by the paper.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from common import MODEL_DIR, RESULTS_DIR, Weights, SAFETENSORS, dump_json, log
from model_ref import TextBackbone, VisionEncoder
from video_data import CALIB_PROMPTS, EVAL_PROMPTS, encode_prompts, load_regime, to_pairs


def official_text(device="cuda", dtype=torch.float32):
    from transformers import Qwen3_5ForCausalLM
    m = Qwen3_5ForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype,
                                           attn_implementation="sdpa")
    return m.to(device).eval()


def official_full(device="cuda", dtype=torch.float32):
    from transformers import Qwen3_5ForConditionalGeneration
    m = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=dtype,
                                                        attn_implementation="sdpa")
    return m.to(device).eval()


def cmp(a, b, name, out, extra=None):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    row = {"name": name, "max_abs": d.max().item(), "mean_abs": d.mean().item(),
           "cosine": cos, "a_absmax": a.abs().max().item(), "b_absmax": b.abs().max().item()}
    if extra:
        row.update(extra)
    out.append(row)
    log(f"  {name:34s} max|Δ|={row['max_abs']:.3e} cos={cos:.6f} "
        f"|a|max={row['a_absmax']:.2f} |b|max={row['b_absmax']:.2f}")
    return row


@torch.no_grad()
def check_text(prompts, dev="cuda", dt=torch.float32):
    off = official_text(dev, dt)
    ref = TextBackbone(Weights(dt, dev))
    rows = []
    for name, text in prompts:
        ids = encode_prompts([text])[0].to(dev)
        o = off(input_ids=ids.unsqueeze(0), output_hidden_states=True)
        r = ref.forward(ids, capture={})
        # official hidden_states[0] = embeddings, [i+1] = after layer i
        oh = o.hidden_states
        cmp(r["logits"][-1], o.logits[0, -1], f"{name}:logits[last]",
            rows, {"n_tokens": len(ids), "argmax_ref": int(r["logits"][-1].argmax()),
                   "argmax_off": int(o.logits[0, -1].argmax())})
        for i in (0, 7, 15, 22):
            cmp(r["hidden_layers"][i], oh[i + 1][0], f"{name}:layer{i}", rows)
        # HF appends the post-final-norm state as the last entry
        cmp(r["hidden"], oh[-1][0], f"{name}:hidden_normed", rows)
    return rows, off, ref


@torch.no_grad()
def check_vision(pairs, dev="cuda", dt=torch.float32):
    off = official_full(dev, dt)
    vis = off.model.visual
    ref = VisionEncoder(Weights(dt, dev))
    rows = []
    x = pairs[0][:2].to(dev, dt)                     # keep it small for fp64 CPU
    # official expects flattened patches: (N, C*T*P*P) and grid_thw=(t,h,w)
    S = x.shape[0]
    flat = ref.patchify(x).to(dev)                   # [S*196, 1536]
    grid = torch.tensor([[1, 14, 14]] * S, device=dev)
    out = vis(hidden_states=flat, grid_thw=grid)
    got = out.pooler_output
    mine = ref.forward(x)
    # ours is [S*49, 1024] in raster-then-merge order; official already merged
    cmp(mine, got, "vision:merged", rows)
    # also compare the pre-merger token stream
    cap = {}
    ref.forward(x, capture=cap)
    cmp(cap["pre_merger"], out.last_hidden_state, "vision:pre_merger", rows)

    return rows, off, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    dev = args.device
    dt = getattr(torch, args.dtype)
    out = {"device": dev, "dtype": args.dtype, "text": [], "vision": []}

    prompts = [("calib0", CALIB_PROMPTS[0]), ("calib3", CALIB_PROMPTS[3]),
               ("eval0", EVAL_PROMPTS[0])]
    log("text: reference vs official")
    rows, off, ref = check_text(prompts, dev, dt)
    out["text"] = rows
    del off

    if args.vision:
        log("vision: reference vs official")
        pairs = to_pairs(load_regime("V2"))
        rows_v, off2, _ = check_vision([pairs], dev, dt)
        out["vision"] = rows_v

    tag = "" if args.dtype == "float32" else "_" + args.dtype
    dest = RESULTS_DIR / f"verify_official{tag}.json"
    if not args.vision and dest.exists():
        # a text-only run must not erase the vision rows recorded by an earlier
        # --vision run; keep them and overwrite only the text block
        try:
            prev = json.loads(dest.read_text())
        except (OSError, ValueError):
            prev = {}
        else:
            if prev.get("vision"):
                out["vision"] = prev["vision"]
                log("kept the vision rows from the previous run (use --vision to refresh)")
    dump_json(out, dest)
    worst_t = max(r["max_abs"] for r in out["text"])
    log(f"worst text max|Δ| = {worst_t:.3e}")
    if out["vision"]:
        log(f"worst vision max|Δ| = {max(r['max_abs'] for r in out['vision']):.3e}")


if __name__ == "__main__":
    main()
