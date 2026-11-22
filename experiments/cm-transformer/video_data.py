"""Real-video (V1-V6, DERF) and deterministic text-prompt inputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import (FRAMES_DIR, REGIMES, REGIME_NAMES, ROOT, MODEL_DIR,
                    require_paper1_frames)

# Clip-style normalisation used by preprocessor_config.json: mean 0.5, std 0.5
MEAN = 0.5
STD = 0.5


def load_regime(name: str, size: int = 224) -> torch.Tensor:
    """[36, 3, size, size] float32 in [-1, 1]."""
    files = sorted(require_paper1_frames().glob(f"{name}_{REGIME_NAMES[name]}_*.png"))
    assert len(files) == 36, (name, len(files))
    out = []
    for f in files:
        im = Image.open(f).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255.0
        out.append((a - MEAN) / STD)
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).contiguous()


def load_all(size: int = 224) -> dict[str, torch.Tensor]:
    return {r: load_regime(r, size) for r in REGIMES}


def to_pairs(frames: torch.Tensor) -> torch.Tensor:
    """[36,3,H,W] -> [18,2,3,H,W] consecutive temporal pairs (M0 protocol)."""
    n = frames.shape[0]
    return torch.stack([frames[2 * i:2 * i + 2] for i in range(n // 2)])


def to_single_steps(frames: torch.Tensor) -> torch.Tensor:
    """[36,3,H,W] -> [36,2,3,H,W] with the frame replicated over the temporal slot.

    Streaming compaction is per displayed frame; temporal_patch_size=2 is satisfied
    by duplicating the frame (documented in README).
    """
    return frames[:, None].expand(-1, 2, -1, -1, -1).contiguous()


# --------------------------------------------------------------------------- #
# deterministic text prompts
# --------------------------------------------------------------------------- #
CALIB_PROMPTS = [
    "The quick brown fox jumps over the lazy dog while the sun sets behind the hills.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n",
    "In a distributed system, consensus requires that every correct node agrees on the same value.",
    "SELECT region, SUM(amount) FROM orders WHERE year = 2025 GROUP BY region ORDER BY 2 DESC;",
    "A vision transformer splits an image into fixed-size patches and attends over all of them.",
    "Explanation: the compiler lowers the intermediate representation into machine code after register allocation.",
    "Numbers 3.14159, 2.71828, 1.41421, and 0.57721 appear frequently in analysis.",
    "The committee will review the proposal on Tuesday, then publish a revised draft.",
]

EVAL_PROMPTS = [
    "Machine learning models learn patterns from data and generalise to unseen examples.",
    "for i in range(10):\n    total += i * i\nprint(total)",
    "The capital of France is Paris, and the capital of Japan is Tokyo.",
    "Attention is all you need: scaled dot-product attention mixes values by query-key similarity.",
]


def encode_prompts(prompts, max_len: int = 96, device="cpu") -> list[torch.Tensor]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    out = []
    for p in prompts:
        ids = tok.encode(p).ids[:max_len]
        out.append(torch.tensor(ids, dtype=torch.long, device=device))
    return out


def token_ids_report():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    rep = {}
    for name, group in (("calibration", CALIB_PROMPTS), ("eval", EVAL_PROMPTS)):
        rep[name] = []
        for p in group:
            ids = tok.encode(p).ids
            rep[name].append({"prompt": p, "n_tokens": len(ids), "ids_head": ids[:16]})
    return rep
