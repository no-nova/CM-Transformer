"""Shared infrastructure for Paper 2 (CM-Transform-Accel) on Qwen3.5-0.8B.

Self-contained: no `transformers` dependency.  Weights are memory-mapped from the
unmodified `models/Qwen3.5-0.8B/model.safetensors` file and materialised as torch
tensors.  Nothing under models/ is ever written.

The reference implementation follows the architecture declared by
config.json + the exact tensor shapes stored in model.safetensors:

  vision : ViT, patch 16x16, temporal patch 2, hidden 768, depth 12, 12 heads,
           intermediate 3072, GELU-tanh, learned 2-D pos-embed (2304 = 48x48),
           2x2 spatial merge -> 3072 -> 1024.
  text   : 24 layers, hidden 1024, FFN 3584 SwiGLU, 6 full-attention layers
           (GQA 8/2 heads, head_dim 256, partial RoPE 0.25, output gate) and
           18 gated-delta-net linear-attention layers (16 heads, k/v dim 128,
           causal depthwise conv k=4, A_log/dt_bias gate).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]          # repo root: .../CNN-CM
MODEL_DIR = ROOT / "models" / "Qwen3.5-0.8B"
SAFETENSORS = MODEL_DIR / "model.safetensors"
EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FINAL_JSON = EXP_DIR / "qwen3_5_0_8b_results.json"

# Paper-1 (CM-CNN) inputs.  Neither directory ships with this repository: the
# 224x224 DERF frames and the paper-1 result table belong to the companion study.
# Only the video-conditioned drivers (p2_stream.py, vlm_eval.py, bench_smolvlm.py)
# need them; tests do not.  Access them through require_paper1_*() so a missing
# copy produces an actionable error instead of a bare FileNotFoundError.
FRAMES_DIR = ROOT / "data" / "real_videos" / "frames_224"
PAPER1_RESULTS = ROOT / "experiments" / "cm-cnn-production" / "real_video_results.json"

_PAPER1_HINT = (
    "not shipped with this repository; see the 'Not in this repository' section of "
    "README.md (these are Paper-1 artefacts: DERF frames plus "
    "experiments/cm-cnn-production/real_video_results.json)"
)


def require_paper1_frames() -> Path:
    if not FRAMES_DIR.is_dir():
        raise FileNotFoundError(f"frame directory {FRAMES_DIR} is {_PAPER1_HINT}")
    return FRAMES_DIR


def require_paper1_results() -> Path:
    if not PAPER1_RESULTS.exists():
        raise FileNotFoundError(f"result table {PAPER1_RESULTS} is {_PAPER1_HINT}")
    return PAPER1_RESULTS

REGIMES = ["V1", "V2", "V3", "V4", "V5", "V6"]
REGIME_NAMES = {
    "V1": "static_surveillance",
    "V2": "local_motion",
    "V3": "fast_natural_object",
    "V4": "camera_pan",
    "V5": "illumination_change",
    "V6": "multi_shot_cuts",
}
CUT_FRAMES = [9, 18, 27]          # 1-indexed, Paper 1 protocol
LANE = 32                          # 1 warp / 2 AVX-512 ZMM registers
BAND = LANE                        # alias used by the FFN block-skipping code


# --------------------------------------------------------------------------- #
# config / weights
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    return json.loads((MODEL_DIR / "config.json").read_text())


class SafetensorsStore:
    """Minimal mmap reader for a single-file safetensors checkpoint."""

    def __init__(self, path: Path = SAFETENSORS):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        n = struct.unpack("<Q", self._f.read(8))[0]
        self.header = json.loads(self._f.read(n))
        self.data_start = 8 + n
        self._mmap = np.memmap(
            self.path, dtype=np.uint8, mode="r", offset=self.data_start
        )

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def info(self, name: str):
        h = self.header[name]
        return h["shape"], h["dtype"], h["data_offsets"]

    def get(self, name: str, dtype=torch.float32) -> torch.Tensor:
        shape, dt, (a, b) = self.info(name)
        raw = np.asarray(self._mmap[a:b])
        if dt == "F32":
            t = torch.from_numpy(raw.view(np.float32).copy())
        elif dt == "BF16":
            t = torch.from_numpy(raw.view(np.uint16).copy()).view(torch.bfloat16)
        elif dt == "F16":
            t = torch.from_numpy(raw.view(np.float16).copy())
        else:  # pragma: no cover
            raise ValueError(f"unsupported dtype {dt} for {name}")
        t = t.reshape(shape)
        return t.to(dtype).contiguous()


class Weights:
    """Float32 (default) or bf16 view of the checkpoint, cached by dtype."""

    def __init__(self, dtype=torch.float32, device="cpu", store: SafetensorsStore | None = None):
        self.dtype = dtype
        self.device = torch.device(device)
        self.store = store or SafetensorsStore()
        self._cache: dict[str, torch.Tensor] = {}

    def to(self, device, dtype=None):
        return Weights(dtype or self.dtype, device, store=self.store)

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self._cache:
            self._cache[name] = self.store.get(name, self.dtype).to(self.device)
        return self._cache[name]

    def has(self, name: str) -> bool:
        return name in self.store.header


# --------------------------------------------------------------------------- #
# small nn helpers (plain functions, weights passed explicitly)
# --------------------------------------------------------------------------- #
def _acc_dtype(dt):
    """Low-precision inputs are accumulated in fp32; fp32/fp64 keep their own width."""
    return torch.float32 if dt in (torch.float16, torch.bfloat16) else dt


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3.5 RMSNorm: the stored weight is zero-centred and applied as (1 + w)."""
    dt = x.dtype
    acc = _acc_dtype(dt)
    xf = x.to(acc)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + w.to(acc))).to(dt)


def gated_rms_norm(x: torch.Tensor, gate: torch.Tensor, w: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """Qwen3.5RMSNormGated (linear attention): plain weight, norm before the SiLU gate."""
    acc = _acc_dtype(x.dtype)
    xf = x.to(acc)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = y * w.to(acc)
    return (y * F.silu(gate.to(acc))).to(x.dtype)


def layer_norm(x, w, b, eps=1e-6):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def silu(x):
    return F.silu(x)


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


# --------------------------------------------------------------------------- #
# determinism / timing / io
# --------------------------------------------------------------------------- #
def set_deterministic(seed: int = 0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(False)   # kernels are deterministic here
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Timer:
    def __init__(self, device):
        self.device = torch.device(device)

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.dt_ms = (time.perf_counter() - self.t0) * 1e3


class Tokenizer:
    """Thin wrapper over the checkpoint's byte-level BPE tokenizer.json."""

    def __init__(self, model_dir=MODEL_DIR):
        from tokenizers import Tokenizer as _T
        self._t = _T.from_file(str(Path(model_dir) / "tokenizer.json"))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._t.encode(text).ids}

    def decode(self, ids):
        return self._t.decode(list(ids), skip_special_tokens=True)

    def encode(self, text):
        return self._t.encode(text).ids

    eos_token_id = 248044


def build_tokenizer(model_dir=MODEL_DIR):
    return Tokenizer(model_dir)


def bench(fn, device, warmup=2, reps=5):
    """Median wall-clock ms over `reps`, warm-up discarded."""
    for _ in range(warmup):
        fn()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        with Timer(device) as t:
            fn()
        ts.append(t.dt_ms)
    ts.sort()
    return float(ts[len(ts) // 2]), ts


def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dump_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False))


def log(*a):
    print(*a, flush=True)
