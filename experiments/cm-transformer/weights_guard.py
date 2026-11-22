"""Skip model-dependent tests when the third-party checkpoints are absent.

The checkpoints are not redistributed with this repository (see ``MODELS.md``).
Importing this module from a test file and calling :func:`require` marks the
whole module as skipped — with an actionable message — instead of failing during
collection with a bare ``FileNotFoundError``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]          # repository root
MODELS = ROOT / "models"

CHECKPOINTS = {
    "qwen35_08b": (MODELS / "Qwen3.5-0.8B" / "config.json",
                   MODELS / "Qwen3.5-0.8B" / "model.safetensors"),
    "qwen3_06b": (MODELS / "Qwen3-0.6B" / "model.safetensors",),
    "qwen35_4b_gguf": (MODELS / "Qwen3.5-4B-IQ4_NL.gguf",),
    "smolvlm2": (MODELS / "SmolVLM2-256M-Video-Instruct" / "model.safetensors",),
    "whisper_base": (MODELS / "whisper-base" / "model.safetensors",),
}


def missing(*keys: str) -> list[Path]:
    keys = keys or ("qwen35_08b",)
    return [p for k in keys for p in CHECKPOINTS[k] if not p.exists()]


def have(*keys: str) -> bool:
    return not missing(*keys)


def require(*keys: str) -> None:
    """Skip the calling module if any required checkpoint file is missing."""
    gone = missing(*keys)
    if gone:
        rel = ", ".join(str(p.relative_to(ROOT)) for p in gone)
        pytest.skip(
            f"checkpoint not found ({rel}); run `python3 scripts/fetch_models.py` "
            "or see MODELS.md",
            allow_module_level=True,
        )
