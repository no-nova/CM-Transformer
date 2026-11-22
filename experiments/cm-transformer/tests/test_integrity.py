"""Integrity guards for the study: the shipped repo code and the weights are
never modified by any experiment."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(EXP))

from common import MODEL_DIR, RESULTS_DIR, SAFETENSORS  # noqa: E402


def git(*args, cwd=REPO):
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError:      # pragma: no cover
        pytest.skip("git not available")
    if out.returncode != 0 and "not a git repository" in out.stderr:
        pytest.skip("not a git repository")
    return out


def test_codec_ts_is_untouched():
    out = git("diff", "--", "src/lib/codec.ts")
    assert out.stdout.strip() == ""


def test_no_tracked_source_file_was_modified_by_the_experiment():
    out = git("status", "--porcelain")
    changed = [l[3:] for l in out.stdout.splitlines() if l[:2].strip() in {"M", "MM", "AM"}]
    # pre-existing Paper-1 work (native kernel, its result files and the
    # regenerated paper bundle); nothing in src/ or models/ may change.
    allowed_prefix = ("native/cm_production_avx512.cpp",
                      "experiments/cm-cnn-production/",
                      "experiments/cm-transformer/results/",   # driver outputs
                      "file/cm_cnn_accel_paper/",
                      "paper1_bundle/")
    unexpected = [c for c in changed if not c.startswith(allowed_prefix)]
    assert unexpected == [], unexpected


def test_model_weights_are_byte_identical():
    baseline = RESULTS_DIR / "m0_baseline.json"
    if not baseline.exists():
        pytest.skip("no baseline run recorded")
    if not SAFETENSORS.exists():
        pytest.skip("checkpoint not present; see MODELS.md")
    rec = json.loads(baseline.read_text())["arch"]["checkpoint"]
    sha = rec.get("sha256") or rec.get("model_sha256")
    if not sha:
        pytest.skip("baseline does not record a checkpoint hash")
    h = hashlib.sha256()
    with open(SAFETENSORS, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    assert h.hexdigest() == sha


def test_models_directory_is_read_only_in_code():
    """The loader must not open weights for writing."""
    src = (EXP / "common.py").read_text()
    assert 'mode="r"' in src                     # np.memmap read-only
    assert "mode=\"w" not in src and "mode='w" not in src


def test_no_stray_weight_files_in_models():
    if not MODEL_DIR.is_dir():
        pytest.skip("checkpoint directory not present; see MODELS.md")
    allowed = {p.name for p in MODEL_DIR.iterdir()}
    assert "model.safetensors" in allowed
    for name in allowed:
        assert not name.endswith((".tmp", ".part", ".bak"))
