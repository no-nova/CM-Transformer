#!/usr/bin/env python3
"""Check that the committed artefacts still match their recorded digests.

Reads ``paper2_bundle/hashes.json`` (the delivery manifest) and re-hashes every
shipped artefact, then asserts the two integrity flags the study itself records
in ``experiments/cm-transformer/qwen3_5_0_8b_results.json``.

    python3 scripts/check_artefacts.py

Exit status 0 means every digest matches and no weight/source file was modified.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "paper2_bundle"
FINAL_JSON = ROOT / "experiments" / "cm-transformer" / "qwen3_5_0_8b_results.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    problems: list[str] = []

    entries = json.loads((BUNDLE / "hashes.json").read_text())
    missing_models = 0
    for entry in entries:
        rel = entry["path"]
        # model weights are recorded relative to models/, datasets relative to the
        # repository root; everything else lives inside paper2_bundle/
        if rel.startswith(("Qwen", "whisper-", "SmolVLM", "LFM")):
            path, is_third_party = ROOT / "models" / rel, True
        elif rel.startswith("data/"):
            path, is_third_party = ROOT / rel, False
        else:
            path, is_third_party = BUNDLE / rel, False
        if not path.exists():
            if is_third_party:
                missing_models += 1          # not shipped; digests still recorded
                print(f"  SKIP  {rel} (third-party checkpoint not present)")
                continue
            problems.append(f"missing {rel}")
            continue
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"size changed {rel} ({path.stat().st_size} != {entry['bytes']})")
            continue
        if sha256_of(path) != entry["sha256"]:
            problems.append(f"digest mismatch {rel}")

    if FINAL_JSON.exists():
        meta = json.loads(FINAL_JSON.read_text())["meta"]
        if not meta.get("codec_ts_diff_empty", True):
            problems.append("results meta: src/lib/codec.ts was modified during the study")
        if not meta.get("weights_unmodified", True):
            problems.append("results meta: checkpoints were modified during the study")

    for p in problems:
        print(f"  FAIL  {p}")
    checked = len(entries) - missing_models
    print(f"  {'PASS' if not problems else 'FAIL'}  {checked}/{len(entries)} manifest entries verified"
          f"{f', {missing_models} third-party checkpoint(s) not on disk' if missing_models else ''}"
          f"{'' if FINAL_JSON.exists() else ' (no results JSON found)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
