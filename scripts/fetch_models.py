#!/usr/bin/env python3
"""Fetch (or verify) the third-party checkpoints this study consumes.

    python3 scripts/fetch_models.py                 # download all five
    python3 scripts/fetch_models.py --only whisper_base,qwen35_4b_gguf
    python3 scripts/fetch_models.py --check         # verify only, no network
    python3 scripts/fetch_models.py --check --only qwen35_08b

Every weight file is checked against the size and SHA-256 recorded in
``MODELS.md`` before it is moved into place, so the directory layout the
experiments expect is always the one the published numbers were produced from.

Nothing is written inside an existing model directory except the files listed
for it; a mismatching download is left as ``<name>.part`` for inspection and the
process exits non-zero.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CHUNK = 1 << 22

# key -> (repo, target dir, weight files, metadata files)
# weight file tuple: (name in repo, name on disk, bytes, sha256)
SPECS: dict[str, dict] = {
    "qwen35_08b": {
        "repo": "Qwen/Qwen3.5-0.8B",
        "dir": "Qwen3.5-0.8B",
        "weights": [
            ("model.safetensors-00001-of-00001.safetensors", "model.safetensors",
             1_746_942_600,
             "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"),
        ],
        "meta": ["config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt",
                 "preprocessor_config.json", "video_preprocessor_config.json"],
    },
    "qwen3_06b": {
        "repo": "Qwen/Qwen3-0.6B",
        "dir": "Qwen3-0.6B",
        "weights": [
            ("model.safetensors", "model.safetensors", 1_503_300_328,
             "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"),
        ],
        "meta": ["config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt"],
    },
    "qwen35_4b_gguf": {
        "repo": "unsloth/Qwen3.5-4B-GGUF",
        "dir": None,                                  # single file at models/
        "weights": [
            ("Qwen3.5-4B-IQ4_NL.gguf", "Qwen3.5-4B-IQ4_NL.gguf", 2_579_944_608,
             "ff5c3e9740a5aa53f04fdf3b0b8cc75da556bf8948cdb19d61c512d3a43465d9"),
        ],
        "meta": [],
    },
    "smolvlm2": {
        "repo": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "dir": "SmolVLM2-256M-Video-Instruct",
        "weights": [
            ("model.safetensors", "model.safetensors", 1_025_998_224,
             "dcc8f4c00d30d70bb3028a1bfc373e1b1972e25919f8788183cae4ae72431ce2"),
        ],
        "meta": ["config.json", "generation_config.json", "chat_template.json",
                 "processor_config.json", "preprocessor_config.json",
                 "special_tokens_map.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "vocab", "merges.txt"],
    },
    "whisper_base": {
        "repo": "openai/whisper-base",
        "dir": "whisper-base",
        "weights": [
            ("model.safetensors", "model.safetensors", 290_403_936,
             "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b"),
        ],
        "meta": ["config.json", "generation_config.json",
                 "preprocessor_config.json", "normalizer.json",
                 "added_tokens.json", "special_tokens_map.json",
                 "tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt"],
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def target_path(spec: dict, name_on_disk: str) -> Path:
    return (MODELS / name_on_disk) if spec["dir"] is None else (MODELS / spec["dir"] / name_on_disk)


def url_for(repo: str, name: str) -> str:
    return f"{ENDPOINT}/{repo}/resolve/main/{name}"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def verify(spec: dict, quiet: bool = False) -> tuple[bool, list[str]]:
    """Check the weight files of one model. Returns (ok, messages)."""
    msgs, ok = [], True
    for _name_in_repo, name_on_disk, size, digest in spec["weights"]:
        p = target_path(spec, name_on_disk)
        if not p.exists():
            msgs.append(f"MISSING  {p.relative_to(ROOT)}")
            ok = False
            continue
        got_size = p.stat().st_size
        if got_size != size:
            msgs.append(f"BAD SIZE {p.relative_to(ROOT)}: {got_size:,} != {size:,}")
            ok = False
            continue
        got = sha256_of(p)
        if got != digest:
            msgs.append(f"BAD SHA  {p.relative_to(ROOT)}: {got[:16]}… != {digest[:16]}…")
            ok = False
            continue
        msgs.append(f"OK       {p.relative_to(ROOT)}  {human(size)}  {digest[:16]}…")
    if not quiet:
        for m in msgs:
            print("   " + m)
    return ok, msgs


def download(url: str, dest: Path, expected_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists() and part.stat().st_size == expected_size:
        pass                                       # finished earlier, verify below
    else:
        resume = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url)
        if resume:
            req.add_header("Range", f"bytes={resume}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab") as fh:
                done = resume
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if expected_size:
                        pct = 100.0 * done / expected_size
                        sys.stderr.write(f"\r      {pct:5.1f}%  {human(done)} / {human(expected_size)}")
                        sys.stderr.flush()
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"download failed ({exc.code}) for {url}") from exc
        sys.stderr.write("\r" + " " * 60 + "\r")
    os.replace(part, dest)


def fetch(key: str, spec: dict, use_hub: bool) -> bool:
    label = spec["dir"] or spec["weights"][0][1]
    print(f"[{key}] {label}  <- {spec['repo']}")
    ok, msgs = verify(spec, quiet=True)
    if ok:
        print("   already present and verified")
        return True
    if use_hub:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            use_hub = False
        else:
            dest = MODELS if spec["dir"] is None else MODELS / spec["dir"]
            names = [w[0] for w in spec["weights"]] + list(spec["meta"])
            snapshot_download(spec["repo"], local_dir=str(dest),
                              allow_patterns=names)
            for name_in_repo, name_on_disk, _s, _d in spec["weights"]:
                if name_in_repo != name_on_disk:
                    src = dest / name_in_repo
                    if src.exists():
                        os.replace(src, dest / name_on_disk)
            return verify(spec)[0]
    if not use_hub:
        for name_in_repo, name_on_disk, size, _digest in spec["weights"]:
            dest = target_path(spec, name_on_disk)
            if dest.exists() and dest.stat().st_size == size:
                continue
            print(f"   downloading {name_in_repo} ({human(size)})")
            download(url_for(spec["repo"], name_in_repo), dest, size)
        for name in spec["meta"]:
            dest = target_path(spec, name)
            if dest.exists():
                continue
            try:
                download(url_for(spec["repo"], name), dest, 0)
            except SystemExit:
                print(f"   warning: metadata file {name} not available upstream")
    return verify(spec)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ", ".join(SPECS))
    ap.add_argument("--check", action="store_true", help="verify digests, do not download")
    ap.add_argument("--no-hub", action="store_true", help="always use plain HTTPS")
    ap.add_argument("--dest", default="", help="override the models/ directory")
    args = ap.parse_args()

    global MODELS
    if args.dest:
        MODELS = Path(args.dest).expanduser().resolve()

    keys = [k.strip() for k in args.only.split(",") if k.strip()] or list(SPECS)
    unknown = [k for k in keys if k not in SPECS]
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}; choose from {', '.join(SPECS)}")

    print(f"models directory: {MODELS}")
    failures = []
    for key in keys:
        if args.check:
            ok, _ = verify(SPECS[key])
        else:
            ok = fetch(key, SPECS[key], use_hub=not args.no_hub)
        if not ok:
            failures.append(key)

    if failures:
        print(f"\n{len(failures)} model(s) failed: {', '.join(failures)}")
        print("See MODELS.md for the expected digests and the two known source mismatches.")
        return 1
    print(f"\nall {len(keys)} model(s) verified against MODELS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
