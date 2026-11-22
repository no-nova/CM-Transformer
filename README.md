# CM-Transformer

Code, evaluation snapshots and manuscript for **CM-Transform**: co-activation permuted
block sparsity with cache-line aligned execution, for transformer inference on commodity
GPUs and AVX-512 CPUs.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24435107.svg)](https://doi.org/10.5281/zenodo.24435107)

The question the project asks is narrow and practical: feed-forward blocks in modern
SwiGLU/GELU transformers have no exact-zero structure, so *any* compute saving has to
come from deciding which 32-lane blocks are worth evaluating — and then from making that
decision cheap enough that it survives contact with real memory traffic. The repository
contains the reference implementation of the mechanism, every kernel/benchmark driver
behind the paper, the raw result files, and the compiled manuscript.

**Manuscript:** [`paper2_bundle/CM_Transform_Paper2_Final.pdf`](paper2_bundle/CM_Transform_Paper2_Final.pdf)
(source: `paper2_bundle/CM_Transform_Paper2.typ`, Typst 0.13 — the host has no TeX).

---

## What the paper measures

On one consumer GPU (RTX 5070 Laptop, 8 GB) and one Zen 5 CPU (AVX-512), across five
real checkpoints — Qwen3-0.6B, Qwen3.5-0.8B, Qwen3.5-4B-IQ4_NL, whisper-base,
SmolVLM2-256M:

| Mechanism | Result |
|---|---|
| Functionally isomorphic 32-lane permutation of the FFN intermediate dimension | `τ = 0` reproduces the dense forward pass to `1e-7` |
| Per-layer adaptive energy threshold + norm-preserving rescaling, at a matched 20% skip budget on Qwen3.5-0.8B | WikiText-2 ΔPPL `+1.68` (6.2%), vs `+2.41` CATS-Block, `+2.30` TEAL-Block, `+13.17` Wanda-style block pruning |
| GSM8K under the same budget | `28.5%` vs `33.0%` dense (paired 95% CI −10.0 … +1.5 pp) — inside the interval |
| Block-major IQ4_NL repack (18-byte blocks inside 64-byte lines) | `1.74×` at 0% skip; `3.84–5.67×` at 20–80% skip on a cold single layer, `1.08–1.97×` for a 32-layer stream |
| Cross-modal transfer | whisper encoder keeps WER `6.21%` with 20% of blocks removed (`6.08%` dense) |
| Dispatch-cost accounting | fixes the boundary: at batch one on GPU the per-layer decision path costs about the `0.04 ms` GEMV it would save, so skipping pays only at operator level or under batching; on CPU and block-major 4.5-bit streams the same sparsity is a net speedup |

The paper also reports the byte ceilings that bound every decode-time number, and the
measurements where skipping *loses* to dense execution. The system claim is placed on
commodity multi-core CPU execution and on the block-major 4.5-bit layout, not on
batch-one GPU decode.

---

## Repository layout

| Path | Contents |
|---|---|
| `paper2_bundle/` | manuscript source, compiled PDF, vector figures (`figures/*.svg`), section fragments, SHA-256 delivery manifest |
| `experiments/cm-transformer/` | reference model, kernels (CUDA / AVX-512 / scalar), benchmark drivers, tests, result JSONs, `REPORT_TEXT.md` |
| `data/benchmarks/` | pinned evaluation splits: WikiText-2, ARC-Easy, BoolQ, PIQA, GSM8K, POPE (read-only, with `manifest.json`) |
| `data/p6_vlm/` | synthetic multimodal probe images, regenerable with `experiments/cm-transformer/make_p6_vlm.py` |
| `scripts/download_benchmarks.py` | regenerates `data/benchmarks/` from the HuggingFace datasets-server |
| `scripts/fetch_models.py` | downloads the five third-party checkpoints and verifies their SHA-256 (see [`MODELS.md`](MODELS.md)) |
| `scripts/verify.sh` | single verification entry point: dependencies → checkpoint digests → tests → reference-vs-official → artefact digests |
| `scripts/check_artefacts.py` | re-hashes everything listed in `paper2_bundle/hashes.json` |

`experiments/cm-transformer/REPORT_TEXT.md` is the long-form lab report: it states, for
every table in the paper, which script produced the numbers and which result file holds
them.

---

## Reproduce

```bash
# 0. environment
pip install -r requirements.txt        # CPU verification path (numpy, pytest, torch)

# 1. checkpoints (~7.1 GB, third-party; not shipped)
python3 scripts/fetch_models.py        # or: --check to verify a copy you already have

# 2. everything else, in one command
bash scripts/verify.sh                 # deps -> digests -> 77 tests -> official-forward -> manifest
bash scripts/verify.sh --fast          # skip the minutes-long official-forward check
bash scripts/verify.sh --no-model      # no checkpoints needed: data/manifest checks only
```

What each stage means:

| Stage | Command it runs | Passes when |
|---|---|---|
| dependencies | import check | `numpy`, `pytest`, `torch` importable, Python ≥ 3.10 |
| checkpoints | `scripts/fetch_models.py --check` | five weight files match the sizes/SHA-256 in `MODELS.md` |
| tests | `pytest tests/ -q` | 77 checks: numerical invariants, kernel gates, paper-source consistency |
| reference | `verify_official.py` | the in-repo reference forward matches official `transformers` (per-layer `max|Δ| ≤ 1.4e-3`, `cos = 1.000000`) |
| artefacts | `scripts/check_artefacts.py` | every digest in `paper2_bundle/hashes.json` still matches |

Without `models/`, the model-dependent modules are skipped with a message naming the
missing file (`pytest tests/ -q` → `33 passed, 10 skipped`), so a clone is usable before
the 7 GB download finishes.

```bash
# figures
cd experiments/cm-transformer
python make_paper2_figs.py && python make_paper2_fig6.py
python make_paper2_p8.py          # regenerates ../paper2_bundle/p8_section.typ

# manuscript
cd ../../paper2_bundle && typst compile CM_Transform_Paper2.typ CM_Transform_Paper2_Final.pdf
```

Model-dependent drivers expect the checkpoints under `models/` (see below) and are run
one at a time; each writes a JSON snapshot into `experiments/cm-transformer/results/`
that the paper tables are generated from. Hash-verified artefacts are listed in
`paper2_bundle/DELIVERY_MANIFEST.md` and `paper2_bundle/hashes.json`.

**Environment used for the measurements:** Python 3.12, PyTorch 2.15.0 (CUDA 13.5,
sm_120), `transformers` 5.23.0; kernels built with `nvcc` and
`g++ -O3 -march=native -mavx512f -mavx512bw -mfma -fopenmp`. CPU timings assume
`OMP_WAIT_POLICY=passive` and single-socket pinning.

---

## Not in this repository

Third-party assets are **not** redistributed: model weights and LibriSpeech audio stay
under their own licences. The five checkpoints the study consumes are pinned in
[`MODELS.md`](MODELS.md) — repository, file name, byte size and SHA-256 — and
`scripts/fetch_models.py` downloads them into `models/` and verifies both size and
digest before the files are used.

Also absent: the real-video frames and the result table from the companion CM-CNN study
(`data/real_videos/`, `experiments/cm-cnn-production/`). Only the video-conditioned
drivers (`p2_stream.py`, `vlm_eval.py`, `bench_smolvlm.py`) need them, and they raise a
message pointing at this section instead of a bare `FileNotFoundError`.

Evaluation splits in `data/benchmarks/` keep the licences of their source datasets; see
`data/benchmarks/README.md`.

Third-party weights are memory-mapped read-only and no script writes under `models/`;
`tests/test_integrity.py` re-checks the checkpoint digest recorded in the baseline run.

---

## Note on the manuscript link

The compiled PDF still carries the URL of the repository where the code lived while the
paper was written (`github.com/no-nova/CM-CNN`). This repository is the canonical home of
the CM-Transform code and results; the manuscript source and PDF are byte-identical to
the submitted version, and `DELIVERY_MANIFEST.md` pins their digests.

## Licence

Code and manuscript: see [`LICENSE`](LICENSE) (Apache-2.0).

Release metadata for DOI minting (title, description, creators, keywords, and the
list of third-party assets that are not redistributed) lives in
[`.zenodo.json`](.zenodo.json); Zenodo reads it when a GitHub release is archived.
