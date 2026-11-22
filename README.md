# CM-Transformer

Code, evaluation snapshots and manuscript for **CM-Transform**: co-activation permuted
block sparsity with cache-line aligned execution, for transformer inference on commodity
GPUs and AVX-512 CPUs.

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

`experiments/cm-transformer/REPORT_TEXT.md` is the long-form lab report: it states, for
every table in the paper, which script produced the numbers and which result file holds
them.

---

## Reproduce

```bash
# 1. correctness gates (numerical invariants, paper-source checks)
cd experiments/cm-transformer && python3 -m pytest tests/ -q

# 2. evaluation splits (already committed; this only verifies/regenerates them)
python3 scripts/download_benchmarks.py --out data/benchmarks

# 3. figures
cd experiments/cm-transformer
python make_paper2_figs.py && python make_paper2_fig6.py
python make_paper2_p8.py          # regenerates ../paper2_bundle/p8_section.typ

# 4. manuscript
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

Third-party assets are **not** redistributed: model weights (Qwen3-0.6B, Qwen3.5-0.8B,
Qwen3.5-4B-IQ4_NL, whisper-base, SmolVLM2-256M) and LibriSpeech audio stay under their
own licences and must be obtained from their publishers. `paper2_bundle/hashes.json`
records the digests of the checkpoints actually used, so a reproduction can confirm it
holds the same bytes. Compiled kernels (`*.so`, `results/build/`) and cached tensors
(`*.pt`) are build products and are ignored as well — every one of them is regenerated
by a script in `experiments/cm-transformer/`.

Evaluation splits in `data/benchmarks/` keep the licences of their source datasets; see
`data/benchmarks/README.md`.

---

## Note on the manuscript link

The compiled PDF still carries the URL of the repository where the code lived while the
paper was written (`github.com/no-nova/CM-CNN`). This repository is the canonical home of
the CM-Transform code and results; the manuscript source and PDF are byte-identical to
the submitted version, and `DELIVERY_MANIFEST.md` pins their digests.

## Licence

Code and manuscript: see [`LICENSE`](LICENSE).
