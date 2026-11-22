# CM-Transform

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24435107.svg)](https://doi.org/10.5281/zenodo.24435107)
[![release](https://img.shields.io/badge/release-paper2--v1.0.0-blue)](https://github.com/no-nova/CM-Transformer/releases/tag/paper2-v1.0.0)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-green)](LICENSE)

**Co-activation permuted block sparsity and cache-line aligned execution for transformer
inference on commodity hardware.** Technical report 2, version 1.0.0, November 2026 ·
[manuscript (PDF, 20 pages)](paper2_bundle/CM_Transform_Paper2_Final.pdf) ·
[archived at Zenodo](https://doi.org/10.5281/zenodo.24435107)

Reference implementation, kernels, drivers and every result file behind the paper.

---

## The question

A forward-feed block in a modern SwiGLU/GELU transformer has no exact-zero structure, so every
compute saving has to start from a decision: which 32-lane blocks of the intermediate dimension
are worth evaluating. Two things usually break that decision in practice — the *ordering* (a
truncated natural-order vector is not the smallest such vector) and the *memory layout* (a
4.5-bit block of 18 bytes shares a 64-byte cache line with 2.5 neighbours, so dropping it saves
nothing in DRAM traffic).

This repository contains the mechanism that addresses both, the kernels that make it survive
contact with real memory traffic, and the measurements — including the ones where it does not
pay off.

## Headline numbers

Dense baselines and ΔPPL are WikiText-2 perplexity over 16×512-token windows, fp32, seed 1234;
downstream figures are ARC-Easy / PIQA / BoolQ / GSM8K. Every value below is a file in
[`experiments/cm-transformer/results/`](experiments/cm-transformer/results).

| Result | Value | File |
|---|---|---|
| Dense PPL, Qwen3.5-0.8B / Qwen3-0.6B | 27.17 / 32.37 | `p6_sota_qwen3_5-0_8b.json`, `p6_sota_qwen3-0_6b.json` |
| Isomorphism at τ = 0 (permuted vs original) | `max|Δlogit|` 1.1e-7, Top-1 1.000, no retraining | `p8_factorial.json`, `verify_official.json` |
| ΔPPL vs named baselines, Qwen3.5-0.8B, τ = 0.10 / 0.20 | **CATS-Block 0.93 / 2.41** · TEAL-Block 0.71 / 2.30 · Wanda-2:4 5.32 / 13.17 · **CM-Transform uncoupled 0.70 / 1.68** | `p6_sota_qwen3_5-0_8b.json` |
| ΔPPL vs named baselines, Qwen3-0.6B, τ = 0.10 / 0.20 | CATS 0.28 / 1.50 · TEAL 0.49 / 2.08 · Wanda 9.10 / 26.37 · **CM-Transform 0.09 / 0.88** | `p6_sota_qwen3-0_6b.json` |
| Ablation ladder (0.8B, τ = 0.20) | S0 native+Top-K 2.30 → **S1 +π 3.18** → S2 +adaptive budget 1.85 → S3 +norm rescaling 1.68 → S4 +static weights 2.12 → S5 +coupled 4.46 | `p6_sota_qwen3_5-0_8b.json` |
| Calibration leak check | 4 corpora × 3 seeds, kept-block Jaccard 0.94–1.00; cross-corpus ΔPPL +0.48 ± 0.31 | `p9_calib2.json`, `p10_calibtail.json` |
| Block-major IQ4_NL, single `ffn_down` (cold, 32 threads) | **0.534 ms at τ = 0** vs 0.930 ms row-major dense → **1.74×**; 3.85× / 3.86× / 4.61× / **5.68×** at τ = 0.2 / 0.4 / 0.6 / 0.8 | `p4_iq4_blockmajor.json` |
| Block-major IQ4_NL, 32-layer coupled FFN stream | 43.1 ms → 40.0 / 37.0 / 32.0 / **21.9 ms** (1.08 / 1.17 / 1.34 / **1.97×**) | `p4_iq4_blockmajor.json` |
| Cache-line traffic becomes linear in the skip rate | 720 whole lines per 32-lane block vs 816-byte shared blocks | `p9_cacheline.json` |
| Real activations from the 4B GGUF (residual stream, 64 tokens) | τ = 0.20: energy captured 0.934, rel-L2 0.256, cosine 0.966, **3.34×**; τ = 0.80: 0.447 / 0.742 / 0.669, **6.70×** | `p6_iq4_real.json` |
| CPU thread scaling (AVX-512, `OMP_WAIT_POLICY=passive`) | 1 row: 0.870 ms → 0.148 ms at 8 threads; sparse loses below dense at every width | `p10_threads_passive.json` |
| SIMD-width control (scalar / AVX2 / AVX-512) | within 5%, dense within 1.5% — the layout, not the ISA width, moves the time | `p10_width.json` |
| Why quality and the gate correlate | Spearman(‖g_b‖₂, ‖a_b‖₂) = 0.976; weight-block norms are near-uniform (CV 0.05) | `p9_corr.json` |
| Generation stability (64 prompts, prompt as unit) | token agreement 0.637 [0.531, 0.726], first divergence 18.7 [15.2, 21.8] steps | `p10_genprompt.json` |
| FFN error → model error | ΔPPL ≈ 18.4·relL2² − 91.6, R² = 0.995 (0.8B); 115.0·relL2² − 0.30, R² = 0.990 (0.6B) | `p10_bridge.json` |
| Whisper-base, 60 LibriSpeech utterances | WER 6.08% → 6.21% (τ = 0.20) → 6.98% (τ = 0.40); CER 2.79% → 3.05% | `p9_whisper100.json` |
| SmolVLM2, 52 samples | accuracy 0.279 dense → 0.191 (language tower skipped, τ = 0.20) → 0.309 (vision tower) | `p6_smolvlm_50.json` |

Where the mechanism does *not* pay: on the 8 GB laptop GPU a single decode token is bounded by
weight bytes (the whole FFN of Qwen3.5-0.8B is 63.9% of the stream), so end-to-end decode is
0.85–0.94× there; the system claim is anchored on AVX-512 CPU decode and on the quantised
block-major path. The manuscript states this in §5.5 and §7.

## What is in here

| Path | Contents |
|---|---|
| [`paper2_bundle/`](paper2_bundle) | Typst source, compiled PDF, vector figures, digests |
| [`experiments/cm-transformer/`](experiments/cm-transformer) | reference implementation (`model_ref.py`, 24-layer Qwen3.5 backbone plus the 12-layer vision tower), CUDA / AVX-512 / scalar kernels, benchmark drivers, 77-test suite |
| [`experiments/cm-transformer/results/`](experiments/cm-transformer/results) | 87 result files — one per experiment or per model/device pair |
| [`experiments/cm-transformer/REPORT_TEXT.md`](experiments/cm-transformer/REPORT_TEXT.md) | long-form report with the tables the paper draws on |
| [`data/benchmarks/`](data/benchmarks) | fixed evaluation splits (WikiText-2, ARC-Easy, PIQA, BoolQ, GSM8K-200, POPE) with manifest and licences |
| [`MODELS.md`](MODELS.md) · [`scripts/fetch_models.py`](scripts/fetch_models.py) | checkpoint provenance, byte sizes, SHA-256, fetch/verify |
| [`scripts/verify.sh`](scripts/verify.sh) | one-command verification of a fresh clone |

Models used: Qwen3-0.6B, Qwen3.5-0.8B (text backbone), Qwen3.5-4B-IQ4_NL (GGUF, 4.5-bit),
whisper-base, SmolVLM2-256M-Video-Instruct.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy, pytest, torch 2.15.0

python scripts/fetch_models.py           # downloads the five checkpoints, verifies digests
bash scripts/verify.sh                   # digest check, 77 tests, reference vs official forward
```

Single-core CPU numbers need `OMP_NUM_THREADS=1`; the threaded CPU numbers are measured with
`OMP_WAIT_POLICY=passive` (the default busy-wait costs 2.8 ms per fork/join against a 0.09 ms
operator, which is why the paper reports it explicitly).

Reproducing one table end-to-end looks like this:

```bash
cd experiments/cm-transformer
python p6_sota.py --model qwen3.5-0.8b --device cuda --stage sweep   # named baselines + ladder
python bench_p4_iq4.py --threads 32                                  # block-major IQ4_NL
python bench_p6_iq4_real.py                                          # 4B real activations
python bench_whisper_real.py --limit 60                              # speech
```

## How the numbers were produced

- **Protocol.** Perplexity is teacher-forced cross-entropy over 16 windows of 512 tokens;
  multiple-choice benchmarks use lm-evaluation-harness semantics (the mean log-probability of
  each choice's own tokens), not single-token scoring; accuracies carry 1000-sample bootstrap
  intervals with the item as the unit.
- **Leakage.** Calibration uses WikiText-2 rows 0–120, evaluation rows 300–420; the harness
  asserts the slices are disjoint and that a 64-token probe never collides.
- **Hardware.** RTX 5070 Laptop (8 GB, 279 GB/s measured) and a Zen 5 core (AVX-512F/BW/DL/VNNI)
  on a Ryzen 9 9955HX, `-O3 -march=native -mavx512f -mavx512bw -mfma`, g++ 16.2.1.
- **Seeds and digests.** Every result file records its seed (1234) and, where data is involved,
  the SHA-256 of the input split.

## Scope

The mechanism is evaluated as *sparsity on top of a dense kernel*: it decides which blocks to
evaluate, so its gain is bounded by the fraction of bytes those blocks carry. It does not
address attention, KV cache growth, or batching. Training is not modified, and the permutation
is derived from calibration data rather than learned.

Companion report: **CM-CNN-Accel** (paper 1), spatial and temporal computation skipping for CNN
inference — [github.com/no-nova/CM-CNN](https://github.com/no-nova/CM-CNN),
[doi:10.5281/zenodo.24435068](https://doi.org/10.5281/zenodo.24435068).

## Citation

```
CM-Transform: Co-Activation Permuted Block Sparsity and Cache-Line Aligned Execution for
Transformer Inference on Commodity Hardware. Technical Report 2, version 1.0.0, November 2026.
doi:10.5281/zenodo.24435107
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff) · licence: [Apache-2.0](LICENSE).
Model weights and LibriSpeech audio are not redistributed.
