# Paper 2 bundle — CM-Transform (text LLM inference acceleration)

Final manuscript: `CM_Transform_Paper2_Final.pdf` (20 pages, vector figures).

## Contents

| File | What it is |
|---|---|
| `CM_Transform_Paper2.typ` | complete manuscript source (Typst 0.13) |
| `CM_Transform_Paper2_Final.pdf` | compiled PDF (letter, 20 pages) |
| `p8_section.typ`, `extra_section.typ` | generated sections (defence experiments, kernel microbenchmarks), spliced into the source |
| `figures/fig1_ppl_tau.svg` … `fig6_frontier.svg` | all six figures, vector SVG |

## Rebuild

```sh
# figures (needs matplotlib from /home/user/.venvs/qwen5)
cd ../experiments/cm-transformer
python make_paper2_figs.py
python make_paper2_fig6.py
python make_paper2_p8.py          # regenerates ../paper2_bundle/p8_section.typ
# manuscript
cd ../../paper2_bundle
typst compile CM_Transform_Paper2.typ CM_Transform_Paper2_Final.pdf
```

Typst is used because this host has no TeX distribution (`pdflatex`, `xelatex`,
`latexmk` absent); the output is a vector PDF and the source is plain markup.

## Where every number comes from

| Script (in `../experiments/cm-transformer/`) | Artefacts | Used for |
|---|---|---|
| `p6_sota.py` | `results/p6_sota_*.json`, `results/p6_sota_gsm*.json` | method comparison, ladder, GSM8K CIs |
| `p1_v2.py`, `p1_gsm.py`, `p1_gen.py` | `results/p1v2_*.json` | PPL sweeps, generation stability |
| `p1_kernels_v2.py` | `results/p1_kernels_v2*.json` | CUDA v4/v7, CPU token- vs row-parallel, end-to-end |
| `bench_p4_iq4.py`, `iq4nl_blockmajor.cpp` | `results/p4_iq4_blockmajor.json` | block-major layout, 8-row kernel, coupled stream |
| `bench_p5_iq4.py` | `results/p5_iq4_blockmajor.json` | gate pre-filter |
| `bench_p6_iq4_real.py` | `results/p6_iq4_real.json` | real 4B activations |
| `bench_whisper_real.py`, `bench_p6_whisper_ffn.py` | `results/p6_whisper*.json` | WER/CER, encoder FFN isolation |
| `bench_p6_smolvlm.py` | `results/p6_smolvlm_50.json` | 52-sample multimodal eval |
| `p8_reviewer_defense.py` | `results/p8_perm*.json`, `p8_factorial.json`, `p8_calib*.json`, `p8_quant*.json` | ordering/budget factorial, calibration robustness, IQ4-NL simulation |
| `p8_system.py` | `results/p8_system.json` | batch/context sweep, end-to-end ladder |

All JSON files carry `seed: 1234`. Datasets are pinned by SHA-256 in
`../data/benchmarks/manifest.json` (GSM8K-200 subset:
`72813874f21da675e729ba2d8de858ff001ed23f062e6c963bbf51b4a044786f`).
`models/` is read-only; `git diff -- src/lib/codec.ts` is empty.

## Rebuild commands (development notes — not part of the manuscript)

```sh
python p6_sota.py --model both --device cuda --mc-limit 200 --windows 16
python p6_sota.py --stage gsm --skip-sweep --gsm-limit 200 --gsm-new 128
python bench_p4_iq4.py --threads 32 && python bench_p6_iq4_real.py --tokens 64
python bench_p6_whisper_ffn.py && python bench_whisper_real.py --limit 32
python bench_p6_smolvlm.py --device cuda --limit 52
python p8_reviewer_defense.py --task perm && python p8_system.py --device cuda
python scripts/p10_4b_closure.py --task bridge
python scripts/p10_extras.py --task genprompt,calibtail,corr
python scripts/p10_width.py --reps 5
pytest tests/ -q          # 75 tests
```

Typesetting: no TeX distribution is installed on the build host (pdflatex, xelatex
and latexmk are absent), so the manuscript is written in Typst 0.13:

```sh
typst compile CM_Transform_Paper2.typ CM_Transform_Paper2_Final.pdf
```
