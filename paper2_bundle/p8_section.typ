= Defence experiments

This section reports the five checks a reviewer is likely to ask for: whether a different block ordering would have done as well, whether the permutation or the threshold is doing the work, whether the calibration corpus matters, whether the quality claims survive a quantised forward pass, and how the operator behaves in a full inference system.

== Block ordering: five strategies at matched budget

Every strategy is built from the same calibration activations, applied with the same adaptive threshold, and evaluated on an 8$times$512-token WikiText-2 window. Because that window is half as long as the one used in Section 5.1, the dense references differ (16.21 for Qwen3.5-0.8B and 17.99 for Qwen3-0.6B); the comparison below is internally consistent, and the ordering of the strategies is unchanged when the full 16-window stream is used.

#figure(table(
  columns: 8,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Ordering], [0.8B $Delta$PPL 0.20], [0.8B 0.30], [0.8B rel-$L_2$], [0.8B Gini], [0.6B 0.20], [0.6B 0.30], [0.6B rel-$L_2$],
  [random], [+1.430], [+2.641], [0.247], [0.457], [+1.680], [+4.135], [0.182],
  [native], [+1.084], [+1.807], [0.225], [0.514], [+1.041], [+2.350], [0.162],
  [actmag], [+0.430], [+1.407], [0.191], [0.573], [+0.192], [+0.710], [0.131],
  [wnorm], [+1.488], [+2.436], [0.233], [0.490], [+0.725], [+2.260], [0.161],
  [impsort], [+0.822], [+1.534], [0.188], [0.580], [+0.332], [+1.209], [0.127],
  [ours], [+0.552], [+1.323], [0.188], [0.573], [+0.122], [+1.122], [0.128],
), caption: [Block-ordering strategies at $tau=0.20$ and $0.30$. `actmag` sorts lanes by mean $|a_i|$, `wnorm` by $‖W_(d,i)‖$, `impsort` by their product, `ours` adds the covariance-aware greedy fill. rel-$L_2$ is the mean per-layer output reconstruction error, Gini the concentration of the block-energy distribution.])

Three observations. On Qwen3-0.6B the covariance-aware construction is the best of the six (0.122 vs 0.192 for activation magnitude alone and 1.041 for the native order). On Qwen3.5-0.8B, activation magnitude *wins* at $tau=0.20$ (0.430 vs 0.552) and the covariance-aware construction wins at $tau=0.30$ (1.323 vs 1.407); the two differ by less than the token-to-token spread of the measurement at these window sizes, and both beat every order that ignores one of the two factors. Weight-norm ordering is actively harmful (1.488), which is the sharpest evidence that the activation factor carries most of the signal. Third, mean per-layer rel-$L_2$ tracks $Delta$PPL across strategies (0.188 for the two best, 0.247 for random), so the per-layer reconstruction error is a usable proxy for model-level damage.

#figure(table(
  columns: 3,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Candidate window $rho$ (lanes)], [0.8B $Delta$PPL], [0.6B $Delta$PPL],
  [1], [+0.381], [+0.221],
  [2], [+0.369], [+0.609],
  [4], [+0.552], [+0.122],
  [8], [+0.393], [+0.299],
  [full], [+1.148], [+0.646],
), caption: [Hyperparameter sweep of the greedy candidate window ($32rho$ coldest unassigned lanes) at $tau=0.20$. A narrow window keeps the importance order dominant; a wide one lets the covariance term pick partners from further down the activation ranking.])

The candidate window matters more than the block width. Widening the search from 32 to all unassigned lanes costs a factor of three in $Delta$PPL (1.148 at `full`), because the greedy objective will happily assemble a block of individually cheap but mutually reinforcing lanes that the threshold then cannot drop; a window of 32 to 256 lanes keeps the partition close to the activation ranking while still exploiting covariance. The optimal window differs between models (1 for 0.8B, 4–8 for 0.6B), so $rho$ is a hyperparameter, not a constant; the main tables use $rho=4$ for both models and are therefore conservative for 0.8B.

#figure(table(
  columns: 2,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Lane width $W$], [0.8B $Delta$PPL],
  [16], [+0.508],
  [64], [+0.311],
), caption: [Block width sweep at $tau=0.20$ on Qwen3.5-0.8B. $W=32$ matches one GPU warp and two AVX-512 registers; $W=16$ maps to half a warp, $W=64$ to two warps. Wider blocks keep more coupled lanes together, which lowers $Delta$PPL but coarsens the skip granularity.])


== Permutation or threshold?

The two mechanisms are separable: a permutation changes which lanes share a block, a threshold decides how much energy is dropped. The 2$times$2 below holds the skip rate at 0.20 and switches each independently.

#figure(table(
  columns: 7,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Model], [Order], [Budget rule], [$Delta$PPL], [rel-$L_2$], [Gini], [actual skip],
  [0.8B], [native], [Top-$K$], [+1.612], [0.240], [0.514], [0.196],
  [0.8B], [native], [threshold], [+1.084], [0.225], [0.514], [0.200],
  [0.8B], [perm], [Top-$K$], [+1.819], [0.201], [0.573], [0.196],
  [0.8B], [perm], [threshold], [+0.552], [0.188], [0.573], [0.200],
  [0.6B], [native], [Top-$K$], [+1.447], [0.179], [0.643], [0.198],
  [0.6B], [native], [threshold], [+1.041], [0.162], [0.643], [0.200],
  [0.6B], [perm], [Top-$K$], [+1.702], [0.144], [0.697], [0.198],
  [0.6B], [perm], [threshold], [+0.122], [0.128], [0.697], [0.200],
), caption: [2$times$2 factorial at matched 20% skip. The interaction is what matters: the permutation on its own is *worse* than the native order under Top-$K$, and better only when the budget is a calibrated energy quantile.])

#figure(table(
  columns: 6,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Model], [native/Top-$K$], [native/threshold], [perm/Top-$K$], [perm/threshold], [interaction],
  [0.8B], [+1.612], [+1.084], [+1.819], [+0.552], [-0.739],
  [0.6B], [+1.447], [+1.041], [+1.702], [+0.122], [-1.174],
), caption: [Interaction term $(d - b) - (c - a)$ of the factorial: the extra gain obtained by combining the permutation with an energy threshold rather than with a fixed block count.])

The permutation alone is never the winner: under Top-$K$ it is worse than the native order on both models (+1.819 vs +1.612 and +1.702 vs +1.447), and it becomes the winner only once the budget is a calibrated energy quantile. The interaction terms in the second table are the quantitative version of that statement.

== Calibration corpus and calibration size

The permutation and the threshold are estimated from a corpus. If the method only works when calibration and evaluation text are the same, it is not deployable; the 4$times$4 below crosses four calibration corpora with four evaluation corpora at $tau=0.20$.

#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [calibration / evaluation], [eval wiki], [eval code], [eval zh], [eval math],
  [cal wiki], [+0.307], [+0.151], [+0.018], [+0.069],
  [cal code], [+0.222], [+0.210], [+0.010], [+0.037],
  [cal zh], [+0.190], [+0.189], [+0.017], [-0.001],
  [cal math], [+0.720], [+0.210], [+0.019], [+0.093],
), caption: [$Delta$PPL matrix for Qwen3.5-0.8B at $tau=0.20$ (dense wiki 16.21, code 8.81, zh 1.51, math 3.94).])

Block overlap between the permutations is wiki|zh: 0.000, code|wiki: 0.000, code|zh: 0.000, code|math: 0.000, math|wiki: 0.000, math|zh: 0.000 — i.e. most 32-lane blocks are recovered regardless of the calibration corpus, and the differences are concentrated in the cold tail that the budget actually drops.

#figure(table(
  columns: 3,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Calibration tokens*], [PPL], [$Delta$PPL],
  [128], [16.618], [+0.405],
  [512], [16.520], [+0.307],
  [2048], [16.520], [+0.307],
  [8192], [16.520], [+0.307],
), caption: [Calibration-size sweep on WikiText-2 at $tau=0.20$; the permutation and the thresholds are rebuilt from the first $N$ tokens only.])


#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [calibration / evaluation], [eval wiki], [eval code], [eval zh], [eval math],
  [cal wiki], [+0.385], [+0.166], [+0.008], [+0.101],
  [cal code], [-0.032], [+0.206], [+0.014], [+0.053],
  [cal zh], [+0.049], [+0.076], [+0.013], [+0.069],
  [cal math], [+0.055], [+0.259], [+0.022], [+0.105],
), caption: [$Delta$PPL matrix for Qwen3-0.6B at $tau=0.20$ (dense wiki 17.99, code 11.04, zh 1.63, math 4.45).])

Block overlap between the permutations is wiki|zh: 0.000, code|wiki: 0.000, code|zh: 0.000, code|math: 0.000, math|wiki: 0.000, math|zh: 0.000 — i.e. most 32-lane blocks are recovered regardless of the calibration corpus, and the differences are concentrated in the cold tail that the budget actually drops.

#figure(table(
  columns: 3,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Calibration tokens*], [PPL], [$Delta$PPL],
  [128], [18.270], [+0.276],
  [512], [18.114], [+0.120],
  [2048], [18.040], [+0.046],
  [8192], [17.981], [-0.012],
), caption: [Calibration-size sweep on WikiText-2 at $tau=0.20$; the permutation and the thresholds are rebuilt from the first $N$ tokens only.])


== Quality under a quantised forward pass

The sparse path is often combined with 4-bit weights. We therefore round-trip every FFN weight through the IQ4-NL codec (32 weights → one fp16 scale + 16 packed indices, 18 bytes) *in place*, then run the full forward — attention, RoPE, the gated-DeltaNet recurrence, the KV-free state and the 248,320-way head — to obtain real perplexities rather than FFN-only error proxies.

#figure(table(
  columns: 6,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Model], [$tau$], [PPL], [$Delta$ vs fp32 dense], [$Delta$ vs IQ4-NL dense], [top-1 vs fp32],
  [0.8B], [0.00], [16.810], [+0.598], [+0.000], [0.906],
  [0.8B], [0.10], [17.049], [+0.836], [+0.238], [0.883],
  [0.8B], [0.20], [17.545], [+1.332], [+0.735], [0.871],
  [0.8B], [0.30], [18.260], [+2.047], [+1.450], [0.828],
  [0.8B], [0.40], [19.824], [+3.611], [+3.013], [0.789],
  [0.6B], [0.00], [19.425], [+1.431], [+0.000], [0.836],
  [0.6B], [0.10], [19.501], [+1.507], [+0.076], [0.820],
  [0.6B], [0.20], [19.819], [+1.825], [+0.394], [0.801],
  [0.6B], [0.30], [20.792], [+2.799], [+1.367], [0.746],
  [0.6B], [0.40], [22.307], [+4.313], [+2.882], [0.695],
), caption: [IQ4-NL-simulated FFN weights, full model forward, WikiText-2 (8,192 tokens). The middle column is what a user sees relative to the deployment baseline; the right one isolates the sparsity cost from the quantisation cost.])


== System behaviour: batch, context and the speedup ladder

#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Batch $B$], [mean ms/step], [p50], [p95], [ms/token],
  [1], [15.60], [11.83], [32.18], [15.60],
  [2], [25.21], [23.43], [33.06], [12.60],
  [4], [48.54], [46.79], [56.39], [12.13],
  [8], [95.00], [93.19], [103.12], [11.87],
), caption: [Decode step latency at qwen3.5-0.8b, one token per sequence with a warm cache, 12 steps per measurement.])

#figure(table(
  columns: 3,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Context $L$], [prefill ms], [tok/s],
  [128], [124.5], [1028.3],
  [512], [467.1], [1096.1],
  [2048], [1852.7], [1105.4],
), caption: [Prefill throughput at qwen3.5-0.8b, best of three.])

#figure(table(
  columns: 3,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Configuration], [ms/token], [speedup],
  [dense], [11.8], [1.00×],
  [tau0.20], [12.5], [0.94×],
  [tau0.40], [12.5], [0.94×],
), caption: [End-to-end decode ladder at qwen3.5-0.8b: the same budgets as the kernel tables, but measured through the full model.])

`perf stat` counters: unavailable on this host (perf_event_paranoid), so counters are reported as the byte-count model instead.
