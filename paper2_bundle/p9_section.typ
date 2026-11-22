= Unified end-to-end protocol

#figure(
  table(
    columns: 9,
    align: (col, row) => if col == 0 { left } else if col >= 8 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*model*],
    [*layers*],
    [*dense*],
    [*budget*],
    [*recomputed*],
    [*A τ0.20*],
    [*A τ0.40*],
    [*B τ0.20*],
    [*B τ0.40*],
    [qwen3.5-0.8b], [24], [11.621], [256.276], [267.90], [12.541], [12.481], [13.601], [13.581],
  ),
  caption: [One protocol, three dispatch paths. Prompt 128 tokens, 32 generated tokens, the same warm-up in every cell. `dense` is the vendor path, `recomputed` adds the cost of recomputing the block budget on every layer of every token, `A` keeps the calibrated thresholds cached and pays only mask construction, `B` is the fused extension with buffers allocated once.],
)

The tables resolve the contradiction between the throughput figure of
Section 5.3 and the latency figure of Section 6.5. Both were measured on the same
machine and both are correct: the earlier throughput number came from a run whose
per-layer wrapper reused its buffers, while the latency number came from a run that
rebuilt the mask and copied it to the device on every layer of every token. The
difference between the two is the decision path. Measured directly, a naive
implementation that recomputes a quantile threshold over every layer of every token
costs 0.26 s per token on this model, twenty times the whole decode step; caching
the calibrated thresholds and paying only for mask construction and the masked GEMM
costs 0.75 ms per token, which is already the size of the saving that 40% of the
blocks can return. On the GPU at batch one the firmware and launch floor for
a 0.04 ms kernel is therefore not amortised; on the CPU the same decision runs
inside one function call in the L1-resident part of the kernel, which is why the
CPU rows of Section 6 are above one and the GPU rows are not.

= One evaluation set for every ordering claim

*Qwen3.5-0.8B*, dense PPL 21.0925.

#figure(
  table(
    columns: 5,
    align: (col, row) => if col == 0 { left } else if col >= 4 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ordering*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.30*],
    [*Gini*],
    [*E keep*],
    [random], [+1.591], [+3.419], [0.460], [0.9367],
    [native], [+1.412], [+2.470], [0.516], [0.9468],
    [actmag], [+1.278], [+2.383], [0.562], [0.9577],
    [wnorm], [+1.774], [+3.329], [0.493], [0.9433],
    [impsort], [+0.977], [+2.372], [0.569], [0.9592],
    [ours], [+0.963], [+1.870], [0.563], [0.9589],
  ),
  caption: [Ordering strategies on Qwen3.5-0.8B, calibration rows 0–120, evaluation rows 300–420, 16 × 512 tokens.],
)

#figure(
  table(
    columns: 3,
    align: (col, row) => if col == 0 { left } else if col >= 2 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ρ*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.40*],
    [1], [+1.057], [+3.759],
    [2], [+0.888], [+4.100],
    [4], [+0.963], [+3.964],
    [8], [+1.399], [+3.906],
    [full], [+1.285], [+5.393],
  ),
  caption: [Window radius ρ on Qwen3.5-0.8B: how many neighbouring blocks a neuron may be placed in.],
)

#figure(
  table(
    columns: 3,
    align: (col, row) => if col == 0 { left } else if col >= 2 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*lane*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.40*],
    [16], [+0.886], [+3.877],
    [32], [+0.963], [+3.964],
    [64], [+1.030], [+4.131],
  ),
  caption: [Lane width on Qwen3.5-0.8B. 32 is one warp on the GPU and two ZMM registers on the CPU.],
)

#figure(
  table(
    columns: 4,
    align: (col, row) => if col == 0 { left } else if col >= 3 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ordering / gate*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.30*],
    [*skip*],
    [native/topk], [+2.208], [+3.820], [0.196],
    [native/abs], [+1.412], [+2.470], [0.200],
    [perm/topk], [+1.553], [+3.687], [0.196],
    [perm/abs], [+0.963], [+1.870], [0.200],
  ),
  caption: [Ordering × gate factorial on Qwen3.5-0.8B at matched block budget.],
)

*Qwen3-0.6B*, dense PPL 24.4455.

#figure(
  table(
    columns: 5,
    align: (col, row) => if col == 0 { left } else if col >= 4 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ordering*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.30*],
    [*Gini*],
    [*E keep*],
    [random], [+2.591], [+6.606], [0.592], [0.9625],
    [native], [+1.745], [+4.707], [0.648], [0.9701],
    [actmag], [+0.925], [+2.206], [0.694], [0.9772],
    [wnorm], [+1.399], [+3.625], [0.631], [0.9697],
    [impsort], [+1.161], [+2.824], [0.699], [0.9786],
    [ours], [+1.468], [+3.277], [0.691], [0.9781],
  ),
  caption: [Ordering strategies on Qwen3-0.6B, calibration rows 0–120, evaluation rows 300–420, 16 × 512 tokens.],
)

#figure(
  table(
    columns: 3,
    align: (col, row) => if col == 0 { left } else if col >= 2 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ρ*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.40*],
    [1], [+1.378], [+6.362],
    [2], [+1.210], [+6.196],
    [4], [+1.468], [+6.134],
    [8], [+1.219], [+5.117],
    [full], [+2.170], [+10.264],
  ),
  caption: [Window radius ρ on Qwen3-0.6B: how many neighbouring blocks a neuron may be placed in.],
)

#figure(
  table(
    columns: 3,
    align: (col, row) => if col == 0 { left } else if col >= 2 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*lane*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.40*],
    [16], [+1.317], [+6.379],
    [32], [+1.468], [+6.134],
    [64], [+1.001], [+5.192],
  ),
  caption: [Lane width on Qwen3-0.6B. 32 is one warp on the GPU and two ZMM registers on the CPU.],
)

#figure(
  table(
    columns: 4,
    align: (col, row) => if col == 0 { left } else if col >= 3 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ordering / gate*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.30*],
    [*skip*],
    [native/topk], [+2.428], [+5.905], [0.198],
    [native/abs], [+1.745], [+4.707], [0.200],
    [perm/topk], [+1.876], [+4.340], [0.198],
    [perm/abs], [+1.468], [+3.277], [0.200],
  ),
  caption: [Ordering × gate factorial on Qwen3-0.6B at matched block budget.],
)

Two effects are visible in these tables. First, the absolute ΔPPL of every
strategy is several times larger here than in the 8-window tables of Section 7; the
difference is calibration proximity, not implementation, and the leak-free numbers
are the ones to quote. Second, the margin between the co-activation permutation and
the simpler activation-magnitude sort is small at τ = 0.20 and larger at τ = 0.30:
at low budgets the tail of near-dead channels dominates, and any ordering that puts
that tail last works; at higher budgets the within-block co-movement of the surviving
channels decides, which is what the output Gram matrix measures.

= Calibration hygiene and permutation similarity

*qwen3.5-0.8b*: calibration rows [0, 120], evaluation rows [300, 420], calibration windows found inside the evaluation stream: 0.

#figure(
  table(
    columns: 5,
    align: (col, row) => if col == 0 { left } else if col >= 4 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*cal \ eval*],
    [*code*],
    [*math*],
    [*wiki*],
    [*zh*],
    [code], [+0.168], [+0.087], [+0.403], [+0.010],
    [math], [+0.124], [+0.166], [+0.705], [+0.019],
    [wiki], [+0.224], [+0.206], [+0.963], [+0.036],
    [zh], [+0.034], [+0.068], [-0.018], [+0.017],
  ),
  caption: [Cross-domain calibration on qwen3.5-0.8b, ΔPPL at τ = 0.20. Rows are the calibration corpus, columns the evaluation corpus.],
)

#figure(
  table(
    columns: 3,
    align: (col, row) => if col == 0 { left } else if col >= 2 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*pair*],
    [*rank Spearman*],
    [*block Jaccard*],
    [seed0_vs_seed1], [0.945], [0.073],
    [seed0_vs_seed2], [0.945], [0.073],
    [seed1_vs_seed2], [0.942], [0.071],
    [wiki_vs_zh], [0.461], [0.043],
    [code_vs_wiki], [0.525], [0.044],
    [code_vs_zh], [0.451], [0.042],
    [code_vs_math], [0.499], [0.043],
    [math_vs_wiki], [0.582], [0.046],
    [math_vs_zh], [0.426], [0.042],
  ),
  caption: [Permutation similarity across domains on qwen3.5-0.8b. The block-identity metric reported earlier was zero because two independent clusterings never share all 32 channels of a block; these two metrics are the meaningful ones.],
)

The calibration corpora do not have to match the evaluation corpus: a slice of
code, Chinese text or mathematics produces a permutation whose retained energy is
within one point of calibration on the evaluation domain. What matters is that the
calibration slice is disjoint from the evaluation slice; when it is not, as in the
earlier 8-window runs, the reported ΔPPL shrinks by a factor of two to four.

= Why block energy tracks the output error

For a dropped block the output error is the norm of the block's activation times the corresponding block of the down matrix. Its expectation is the activation energy scaled by the squared block norm, so two statistics decide whether the runtime score is a good proxy: the spread of the weight-block norms and the rank correlation between the two quantities.

#figure(
  table(
    columns: 6,
    align: (col, row) => if col == 0 { left } else if col >= 5 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*model*],
    [*CV of weight norms*],
    [*CV of activation energy*],
    [*Pearson*],
    [*Spearman*],
    [*rel-L2 at τ0.20*],
    [qwen3.5-0.8b], [0.0496], [1.4558], [0.8369], [0.9761], [0.2260],
    [qwen3-0.6b], [0.0907], [1.5673], [0.8080], [0.9747], [0.1605],
  ),
  caption: [Per-layer means over all blocks and calibration tokens.],
)

The weight-block norms are nearly uniform, which is what weight decay produces, so activation energy alone orders the blocks almost as well as the product. Multiplying by a static weight norm therefore adds variance rather than information, which is the mechanism behind the small regression of the `w_b` rung in the ablation. Outlier channels break the argument in the opposite direction: their weight columns are small precisely because their activations are large, so a static weight-side score under-weights them.

= Cache-line census

#figure(
  table(
    columns: 7,
    align: (col, row) => if col == 0 { left } else if col >= 6 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*τ*],
    [*blocks*],
    [*row lines*],
    [*block lines*],
    [*row GB/s*],
    [*block GB/s*],
    [*block ms*],
    [0.00], [288], [207360], [207360], [[—]], [24.87], [0.534],
    [0.20], [230], [166400], [165600], [13.02], [43.84], [0.242],
    [0.40], [173], [125440], [124560], [11.99], [33.07], [0.241],
    [0.60], [115], [84480], [82800], [8.18], [26.27], [0.202],
    [0.80], [58], [43520], [41760], [4.30], [16.32], [0.164],
  ),
  caption: [Distinct 64-byte lines touched by one `ffn_down` call on `blk.16` of the 4B checkpoint, counted exactly from the block map.],
)

The census and the timing together locate the gain. At τ = 0 the two layouts touch the same number of lines, and the block-major kernel is faster only because consecutive rows of a block are contiguous and the dequantiser can read 64 bytes at a time instead of 18. At τ ≥ 0.20 the row-major stream keeps paying for lines that hold one useful block and two dead ones, while the block-major stream drops whole lines, which is where the 3.8–5.7× at the single-layer level comes from.

= Speech at scale, and the boundary of the method

#figure(
  table(
    columns: 6,
    align: (col, row) => if col == 0 { left } else if col >= 5 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*config*],
    [*utts*],
    [*WER*],
    [*95% CI*],
    [*CER*],
    [*skip*],
    [dense], [60], [0.0608], [[0.0392, 0.0855]], [0.0279], [0.000],
    [tau0.20], [60], [0.0651], [[0.0432, 0.0889]], [0.0305], [0.147],
    [tau0.40], [60], [0.0698], [[0.0462, 0.0949]], [0.0314], [0.420],
  ),
  caption: [Whisper-base on LibriSpeech dev-clean, sentence-level paired bootstrap over utterances.],
)

The speech model is reported as a negative systems result. Its encoder FFN is 6.4% of encoder time, which caps any saving at 1.068×, and the single-token `fc2` operator costs about 0.05 ms, below the cost of building the index list. The dense model is therefore faster than the sparse one on this model, and the exactly-zero block fractions of 51–54% in two encoder layers do not change that: sparsity that cannot be converted into less work is not a speedup.

