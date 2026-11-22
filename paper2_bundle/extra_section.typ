= Kernel microbenchmarks and ceilings

The tables below are the raw kernel data behind Figures 3 and 4 and the per-token claims in Section 5.3. All rows use one `down_proj` of layer 11 of Qwen3.5-0.8B (3584$times$1024) unless stated otherwise; the dense reference is cuBLAS on the GPU and oneDNN on the CPU.

#figure(table(
  columns: 8,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [skip], [dense ms (T=1)], [v7 list ms], [v7 speedup], [v4 mask speedup], [dense ms (T=64)], [v7 list ms], [v7 speedup],
  [0.10], [0.0097], [0.0093], [1.04×], [1.02×], [0.459], [0.456], [1.01×],
  [0.20], [0.0097], [0.0084], [1.16×], [1.04×], [0.460], [0.407], [1.13×],
  [0.30], [0.0097], [0.0075], [1.30×], [1.05×], [0.460], [0.354], [1.30×],
  [0.40], [0.0097], [0.0067], [1.46×], [1.04×], [0.460], [0.306], [1.51×],
  [0.50], [0.0097], [0.0056], [1.75×], [1.05×], [0.461], [0.255], [1.81×],
  [0.60], [0.0097], [0.0048], [2.02×], [1.06×], [0.460], [0.208], [2.21×],
  [0.75], [0.0097], [0.0033], [2.93×], [1.07×], [0.459], [0.133], [3.45×],
), caption: [CUDA `down_proj` at one token (columns 2–5) and 64 tokens (6–8). v4 indexes a 32-bit mask inside each warp; v7 reads a compacted active-block list produced once per token by a `__ballot_sync` compaction kernel. Numerical agreement between list and mask kernels is at the 1e-8 level.])

#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [skip], [mask ms], [list ms], [list/mask], [bytes traversed (MB)],
  [0.10], [0.1123], [0.1246], [0.90×], [13.2],
  [0.20], [0.0907], [0.1065], [0.85×], [11.8],
  [0.30], [0.0831], [0.0921], [0.90×], [10.2],
  [0.40], [0.0731], [0.0820], [0.89×], [8.8],
  [0.50], [0.0621], [0.0748], [0.83×], [7.3],
  [0.60], [0.0536], [0.0623], [0.86×], [5.9],
  [0.75], [0.0383], [0.0442], [0.87×], [3.7],
), caption: [CPU AVX-512 variants of the same kernel, token-parallel, 16 threads. The compacted list is *slower* than the mask on the CPU: the mask loop is already branch-free and the extra indirection costs a dependent load per block. This is why the CPU path keeps the mask and parallelises over output rows instead.])

#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Configuration], [GPU ms/token], [GPU tok/s], [CPU ms/token], [CPU tok/s],
  [dense], [11.62], [86.0], [196.0], [5.1],
  [kernel, 0% skip], [11.40], [87.7], [195.7], [5.1],
), caption: [End-to-end decode of all 24 layers with the FFN replaced by the masked PyTorch reference path (GPU) and the AVX-512 masked kernel (CPU). At $T=1$ the whole model is dominated by the attention/KV work and by kernel-launch overhead, so the FFN saving does not reach the top line.])

== Byte ceilings, recomputed

A decode step touches every weight once, so removing a fraction $f$ of the per-token bytes bounds the speedup by $1 ⁄ (1 - f)$ before any kernel inefficiency is counted. Column 3 applies that bound to a 50% block budget on the projection named in column 1.

#figure(table(
  columns: 3,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Scope of the skip], [share of per-token bytes], [ceiling at 50% block budget],
  [Qwen3.5-0.8B, `down_proj` only], [10.4%], [1.055×],
  [Qwen3.5-0.8B, `up` + `down` (coupled)], [20.8%], [1.116×],
  [Qwen3.5-0.8B, whole FFN], [63.9%], [1.470×],
  [Qwen3.5-0.8B, whole model (FFN + attention)], [100%], [2.000×],
  [Qwen3-0.6B, `down_proj` only], [29.0%], [1.170×],
  [Qwen3-0.6B, whole FFN], [62.0%], [1.450×],
), caption: [Amdahl ceilings per decode step. The measured end-to-end numbers that correspond to these rows are 0.94× (0.8B, `down_proj`, GPU, $T=1$), 1.01× (0.8B, coupled, CPU, $tau=0.40$) and 1.17× (0.8B, `down_proj`, CPU, $tau=0.75$). The whole-model row is listed only to show how much of the budget the FFN can address: closing the gap to it requires attention sparsity as well, which this paper does not attempt.])
