# CPU Latency Deep Analysis

This note explains three behaviors observed in the CPU latency benchmark for the GNR-D host:

1. Why single-thread latency at batch size 128 is often only about 2x the latency at batch size 1.
2. Why BF16 is usually slower than FP32 in this benchmark.
3. Why the execution-mode ranking is usually `jit < compile < eager` on CPU.

The raw benchmark artifacts for this analysis are in the same directory:

- `latency_results.csv`
- `latency_results.json`
- `latency_samples.npz`
- `LATENCY_REPORT.md`
- `hardware_manifest.json`

## 1. Measurement Scope

The benchmark measures wall-clock latency for one full forward pass of one full batch.

In the benchmark implementation, the timed region wraps a single `model(dummy_input)` call, where `dummy_input` already includes the configured batch size. Throughput is then derived as:

`throughput = batch_size / mean_latency`

The CSV also exports a derived per-sample column:

- `p50_latency_us`
- `latency_per_sample_us = p50_latency_us / batch_size`

This means:

- `p50_latency_ms` is batch latency, not sample latency.
- `latency_per_sample_us` is a post-processing view, not the directly measured quantity.

For the full-MLP models, the model forward path also processes the batch jointly rather than looping over samples in Python. The forward path takes input of shape `(B, 2L)`, runs the joint MLP on the whole batch, and reshapes once at the end.

## 2. Why Single-Thread Batch-128 Latency Is Often Only About 2x Batch-1

### 2.1 Root cause

Single-thread does not mean the program executes `batch_size` independent sample forwards serially at the Python level. It means one CPU thread runs one larger kernel.

For these full-MLP models, increasing batch size mainly changes the matrix shape seen by the linear layers:

- `bs=1` behaves like a very small matrix-vector style workload.
- `bs=128` behaves like a larger matrix-matrix style workload.

Even with one thread, the larger matrix is much more efficient for the CPU backend because:

1. Fixed framework and kernel-launch overhead is amortized across more samples.
2. Small GEMV-like kernels underutilize vector units and cache pipelines.
3. Larger GEMM-like kernels let oneDNN/MKL use wider vectorized inner loops more efficiently.
4. The host supports AVX512 and BF16/AMX-related CPU features, so the kernel library has strong vectorized CPU implementations available.

### 2.2 Why this is especially strong for the small models

This effect is strongest for the shallowest models, where the per-sample compute is tiny.

For this codebase's current `full_mlp` definition, `depth=2` means exactly one learned linear transform from input to flat output, with no hidden layer. That is why the `depth=2` runs do not meaningfully vary with `hidden_dim`: the parameter is present in the config but not used by the model implementation in that branch.

For depth-2 runs, the median `p50(bs=128) / p50(bs=1)` ratios under single-thread FP32 are:

- eager: `1.402`
- jit: `1.683`
- compile: `1.473`

Examples for `full_mlp_capacity_search_hd128_depth2`:

- eager: `0.0280 ms -> 0.0407 ms`, ratio `1.453`
- jit: `0.0183 ms -> 0.0307 ms`, ratio `1.678`
- compile: `0.0290 ms -> 0.0400 ms`, ratio `1.380`

But the corresponding throughput increases are much larger:

- eager: `88.8x`
- jit: `75.6x`
- compile: `94.5x`

That combination is the signature of fixed-cost amortization plus better large-matrix kernel efficiency.

### 2.3 Why this effect weakens on larger models

As model depth and parameter count increase, actual arithmetic cost dominates fixed overhead more strongly, so batch latency grows by more than 2x.

For single-thread FP32, median `p50(bs=128) / p50(bs=1)` by depth is:

#### eager

- depth 2: `1.402`
- depth 3: `2.205`
- depth 4: `2.742`
- depth 5: `2.997`

#### jit

- depth 2: `1.683`
- depth 3: `2.791`
- depth 4: `3.590`
- depth 5: `4.242`

#### compile

- depth 2: `1.473`
- depth 3: `2.280`
- depth 4: `2.596`
- depth 5: `2.727`

Large-model examples show the opposite regime:

- `full_mlp_capacity_search_hd512_depth5`, eager: `0.1567 ms -> 1.1426 ms`, ratio `7.293`
- `full_mlp_capacity_search_hd256_depth5`, jit: `0.0303 ms -> 0.2732 ms`, ratio `9.015`
- `full_mlp_capacity_search_hd512_depth4`, compile: `0.0826 ms -> 0.6687 ms`, ratio `8.098`

So the "batch-128 is only about 2x batch-1" behavior is real, but it is mostly a small-model effect rather than a universal CPU rule.

## 3. Why BF16 Is Usually Slower Than FP32 on CPU in This Benchmark

### 3.1 What the benchmark is actually doing for BF16

On CPU, BF16 is enabled through `torch.autocast(device_type='cpu', dtype=torch.bfloat16)`.

That is important because this benchmark is not explicitly converting the entire model and all tensors to native BF16 ahead of time. Instead, it uses CPU autocast and lets PyTorch choose BF16-capable operations dynamically.

This creates two consequences:

1. The run is mixed-precision/autocast-driven, not a pure preconverted BF16 graph.
2. Some operations may still run in FP32 or require dtype conversions around the fast kernels.

For small models and small batches, those conversion and dispatch costs can dominate the arithmetic savings.

### 3.2 Aggregate result

Across all single-thread CPU points in this benchmark, BF16 is usually slower than FP32.

Median `BF16 latency / FP32 latency`:

- eager: `1.827`
- jit: `1.970`
- compile: `1.831`

Median BF16 throughput relative to FP32:

- eager: `0.548`
- jit: `0.503`
- compile: `0.548`

So the typical result here is not a small regression. It is roughly:

- BF16 latency about `1.8x` to `2.0x` FP32 latency
- BF16 throughput about `50%` to `55%` of FP32 throughput

### 3.3 Why the regression is strongest at small batch

By batch size, the median BF16/FP32 latency ratio is:

#### eager

- bs 1: `1.884`
- bs 2: `2.097`
- bs 4: `1.987`
- bs 8: `2.013`
- bs 16: `1.872`
- bs 32: `1.702`
- bs 64: `1.575`
- bs 128: `1.515`

#### jit

- bs 1: `1.926`
- bs 2: `2.058`
- bs 4: `2.200`
- bs 8: `2.316`
- bs 16: `1.998`
- bs 32: `1.882`
- bs 64: `1.720`
- bs 128: `1.520`

#### compile

- bs 1: `1.748`
- bs 2: `2.036`
- bs 4: `2.063`
- bs 8: `2.026`
- bs 16: `1.833`
- bs 32: `1.699`
- bs 64: `1.509`
- bs 128: `1.447`

This pattern strongly suggests the following:

1. BF16 overhead is mostly not worth paying on small workloads.
2. As batch size grows, compute becomes larger and BF16 gets closer to being worthwhile.
3. Even then, the benefit only appears for the largest and heaviest model points.

### 3.4 Where BF16 actually wins

BF16 does win in some cases, but almost all wins are at the largest batches and largest models.

Best examples:

- eager, `full_mlp_capacity_search_hd512_depth5`, bs 128: ratio `0.491`, throughput `2.022x`
- jit, `full_mlp_capacity_search_hd512_depth5`, bs 128: ratio `0.430`, throughput `2.308x`
- compile, `full_mlp_capacity_search_hd512_depth5`, bs 128: ratio `0.453`, throughput `2.197x`

Other BF16 wins also cluster around:

- depth 4 to depth 5
- hidden dim 256 to 512
- batch 64 or 128

This is exactly the regime where arithmetic density is finally large enough for BF16-capable CPU kernels to pay back their overhead.

### 3.5 Practical explanation

The most likely practical explanation on this CPU is:

1. FP32 kernels are already very efficient for these small dense MLPs.
2. CPU BF16 is enabled through autocast rather than full graph-native BF16 conversion.
3. Small shapes do not exploit BF16 hardware efficiently enough to beat FP32.
4. BF16 only becomes advantageous when batch and model size are large enough for kernel throughput to dominate conversion and dispatch overhead.

So the observed result is not "BF16 support is broken". It is "BF16 acceleration only pays off once the workload is large enough".

## 4. Why `jit` Is Usually Better Than `compile`, and `compile` Better Than `eager`, on CPU

### 4.1 What each mode does in this benchmark

- `eager`: plain eager execution, no graph conversion.
- `jit`: `torch.jit.trace(...)`, then `freeze(...)`, then `optimize_for_inference(...)` where possible.
- `compile`: `torch.compile(model, mode='reduce-overhead')`.

The key detail is that the JIT path here is aggressively inference-oriented for a stable static graph, while the compile path uses a general-purpose compiler stack that still carries more overhead than the very optimized traced-and-frozen JIT path on these CPU MLPs.

### 4.2 Aggregate ranking stability

For single-thread FP32, median mode ratios are:

- `jit / eager = 0.600`
- `compile / eager = 0.940`
- `compile / jit = 1.563`

For single-thread BF16, median mode ratios are:

- `jit / eager = 0.641`
- `compile / eager = 0.919`
- `compile / jit = 1.444`

So the typical ordering is:

`jit < compile < eager`

### 4.3 How stable that ordering is

For FP32, the exact order counts across all run-and-batch points are:

- `jit < compile < eager`: `148 / 160`
- `jit < eager < compile`: `11 / 160`
- `compile < jit < eager`: `1 / 160`

For BF16:

- `jit < compile < eager`: `136 / 160`
- `jit < eager < compile`: `24 / 160`

For FP32 specifically, JIT is the fastest mode for:

- bs 1: `20 / 20`
- bs 2: `19 / 20`
- bs 4: `20 / 20`
- bs 8: `20 / 20`
- bs 16: `20 / 20`
- bs 32: `20 / 20`
- bs 64: `20 / 20`
- bs 128: `20 / 20`

This is not a fragile effect. On this benchmark, JIT is almost always the best CPU mode.

### 4.4 Why JIT helps so much here

These full-MLP models are almost ideal JIT-trace candidates:

1. Static tensor shapes for each benchmark point.
2. Very regular dense linear and ReLU structure.
3. Minimal control flow.
4. Inference-only execution.

That lets tracing, freezing, and inference optimization reduce Python overhead and expose a clean inference graph to the backend.

As model depth increases, JIT's advantage over eager becomes even stronger. For FP32, median `jit / eager` by depth is:

- depth 2: `0.657`
- depth 3: `0.581`
- depth 4: `0.553`
- depth 5: `0.521`

So deeper models give JIT more opportunities to remove eager overhead and streamline execution.

### 4.5 Why compile is better than eager but worse than JIT here

`torch.compile` still helps relative to eager in most cases, but not nearly as much as JIT for this workload.

The most likely reasons are:

1. `torch.compile(mode='reduce-overhead')` is using a more general compiler stack than the specialized JIT inference path.
2. These are small, dense, static CPU MLPs where the extra compiler machinery has less room to pay off.
3. The JIT path here explicitly applies `freeze` and `optimize_for_inference`, which is very well matched to static CPU inference.
4. Compile can still reduce some eager overhead, which explains why it is usually slightly faster than eager.

In short:

- JIT is the best fit for this specific CPU workload.
- Compile is helpful, but not as specialized.
- Eager keeps the most framework overhead exposed.

## 5. Combined Interpretation

All three observed effects are consistent with the same underlying CPU behavior.

### Small-workload regime

This regime includes small hidden size, shallow depth, and small batch size.

Characteristics:

- fixed overhead is a large fraction of total latency
- `bs=128` may be only `1.4x` to `2x` slower than `bs=1`
- BF16 is usually worse than FP32
- JIT provides the largest benefit by reducing execution overhead

### Large-workload regime

This regime includes larger hidden size, deeper MLP, and batch 64 or 128.

Characteristics:

- arithmetic dominates fixed overhead more strongly
- `bs=128` grows much more than `2x` over `bs=1`
- BF16 can finally beat FP32 on the largest points
- JIT still tends to be best, but all modes become more kernel-dominated

## 6. Bottom-Line Answers

### Why is single-thread batch-128 latency only about 2x batch-1?

Because the benchmark measures one whole-batch forward pass, and the model executes the whole batch jointly. On small CPU MLPs, `bs=1` is dominated by fixed overhead and poor small-matrix efficiency, while `bs=128` becomes a much more efficient single-thread matrix kernel. Total latency rises slowly, while throughput rises dramatically.

### Why is BF16 slower than FP32?

Because in this benchmark BF16 is driven by CPU autocast, not by a fully native BF16 graph, and most measured workloads are too small for BF16 kernel advantages to overcome conversion, dispatch, and mixed-precision overhead. BF16 only starts winning on the largest batches and largest models.

### Why is `jit` better than `compile`, and `compile` better than `eager`, on CPU?

Because this workload is an almost perfect match for static traced inference optimization. The JIT path here applies tracing, freezing, and inference-specific optimization to a regular dense MLP, which strips more overhead than eager and outperforms the more general `torch.compile` stack for these small static CPU graphs.

## 7. Suggested Follow-Up Checks

If the goal is deployment guidance rather than diagnosis, the most useful next checks are:

1. Benchmark a true native BF16 path by explicitly converting a copy of the model and inputs to BF16 instead of relying only on CPU autocast.
2. Repeat the same analysis at larger hidden sizes or sequence sizes to locate the BF16 crossover point more precisely.
3. Pin CPU affinity and compare with explicit oneDNN/MKL thread environment settings to separate scheduler noise from kernel behavior.
4. If deployment batch is always 1, choose JIT + FP32 on CPU for this model family unless a larger model-specific BF16 win is demonstrated.