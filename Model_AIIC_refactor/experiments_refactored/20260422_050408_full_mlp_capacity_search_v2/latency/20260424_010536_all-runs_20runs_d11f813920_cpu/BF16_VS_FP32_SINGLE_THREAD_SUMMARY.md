# BF16 vs FP32 Single-Thread Summary

## Scope

This note compares two CPU latency benchmark runs under `num_threads=1`:

- Old timing scope, autocast included in timed region:
  - `20260423_060055_all-runs_20runs_d11f813920_cpu`
- New timing scope, autocast excluded from timed region:
  - `20260424_010536_all-runs_20runs_d11f813920_cpu`

Both runs cover:

- execution modes: `eager`, `jit`, `compile`
- precision profiles: `fp32`, `bf16`
- batch sizes: `1, 2, 4, 8, 16, 32, 64, 128`
- model set: 20 MLP variants

Analysis below uses matched-model comparisons at `num_threads=1`, with geometric-mean ratios across models.

## Executive Conclusion

After moving autocast outside the timed region, `bf16` becomes materially better than before, especially at `batch_size=1`.

However, the overall conclusion does **not** change:

- `bf16` is still generally worse than `fp32` for single-thread CPU latency.
- This remains true for `eager` and `jit` at both `batch_size=1` and `batch_size=128`.
- For `compile` at `batch_size=128`, `bf16` shows a slight aggregate advantage, but the win is weak and not consistent across models.

Practical interpretation:

- Excluding autocast fixed an unfair benchmark penalty on `bf16`.
- It did **not** turn `bf16` into the overall best single-thread precision.

## BF16 Relative To FP32

Interpretation:

- latency ratio `< 1.0` means `bf16` is faster than `fp32`
- throughput ratio `> 1.0` means `bf16` is better than `fp32`

### Old Run: autocast included in timing

| Mode | Batch | BF16/FP32 Latency | BF16/FP32 Throughput | Reading |
| --- | ---: | ---: | ---: | --- |
| eager | 1 | 1.864x | 0.532x | bf16 much worse |
| eager | 128 | 1.249x | 0.811x | bf16 worse |
| jit | 1 | 1.806x | 0.552x | bf16 much worse |
| jit | 128 | 1.189x | 0.838x | bf16 worse |
| compile | 1 | 1.783x | 0.560x | bf16 much worse |
| compile | 128 | 1.161x | 0.859x | bf16 worse |

### New Run: autocast excluded from timing

| Mode | Batch | BF16/FP32 Latency | BF16/FP32 Throughput | Reading |
| --- | ---: | ---: | ---: | --- |
| eager | 1 | 1.317x | 0.768x | bf16 still worse |
| eager | 128 | 1.170x | 0.848x | bf16 still worse |
| jit | 1 | 1.418x | 0.709x | bf16 still worse |
| jit | 128 | 1.367x | 0.729x | bf16 worse, and worse than expected |
| compile | 1 | 1.440x | 0.698x | bf16 still worse |
| compile | 128 | 0.971x | 1.026x | slight aggregate bf16 win |

## What Changed After Excluding Autocast

`fp32` stayed nearly unchanged across the two runs. The meaningful differences came almost entirely from `bf16`.

### New Run vs Old Run: FP32

| Mode | Batch | New/Old Latency | New/Old Throughput | Reading |
| --- | ---: | ---: | ---: | --- |
| eager | 1 | 0.995x | 0.991x | unchanged |
| eager | 128 | 1.011x | 1.000x | unchanged |
| jit | 1 | 0.993x | 1.005x | unchanged |
| jit | 128 | 1.003x | 0.999x | unchanged |
| compile | 1 | 1.006x | 0.990x | unchanged |
| compile | 128 | 1.003x | 0.995x | unchanged |

### New Run vs Old Run: BF16

| Mode | Batch | New/Old Latency | New/Old Throughput | Reading |
| --- | ---: | ---: | ---: | --- |
| eager | 1 | 0.703x | 1.433x | large improvement |
| eager | 128 | 0.948x | 1.047x | small improvement |
| jit | 1 | 0.779x | 1.290x | large improvement |
| jit | 128 | 1.153x | 0.869x | regression |
| compile | 1 | 0.812x | 1.234x | clear improvement |
| compile | 128 | 0.839x | 1.189x | clear improvement |

## Why The Overall Conclusion Is Still "BF16 Is Worse"

The benchmark change removed fixed autocast overhead from the measured region. That matters most when the actual model compute is very small, especially at `batch_size=1`.

But after removing that overhead, several effects still keep `bf16` from broadly beating `fp32` in single-thread CPU runs:

- Many tested MLPs are still too small for `bf16` compute advantages to dominate.
- On CPU, `fp32` kernels are already highly optimized and remain very competitive for small and medium matrix sizes.
- `bf16` benefits appear mainly on larger models and larger batches, where kernel efficiency and fusion have enough work to amortize fixed costs.
- `compile` at `batch_size=128` is the only configuration where this effect becomes visible at aggregate level.

## Model-Level Pattern

In the new run, the `bf16` wins at `num_threads=1` and `batch_size=128` are concentrated in larger models, especially deeper and wider MLPs.

Representative strong `bf16` cases:

- `compile`: `hd512_depth5`, `hd512_depth4`, `hd256_depth5`
- similar tendency also appears in `eager` and `jit`

Representative weak `bf16` cases:

- many `hd32_*` variants remain worse under `bf16`
- several shallow or narrow models still favor `fp32` even after excluding autocast overhead

So the result is not "bf16 is better now". The result is:

- `bf16` penalty became smaller
- some large-model cases turned favorable
- overall single-thread winner is still usually `fp32`

## Recommended One-Line Summary

For single-thread CPU benchmarking on this GNRD system, excluding autocast from the timed region makes `bf16` look more fair, but `bf16` is still overall worse than `fp32`; only `compile + batch_size=128` shows a weak aggregate advantage, driven mainly by larger models.