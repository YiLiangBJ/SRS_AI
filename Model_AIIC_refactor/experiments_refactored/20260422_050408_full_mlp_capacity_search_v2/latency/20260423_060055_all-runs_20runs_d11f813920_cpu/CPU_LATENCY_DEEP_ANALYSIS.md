
# CPU Latency 深度分析报告

## 1. 报告目的

这份报告面向 CPU 侧推理延迟结果，主要回答下面几个问题：

1. 为什么单线程下 `batch=128` 的 latency 往往只比 `batch=1` 大约高 `1.4x` 到 `2x`。
2. 为什么在这组 CPU 测试里，`BF16` 大多数情况下比 `FP32` 更慢。
3. 为什么执行模式通常表现为 `jit < compile < eager`。
4. 为什么多线程整体上大多是负收益，尤其是 `128` 线程会出现非常严重的退化。
5. 当前 `full_mlp` 定义下，为什么 `depth=2` 时不同 `hidden_dim` 基本没有差别。
6. 如果怀疑小矩阵 GEMM 存在固定开销，那么从 `depth=2->3`、`3->4`、`4->5` 的 latency 增量来看，这种固定开销假设是否成立。

本目录下用于分析的原始文件包括：

- `latency_results.csv`
- `latency_results.json`
- `latency_samples.npz`
- `LATENCY_REPORT.md`
- `hardware_manifest.json`

## 2. 测试平台与基本背景

本次 CPU benchmark 运行在以下平台：

- CPU 型号：`Intel(R) Xeon(R) 6760P`
- 主机名：`sh14l07002s1404`
- 物理核心数：`128`
- 逻辑 CPU 数：`256`
- CPU capability：`AVX512`
- mkldnn：启用
- Python：`3.11.9`
- PyTorch：`2.1.2+cu121`

这个平台本身支持较强的向量化执行能力，因此在小矩阵和大矩阵之间，单线程 CPU kernel 的效率差异会被明显放大。

## 3. 测量口径

这个 benchmark 测的是“整个 batch 一次 forward 的 wall-clock latency”，不是单样本 latency。

也就是说，计时代码包住的是一次：

- `model(dummy_input)`

其中 `dummy_input` 已经带有配置好的 `batch_size`。随后吞吐率按下面方式计算：

- `throughput = batch_size / mean_latency`

CSV 中虽然有：

- `p50_latency_us`
- `latency_per_sample_us`

但这两个里的 `latency_per_sample_us` 是后处理派生值，不是原始测量值。

因此要注意：

1. `p50_latency_ms` 是整批延迟，不是单样本延迟。
2. `latency_per_sample_us` 只是为了帮助观察摊销效果。

## 4. 模型输入输出维度与 depth 语义

本实验中的 `full_mlp` 都共享相同的输入输出几何，只改变 `hidden_dim` 和 `mlp_depth`。

### 4.1 输入输出维度

已知：

- `seq_len = 12`
- `num_ports = 6`

所以在 benchmark 中的实数堆叠表示为：

- 输入形状：`(B, 24)`，因为 `2 * seq_len = 24`
- 输出形状：`(B, 6, 24)`

如果按复数视角理解，则对应：

- 输入形状：`(B, 12)` complex
- 输出形状：`(B, 6, 12)` complex

### 4.2 当前代码里 `depth=2` 的真实含义

这里必须明确说明当前实现语义，否则很容易误读结果。

在这套 `full_mlp` 代码里：

- `depth=2` 表示只有一层从输入直接到输出的线性映射
- 即：`24 -> 144`
- 没有隐藏层

因此：

- `depth=2` 时，`hidden_dim` 在配置里虽然存在，但不会被模型结构使用
- 所以 `hd32 / hd64 / hd128 / hd256 / hd512` 这些 `depth=2` 组合，本质上是同一种模型结构

这也解释了为什么本目录里历史结果会出现 `20` 个 run，而不是文档中预期的 `16` 个有效组合：

- `depth=2` 和 `hidden_dim` 被做了笛卡尔积
- 但这些组合在 `full_mlp` 上是重复语义

这批结果本身仍然是“测对了”，只是 sweep 设计包含了冗余组合。

## 5. 为什么单线程下 batch=128 往往只比 batch=1 慢约 2 倍

### 5.1 根因

单线程不等于“Python 层串行跑 128 次单样本推理”。

对 `full_mlp` 来说，batch 会被整体送进 joint MLP，一次 forward 里处理整个 batch。于是：

- `bs=1` 更像很小的矩阵乘 / 矩阵向量运算
- `bs=128` 更像更大的矩阵乘

即使只有一个线程，大矩阵也更容易让 oneDNN / MKL 把 AVX512 向量单元吃满，因此固定开销会被更好地摊薄。

### 5.2 在小模型上最明显

对单线程 `FP32`，`p50(bs=128) / p50(bs=1)` 的深度中位数如下：

| mode | depth 2 | depth 3 | depth 4 | depth 5 |
| --- | ---: | ---: | ---: | ---: |
| eager | 1.402 | 2.205 | 2.742 | 2.997 |
| jit | 1.683 | 2.791 | 3.590 | 4.242 |
| compile | 1.473 | 2.280 | 2.596 | 2.727 |

对 `full_mlp_capacity_search_hd128_depth2`：

| mode | bs=1 p50 ms | bs=128 p50 ms | 比值 |
| --- | ---: | ---: | ---: |
| eager | 0.0280 | 0.0407 | 1.453 |
| jit | 0.0183 | 0.0307 | 1.678 |
| compile | 0.0290 | 0.0400 | 1.380 |

但对应吞吐提升非常大：

- eager：`88.8x`
- jit：`75.6x`
- compile：`94.5x`

这正是“固定开销被摊薄 + 大矩阵 kernel 更高效”的典型特征。

### 5.3 在大模型上会变得没那么夸张

随着 depth 和 hidden width 增大，算术量本身开始主导总时间，batch 放大后 latency 的增长也会更接近“随计算量增加”。

例如：

- `full_mlp_capacity_search_hd512_depth5`, eager：`0.1567 ms -> 1.1426 ms`，比值 `7.293`
- `full_mlp_capacity_search_hd256_depth5`, jit：`0.0303 ms -> 0.2732 ms`，比值 `9.015`
- `full_mlp_capacity_search_hd512_depth4`, compile：`0.0826 ms -> 0.6687 ms`，比值 `8.098`

因此“`batch=128` 只比 `batch=1` 慢约 2 倍”不是普适规律，而是小模型、浅层模型在 CPU 上的典型现象。

## 6. BF16 为什么通常比 FP32 更慢

### 6.1 总体结论

在这份单线程 CPU 测试中，`BF16` 大多数情况下比 `FP32` 更慢。

`BF16 latency / FP32 latency` 的中位数：

- eager：`1.827`
- jit：`1.970`
- compile：`1.831`

`BF16 throughput / FP32 throughput` 的中位数：

- eager：`0.548`
- jit：`0.503`
- compile：`0.548`

也就是典型情况下：

- BF16 延迟大约是 FP32 的 `1.8x` 到 `2.0x`
- BF16 吞吐只有 FP32 的 `50%` 到 `55%`

### 6.2 原因

CPU 上 BF16 在这个 benchmark 里是通过：

- `torch.autocast(device_type='cpu', dtype=torch.bfloat16)`

来启用的。也就是说，这里不是“整张图预先转成纯 BF16”，而是运行时混合精度/autocast 选择。

这会带来几个现实影响：

1. 部分操作仍可能在 FP32 中执行。
2. 可能存在额外的 dtype 转换和 dispatch 开销。
3. 对于小型 dense MLP，FP32 kernel 本身已经非常高效。
4. 在小 batch、小矩阵下，BF16 的硬件优势不足以抵消这些额外开销。

### 6.3 batch 越大，BF16 越“不吃亏”

按 batch 汇总的 BF16/FP32 latency 中位数：

| mode | bs1 | bs2 | bs4 | bs8 | bs16 | bs32 | bs64 | bs128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | 1.884 | 2.097 | 1.987 | 2.013 | 1.872 | 1.702 | 1.575 | 1.515 |
| jit | 1.926 | 2.058 | 2.200 | 2.316 | 1.998 | 1.882 | 1.720 | 1.520 |
| compile | 1.748 | 2.036 | 2.063 | 2.026 | 1.833 | 1.699 | 1.509 | 1.447 |

可以看出：

1. BF16 在小 batch 最差。
2. 随着 batch 增大，BF16 会逐渐接近 FP32。
3. 但只有在最大模型、最大 batch 上，BF16 才真正开始反超。

### 6.4 BF16 真正占优的区域

最明显的 BF16 胜出点集中在大模型、大 batch：

- eager，`hd512_depth5`, `bs=128`：ratio `0.491`，throughput `2.022x`
- jit，`hd512_depth5`, `bs=128`：ratio `0.430`，throughput `2.308x`
- compile，`hd512_depth5`, `bs=128`：ratio `0.453`，throughput `2.197x`

所以更准确的说法不是“BF16 不行”，而是：

- 这组 CPU workload 太小，绝大多数点还没大到让 BF16 的算术优势回本

## 7. 为什么 `jit < compile < eager`

### 7.1 总体排序

单线程 `FP32` 下的中位数比值：

- `jit / eager = 0.600`
- `compile / eager = 0.940`
- `compile / jit = 1.563`

单线程 `BF16` 下：

- `jit / eager = 0.641`
- `compile / eager = 0.919`
- `compile / jit = 1.444`

因此典型排序是：

- `jit < compile < eager`

### 7.2 稳定性

对 `FP32`：

- `jit < compile < eager`：`148 / 160`
- `jit < eager < compile`：`11 / 160`
- `compile < jit < eager`：`1 / 160`

对 `BF16`：

- `jit < compile < eager`：`136 / 160`
- `jit < eager < compile`：`24 / 160`

对 `FP32` 来说，JIT 几乎在所有 batch 上都是最优。

### 7.3 为什么 JIT 这么适合这类模型

这些 `full_mlp` 几乎是 TorchScript 推理优化的理想对象：

1. 每个 benchmark 点 shape 固定。
2. 结构高度规则，几乎全是 `Linear + ReLU`。
3. 控制流极少。
4. 完全是 inference-only 场景。

而 benchmark 的 JIT 路径还做了：

- tracing
- freeze
- optimize_for_inference

这条链路很适合静态 CPU 图推理，因此比 eager 去掉了更多框架层面的开销，也比通用的 `torch.compile` 更贴合这个 workload。

## 8. 多线程影响：为什么“基本是坏事”这个判断大体成立

### 8.1 整体趋势

相对 `1` 线程的中位数表现如下：

| 线程数 | latency 相对 1T | throughput 相对 1T |
| --- | ---: | ---: |
| 2 | 1.104 | 0.908 |
| 4 | 1.122 | 0.893 |
| 8 | 1.089 | 0.918 |
| 128 | 2.561 | 0.378 |

这说明：

- `2/4/8` 线程整体上对这组 benchmark 也是略有负收益
- `128` 线程则是明显的高风险配置

### 8.2 什么时候线程有用

虽然总体趋势是负收益，但在线程最有价值的少数场景里，受益点主要集中在最大模型：

- `hd512_depth5`, jit, fp32, `bs=2`：`8` 线程 throughput 提升 `3.00x`
- `hd512_depth5`, eager, fp32, `bs=128`：`8` 线程 throughput 提升 `2.57x`
- `hd512_depth5`, jit, fp32, `bs=128`：`8` 线程 throughput 提升 `2.53x`
- `hd512_depth4`, compile, fp32, `bs=128`：`8` 线程 throughput 提升 `2.36x`

因此更准确的表述是：

- 对大多数点，多线程是负收益
- 对最大的模型，`8` 线程有时能显著提升吞吐

### 8.3 为什么 128 线程会特别糟糕

最极端的退化例子包括：

- `hd256_depth3`, jit, fp32, `bs=64`：`0.0535 ms -> 33.9857 ms`，慢了 `635.68x`
- `hd512_depth3`, jit, fp32, `bs=32`：`0.0541 ms -> 33.2910 ms`，慢了 `615.62x`
- `hd512_depth3`, jit, fp32, `bs=64`：`0.0832 ms -> 30.1409 ms`，慢了 `362.12x`

这不是一般意义上的“扩展性差”，而是调度和同步开销已经远远压过了计算本身。

最可能的原因包括：

1. 小 kernel 被过度并行化。
2. 线程唤醒和同步成本远大于实际算术工作量。
3. cache/跨核协调成本过高。
4. 对于这种很短的 inference 图，全核 fan-out 完全不划算。

## 9. 用相邻 depth 增量来观察固定开销影响

这是这次新增的重点分析。

思路很简单：如果固定开销占比较大，那么在同一 `hidden_dim`、同一 `mode`、同一 `batch` 下，随着 depth 从 `2 -> 3 -> 4 -> 5` 增加，latency 的相邻增量不一定严格按“新增层数”线性放大；但如果算术部分开始主导，总体会更接近“每多一层，增加一个相对稳定的时间”。

这里定义：

- `delta23 = latency(depth3) - latency(depth2)`
- `delta34 = latency(depth4) - latency(depth3)`
- `delta45 = latency(depth5) - latency(depth4)`

### 9.1 整体结论

结论不是“完全线性”，而是：

1. 对中小模型和多数常见配置，相邻 depth 增量已经有明显的“近似常数”特征。
2. 对大模型，尤其 `hd512`，增量会越来越大，说明算术量开始主导，固定开销占比在下降。
3. 因此“存在固定开销”这个解释是成立的，但不能把所有 depth 增长都简单看成常数级增加。

### 9.2 batch=1 时的相邻增量

单线程 `FP32`，按 mode 汇总后的相邻增量中位数：

| mode | d2->3 | d3->4 | d4->5 |
| --- | ---: | ---: | ---: |
| eager | 0.0078 ms | 0.0070 ms | 0.0075 ms |
| jit | 0.0023 ms | 0.0025 ms | 0.0023 ms |
| compile | 0.0072 ms | 0.0081 ms | 0.0099 ms |

这组结果非常有代表性：

- eager 和 jit 在 batch=1 时已经表现出相当明显的“近似线性增量”
- 特别是 jit，`2->3`、`3->4`、`4->5` 三段几乎一样

例如一些最线性的例子：

- eager, `bs=1`, `hd64`：`0.0074 / 0.0066 / 0.0069 ms`
- eager, `bs=1`, `hd128`：`0.0078 / 0.0070 / 0.0075 ms`
- jit, `bs=1`, `hd128`：`0.0021 / 0.0025 / 0.0023 ms`

这说明在这些点上，可以把总 latency 近似理解为：

- 一个固定底座
- 加上每层差不多固定的增量

### 9.3 batch=128 时的相邻增量

单线程 `FP32`，按 mode 汇总后的相邻增量中位数：

| mode | d2->3 | d3->4 | d4->5 |
| --- | ---: | ---: | ---: |
| eager | 0.0384 ms | 0.0384 ms | 0.0334 ms |
| jit | 0.0262 ms | 0.0252 ms | 0.0245 ms |
| compile | 0.0356 ms | 0.0293 ms | 0.0324 ms |

对中等 hidden_dim，这种“近似线性”依然很明显：

- eager, `bs=128`, `hd128`：`0.0384 / 0.0384 / 0.0334 ms`
- jit, `bs=128`, `hd128`：`0.0262 / 0.0252 / 0.0245 ms`
- compile, `bs=128`, `hd128`：`0.0356 / 0.0293 / 0.0324 ms`

这进一步支持了固定开销存在的判断：

- 对于中等规模模型，depth 增长带来的额外开销相当稳定

### 9.4 为什么在 `hd512` 上不再线性

最不线性的点几乎都集中在 `hd512`，例如：

- eager, `bs=1`, `hd512`：`0.0118 / 0.0437 / 0.0721 ms`
- jit, `bs=1`, `hd512`：`0.0050 / 0.0483 / 0.0589 ms`
- compile, `bs=1`, `hd512`：`0.0072 / 0.0447 / 0.0683 ms`

还有：

- jit, `bs=128`, `hd512`：`0.1252 / 0.3779 / 0.3724 ms`
- compile, `bs=128`, `hd512`：`0.1407 / 0.4856 / 0.4217 ms`

原因很直接：

1. `depth=2` 在当前实现里没有隐藏层，结构是 `24 -> 144`，非常轻。
2. 一旦进入 `depth=3`，结构变成 `24 -> 512 -> 144`，算术量骤增。
3. 再从 `depth=3` 到 `4`、`4` 到 `5`，新增的是 `512 -> 512` 的隐藏层，这部分代价远大于 `depth=2` 时的单层线性映射。

因此，对 `hd512` 来说，`depth=2 -> 3` 不只是“多一层”，而是“从无隐藏层切换到大隐藏层架构”，所以它天然比中等 hidden_dim 更不线性。

### 9.5 用这组结果怎么简洁解释固定开销

最简洁、又不失准确的说法可以写成：

- 在中小规模 full-MLP 上，`depth=2->3->4->5` 的相邻 latency 增量已经表现出明显的近似线性特征，说明总 latency 可以看成“固定开销 + 每增加一层带来的近似固定计算开销”。
- 但在 `hd512` 这类大隐藏层配置上，相邻增量显著变大，说明算术量开始主导，总延迟不再由固定开销主导。

这样既能支持“固定开销存在”的判断，又不会把结论说得过头。

## 10. 冷启动角度：graph preparation 成本

steady-state latency 之外，还要看图准备时间。

中位数如下：

| mode | prep 中位数 |
| --- | ---: |
| eager | 0.0 ms |
| jit | 49.506 ms |
| compile | 512.293 ms |

这意味着：

1. JIT 在 steady-state latency 上最好，同时冷启动成本可接受。
2. compile 不仅 warm latency 不占优，冷启动还明显更重。
3. 如果场景是短生命周期 worker 或频繁 reload model，compile 会更不划算。

## 11. 推荐结论

### 11.1 默认 CPU 推理配置

对这组 `full_mlp`，默认最优选择是：

- precision：`fp32`
- execution mode：`jit`
- threads：`1`

### 11.2 什么时候值得试多线程

只有在最大模型、目标是纯吞吐时，才建议额外测试：

- `8` 线程

不建议把 `128` 线程当默认配置。

### 11.3 什么时候值得试 BF16

只有在下面条件同时成立时，才值得认真比较 BF16：

1. 模型较大，例如 `hd256/hd512`
2. depth 较深，例如 `4/5`
3. batch 较大，例如 `64/128`

## 12. 一句话总结

这份 CPU benchmark 的主要规律是：

- 小模型上，单线程已经很高效，batch 放大只会让 latency 缓慢上升，但吞吐大幅提升；
- BF16 在大多数点都还没大到能回本，因此普遍慢于 FP32；
- JIT 非常适合这类静态 dense MLP 推理；
- 多线程整体上大多是坏事，尤其 `128` 线程风险极高；
- `depth=2` 当前就是“无隐藏层”，因此不同 `hidden_dim` 的 `depth=2` 历史组合没有结构差异；
- 从 `depth=2->3->4->5` 的相邻增量看，中小模型已经呈现出相当明显的近似线性增长，这很好地支持了“固定开销 + 每层增量开销”的解释。

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