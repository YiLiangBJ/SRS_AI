# CPU 部署后端选择摘要

## 结论一句话

这次指导性测试说明：

- 小 batch CPU 部署，优先看 ONNX Runtime。
- 小模型在整个测试 batch 范围内，ONNX Runtime 都是最优。
- 大模型在低 batch 下仍然是 ONNX Runtime 更好。
- 大模型从 batch 约 `8` 开始，OpenVINO 开始追平或反超。
- 大模型在 batch `16` 及以上时，OpenVINO 优势明显，尤其高 batch 下最值得关注。

## 测试范围

- 设备：`cpu`
- 精度：`fp32`
- 线程数：`1`
- 后端：`pytorch + jit`、`onnxruntime`、`openvino`
- batch：`1, 2, 4, 8, 16, 32, 64, 128`
- 小模型代表：`full_mlp_capacity_search_hd32_depth2`
- 大模型代表：`full_mlp_capacity_search_hd512_depth5`

## 小模型结果

模型：`full_mlp_capacity_search_hd32_depth2`

关键结论：

- ONNX Runtime 在所有 batch 下都是最快。
- OpenVINO 在这个小模型上没有出现反超点。
- 对小模型 CPU 部署，没有证据表明 OpenVINO 值得优先于 ONNX Runtime。

代表结果：

| Batch | PyTorch JIT | ONNX Runtime | OpenVINO | 最快 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.038277 ms | 0.013817 ms | 0.087931 ms | ONNX Runtime |
| 8 | 0.043993 ms | 0.015417 ms | 0.099812 ms | ONNX Runtime |
| 32 | 0.047215 ms | 0.019346 ms | 0.138435 ms | ONNX Runtime |
| 128 | 0.059075 ms | 0.033757 ms | 0.115284 ms | ONNX Runtime |

## 大模型结果

模型：`full_mlp_capacity_search_hd512_depth5`

关键结论：

- batch `1/2/4` 时，ONNX Runtime 最快。
- batch `8` 附近开始出现转折，OpenVINO 与 ONNX Runtime 基本打平并略快。
- batch `16` 以后，OpenVINO 明显成为最优后端。
- batch 越大，OpenVINO 优势越明显。

代表结果：

| Batch | PyTorch JIT | ONNX Runtime | OpenVINO | 最快 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.156535 ms | 0.108534 ms | 0.135717 ms | ONNX Runtime |
| 8 | 0.179724 ms | 0.143913 ms | 0.143604 ms | OpenVINO |
| 16 | 0.231667 ms | 0.165528 ms | 0.155414 ms | OpenVINO |
| 32 | 0.340651 ms | 0.292865 ms | 0.132299 ms | OpenVINO |
| 128 | 0.991357 ms | 0.986199 ms | 0.253385 ms | OpenVINO |

## 工程判断

如果目标是做 CPU 部署选型，可以直接按下面理解：

- 低 batch 延迟优先：默认先看 ONNX Runtime。
- 小模型部署：优先 ONNX Runtime，不必优先投入 OpenVINO。
- 大模型高 batch 部署：应认真比较 OpenVINO，因为它可能明显优于 ONNX Runtime。
- PyTorch JIT 更适合作为开发期参考基线，而不是最终部署路径。

## 建议的后续测试策略

如果后续要在别的机器上继续扩大测试，最有效的做法是：

- 小模型：以 ONNX Runtime 为主，不必大范围补 OpenVINO。
- 大模型：重点测试 `8, 16, 32, 64, 128` 这些 batch 区间。
- 如果部署场景主要是 `batch=1`，优先验证 ONNX Runtime 即可。
- 如果部署场景允许聚合到较大 batch，应把 OpenVINO 纳入正式候选。