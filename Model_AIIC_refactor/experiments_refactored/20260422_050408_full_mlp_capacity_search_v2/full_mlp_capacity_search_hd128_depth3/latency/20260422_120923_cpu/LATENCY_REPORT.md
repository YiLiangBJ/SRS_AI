# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32']`
- Batch sizes: `[1]`
- Thread counts: `[1]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd128_depth3

- Best throughput config: threads=`1`, batch=`1`, precision=`fp32`, throughput=`8068.030` samples/s, p50=`0.124` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.124` ms, throughput=`8068.030` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `21,776`
- MACs / sample: `21,504`
- FLOPs / sample estimate: `43,352`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth3` | `fp32` | 1 | 1 | ok | 0.123946 | 0.123946 | 0.123946 | 8068.0296258047865 | - |
