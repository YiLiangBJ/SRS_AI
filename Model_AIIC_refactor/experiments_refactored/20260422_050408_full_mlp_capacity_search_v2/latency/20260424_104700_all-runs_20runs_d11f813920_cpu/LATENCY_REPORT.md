# Latency Report

- Device: `cpu`
- Runtime backends: `['pytorch', 'onnxruntime']`
- Execution modes: `['jit']`
- Precision profiles: `['fp32']`
- Batch sizes: `[1, 128]`
- Thread counts: `[1]`

## Hardware Summary

- Runtime backend: `pytorch`
- Hostname: `nex-flexran-gpu-01`
- CPU model: `Intel(R) Xeon(R) Platinum 8358 CPU $@ $@`
- CPU capability: `AVX512`
- CPU flag summary: `['avx2', 'avx512f', 'avx512bw', 'avx512vl', 'avx512_vnni', 'fma']`
- Logical CPU count: `128`
- Physical CPU count: `64`
- mkldnn available: `True`
- mkldnn enabled: `True`
- oneDNN version: `None`
- torch.compile available: `True`
- Python: `3.11.9`
- PyTorch: `2.1.2+cu121`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd128_depth2::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1988034.517` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.043` ms, throughput=`23411.966` samples/s

### full_mlp_capacity_search_hd128_depth2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3880998.872` samples/s, p50=`0.032` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.014` ms, throughput=`68318.144` samples/s

### full_mlp_capacity_search_hd128_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1249934.086` samples/s, p50=`0.103` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.049` ms, throughput=`20027.478` samples/s

### full_mlp_capacity_search_hd128_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1940440.601` samples/s, p50=`0.065` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.016` ms, throughput=`62609.567` samples/s

### full_mlp_capacity_search_hd128_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`999504.933` samples/s, p50=`0.129` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.049` ms, throughput=`19553.400` samples/s

### full_mlp_capacity_search_hd128_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1364855.816` samples/s, p50=`0.093` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.017` ms, throughput=`58478.848` samples/s

### full_mlp_capacity_search_hd128_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`825671.148` samples/s, p50=`0.154` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.057` ms, throughput=`17223.740` samples/s

### full_mlp_capacity_search_hd128_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1058285.049` samples/s, p50=`0.120` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.018` ms, throughput=`53698.195` samples/s

### full_mlp_capacity_search_hd256_depth2::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1919564.259` samples/s, p50=`0.066` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.041` ms, throughput=`23350.301` samples/s

### full_mlp_capacity_search_hd256_depth2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3856653.028` samples/s, p50=`0.033` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.014` ms, throughput=`69205.110` samples/s

### full_mlp_capacity_search_hd256_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`915176.050` samples/s, p50=`0.139` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.048` ms, throughput=`20594.692` samples/s

### full_mlp_capacity_search_hd256_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1245790.590` samples/s, p50=`0.102` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.016` ms, throughput=`58586.427` samples/s

### full_mlp_capacity_search_hd256_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`498155.269` samples/s, p50=`0.256` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.060` ms, throughput=`16115.308` samples/s

### full_mlp_capacity_search_hd256_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`618397.519` samples/s, p50=`0.207` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.020` ms, throughput=`48008.603` samples/s

### full_mlp_capacity_search_hd256_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`351559.882` samples/s, p50=`0.362` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.074` ms, throughput=`13317.140` samples/s

### full_mlp_capacity_search_hd256_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`405948.155` samples/s, p50=`0.313` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.028` ms, throughput=`36153.813` samples/s

### full_mlp_capacity_search_hd32_depth2::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1987799.878` samples/s, p50=`0.063` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.044` ms, throughput=`22168.034` samples/s

### full_mlp_capacity_search_hd32_depth2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3869688.248` samples/s, p50=`0.033` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.014` ms, throughput=`68506.289` samples/s

### full_mlp_capacity_search_hd32_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1773713.019` samples/s, p50=`0.071` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.045` ms, throughput=`21397.789` samples/s

### full_mlp_capacity_search_hd32_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3375919.147` samples/s, p50=`0.037` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.015` ms, throughput=`66275.201` samples/s

### full_mlp_capacity_search_hd32_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1683249.302` samples/s, p50=`0.076` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.049` ms, throughput=`20031.811` samples/s

### full_mlp_capacity_search_hd32_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3092564.316` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.015` ms, throughput=`64377.406` samples/s

### full_mlp_capacity_search_hd32_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1520529.524` samples/s, p50=`0.084` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.049` ms, throughput=`19814.143` samples/s

### full_mlp_capacity_search_hd32_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2891139.561` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.016` ms, throughput=`60830.211` samples/s

### full_mlp_capacity_search_hd512_depth2::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2057189.879` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.041` ms, throughput=`23427.434` samples/s

### full_mlp_capacity_search_hd512_depth2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3652342.635` samples/s, p50=`0.033` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.014` ms, throughput=`68541.036` samples/s

### full_mlp_capacity_search_hd512_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`576892.048` samples/s, p50=`0.220` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.053` ms, throughput=`19490.746` samples/s

### full_mlp_capacity_search_hd512_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`709694.876` samples/s, p50=`0.180` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.019` ms, throughput=`52541.981` samples/s

### full_mlp_capacity_search_hd512_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`206857.253` samples/s, p50=`0.617` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.108` ms, throughput=`9266.741` samples/s

### full_mlp_capacity_search_hd512_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`217036.114` samples/s, p50=`0.590` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.058` ms, throughput=`17133.615` samples/s

### full_mlp_capacity_search_hd512_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`128359.792` samples/s, p50=`0.998` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.158` ms, throughput=`6298.633` samples/s

### full_mlp_capacity_search_hd512_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`130038.087` samples/s, p50=`0.985` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.106` ms, throughput=`9288.691` samples/s

### full_mlp_capacity_search_hd64_depth2::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1976516.513` samples/s, p50=`0.063` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.043` ms, throughput=`22808.034` samples/s

### full_mlp_capacity_search_hd64_depth2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3861376.581` samples/s, p50=`0.033` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.015` ms, throughput=`66887.842` samples/s

### full_mlp_capacity_search_hd64_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1584797.827` samples/s, p50=`0.080` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.048` ms, throughput=`20345.300` samples/s

### full_mlp_capacity_search_hd64_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2727547.807` samples/s, p50=`0.047` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.015` ms, throughput=`65674.543` samples/s

### full_mlp_capacity_search_hd64_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1396111.829` samples/s, p50=`0.091` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.050` ms, throughput=`20155.439` samples/s

### full_mlp_capacity_search_hd64_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2303931.804` samples/s, p50=`0.055` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.016` ms, throughput=`60885.025` samples/s

### full_mlp_capacity_search_hd64_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1138026.626` samples/s, p50=`0.113` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.056` ms, throughput=`18034.655` samples/s

### full_mlp_capacity_search_hd64_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2011408.457` samples/s, p50=`0.063` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.017` ms, throughput=`58637.270` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd128_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `21,776`
- MACs / sample: `21,504`
- FLOPs / sample estimate: `43,352`

### full_mlp_capacity_search_hd128_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,288`
- MACs / sample: `37,888`
- FLOPs / sample estimate: `76,248`

### full_mlp_capacity_search_hd128_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `54,800`
- MACs / sample: `54,272`
- FLOPs / sample estimate: `109,144`

### full_mlp_capacity_search_hd256_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd256_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `43,408`
- MACs / sample: `43,008`
- FLOPs / sample estimate: `86,488`

### full_mlp_capacity_search_hd256_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `109,200`
- MACs / sample: `108,544`
- FLOPs / sample estimate: `217,816`

### full_mlp_capacity_search_hd256_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `174,992`
- MACs / sample: `174,080`
- FLOPs / sample estimate: `349,144`

### full_mlp_capacity_search_hd32_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd32_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `5,552`
- MACs / sample: `5,376`
- FLOPs / sample estimate: `11,000`

### full_mlp_capacity_search_hd32_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `6,608`
- MACs / sample: `6,400`
- FLOPs / sample estimate: `13,080`

### full_mlp_capacity_search_hd32_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `7,664`
- MACs / sample: `7,424`
- FLOPs / sample estimate: `15,160`

### full_mlp_capacity_search_hd512_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd512_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `86,672`
- MACs / sample: `86,016`
- FLOPs / sample estimate: `172,760`

### full_mlp_capacity_search_hd512_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `349,328`
- MACs / sample: `348,160`
- FLOPs / sample estimate: `697,560`

### full_mlp_capacity_search_hd512_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `611,984`
- MACs / sample: `610,304`
- FLOPs / sample estimate: `1,222,360`

### full_mlp_capacity_search_hd64_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

### full_mlp_capacity_search_hd64_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,960`
- MACs / sample: `10,752`
- FLOPs / sample estimate: `21,784`

### full_mlp_capacity_search_hd64_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `15,120`
- MACs / sample: `14,848`
- FLOPs / sample estimate: `30,040`

### full_mlp_capacity_search_hd64_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `19,280`
- MACs / sample: `18,944`
- FLOPs / sample estimate: `38,296`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 53.572976 | 0.042531 | 0.0474524 | 0.04839368 | 23411.966324227637 | - |
| `full_mlp_capacity_search_hd128_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 54.671838 | 0.062423 | 0.0705184 | 0.07082048 | 1988034.5172493057 | - |
| `full_mlp_capacity_search_hd128_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.131745 | 0.014376 | 0.016569999999999998 | 0.016825999999999997 | 68318.14393266564 | - |
| `full_mlp_capacity_search_hd128_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.057887 | 0.032451 | 0.034891399999999996 | 0.03504548 | 3880998.8720847024 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 58.743408 | 0.049389 | 0.053420999999999996 | 0.0536786 | 20027.477699403582 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 60.321241 | 0.10276 | 0.1069122 | 0.10693443999999999 | 1249934.0855072096 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.342946 | 0.015869 | 0.0179348 | 0.018180560000000002 | 62609.56674179815 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.150252 | 0.065081 | 0.0683864 | 0.06868128 | 1940440.6012940311 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 62.885215 | 0.049035 | 0.0556704 | 0.055774080000000004 | 19553.400336318486 | - |
| `full_mlp_capacity_search_hd128_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 64.918131 | 0.128881 | 0.1319494 | 0.13223228 | 999504.9327130157 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.442082 | 0.016565 | 0.0192554 | 0.019555879999999998 | 58478.84820060584 | - |
| `full_mlp_capacity_search_hd128_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.431379 | 0.093478 | 0.0957092 | 0.09578264 | 1364855.815778586 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 67.231078 | 0.057441 | 0.0632908 | 0.06377816 | 17223.7398250757 | - |
| `full_mlp_capacity_search_hd128_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 70.01345 | 0.153894 | 0.1586756 | 0.15868392 | 825671.1480828302 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.596085 | 0.018201 | 0.020877 | 0.0211498 | 53698.19466669531 | - |
| `full_mlp_capacity_search_hd128_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.699353 | 0.120329 | 0.12311040000000001 | 0.12322768 | 1058285.049077969 | - |
| `full_mlp_capacity_search_hd256_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 53.368131 | 0.0413 | 0.0462518 | 0.04647276 | 23350.301218885725 | - |
| `full_mlp_capacity_search_hd256_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 54.669847 | 0.066134 | 0.07144980000000001 | 0.07174436000000001 | 1919564.258913227 | - |
| `full_mlp_capacity_search_hd256_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.139271 | 0.01437 | 0.0162068 | 0.016408560000000003 | 69205.11010533018 | - |
| `full_mlp_capacity_search_hd256_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.058421 | 0.032644 | 0.0349928 | 0.03516176 | 3856653.027773928 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 59.35342 | 0.048336 | 0.052462800000000004 | 0.05270056 | 20594.692335891195 | - |
| `full_mlp_capacity_search_hd256_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 60.798425 | 0.139004 | 0.1444678 | 0.14470636 | 915176.0498427756 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.439344 | 0.016374 | 0.0190742 | 0.01933004 | 58586.42669666292 | - |
| `full_mlp_capacity_search_hd256_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.364663 | 0.102209 | 0.1047172 | 0.10480984 | 1245790.590387947 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 64.278079 | 0.060222 | 0.0712684 | 0.07272328 | 16115.308253616276 | - |
| `full_mlp_capacity_search_hd256_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 66.608062 | 0.255648 | 0.2635816 | 0.26397632 | 498155.26877033483 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.70222 | 0.020307 | 0.022949 | 0.0231186 | 48008.60314168299 | - |
| `full_mlp_capacity_search_hd256_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.977707 | 0.20652 | 0.21016479999999998 | 0.21032656 | 618397.5194529501 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 68.419292 | 0.073959 | 0.0797448 | 0.07992576 | 13317.13969146851 | - |
| `full_mlp_capacity_search_hd256_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 72.118499 | 0.361604 | 0.3787734 | 0.38189148 | 351559.8821834945 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.10667 | 0.027826 | 0.0295952 | 0.029683039999999997 | 36153.8127810959 | - |
| `full_mlp_capacity_search_hd256_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.389087 | 0.313005 | 0.3254338 | 0.32748436 | 405948.1553462104 | - |
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 53.70289 | 0.043716 | 0.0499216 | 0.05069392 | 22168.033695411217 | - |
| `full_mlp_capacity_search_hd32_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 54.719938 | 0.063299 | 0.0689142 | 0.06912124 | 1987799.8782472573 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.076805 | 0.014169 | 0.016416 | 0.0166696 | 68506.28887731893 | - |
| `full_mlp_capacity_search_hd32_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.075541 | 0.032645 | 0.034915 | 0.0350774 | 3869688.2482405016 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 58.683348 | 0.0449 | 0.051533999999999996 | 0.0515684 | 21397.78918042188 | - |
| `full_mlp_capacity_search_hd32_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 60.872842 | 0.070691 | 0.0775894 | 0.07809948 | 1773713.018776415 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.33534 | 0.01467 | 0.0170604 | 0.01728328 | 66275.20114523548 | - |
| `full_mlp_capacity_search_hd32_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.312308 | 0.037406 | 0.039770599999999996 | 0.03990212 | 3375919.1467364356 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 63.046676 | 0.048583 | 0.0561132 | 0.05656424 | 20031.810515097975 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 64.27192 | 0.076258 | 0.0808018 | 0.08109875999999999 | 1683249.3023720668 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.420058 | 0.01487 | 0.0177506 | 0.01807012 | 64377.4061055532 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.185815 | 0.040773 | 0.043358400000000005 | 0.04343728 | 3092564.3156735026 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 66.883645 | 0.04857 | 0.0568482 | 0.05800164 | 19814.14333551289 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 68.931559 | 0.084054 | 0.0892674 | 0.08969668 | 1520529.5244068748 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.523717 | 0.016121 | 0.0183886 | 0.01860092 | 60830.21071584992 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.273109 | 0.043854 | 0.046327 | 0.0465278 | 2891139.5607274827 | - |
| `full_mlp_capacity_search_hd512_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 53.080073 | 0.040985 | 0.0473086 | 0.04758172 | 23427.433524657372 | - |
| `full_mlp_capacity_search_hd512_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 55.076369 | 0.062369 | 0.06679260000000001 | 0.06699212 | 2057189.878625797 | - |
| `full_mlp_capacity_search_hd512_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.166485 | 0.014057 | 0.0170156 | 0.01742152 | 68541.0355179646 | - |
| `full_mlp_capacity_search_hd512_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.062674 | 0.032583 | 0.0425022 | 0.04416284 | 3652342.6353934826 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 59.253763 | 0.052652 | 0.055547599999999996 | 0.055855919999999996 | 19490.74579389706 | - |
| `full_mlp_capacity_search_hd512_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 62.10914 | 0.220345 | 0.22691640000000002 | 0.22711928 | 576892.0481740915 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.612151 | 0.018659 | 0.0210886 | 0.02133452 | 52541.98104285324 | - |
| `full_mlp_capacity_search_hd512_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.595126 | 0.180094 | 0.1825188 | 0.18266296 | 709694.8755594392 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 64.874615 | 0.108108 | 0.1146196 | 0.11548232 | 9266.741294823227 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 69.365419 | 0.617398 | 0.6278992 | 0.6294510400000001 | 206857.2533041406 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.144412 | 0.057571 | 0.0607836 | 0.06080712 | 17133.614781512144 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.911647 | 0.589547 | 0.5925312 | 0.59266384 | 217036.11413115353 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 69.878645 | 0.157501 | 0.1629864 | 0.16314688 | 6298.633322541675 | - |
| `full_mlp_capacity_search_hd512_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 76.872424 | 0.998493 | 0.9997777999999999 | 0.99992996 | 128359.79249837293 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.443235 | 0.105961 | 0.1139056 | 0.11497231999999999 | 9288.690647588934 | - |
| `full_mlp_capacity_search_hd512_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 8.113067 | 0.985162 | 0.988143 | 0.9885974 | 130038.08693655663 | - |
| `full_mlp_capacity_search_hd64_depth2` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 53.192838 | 0.042618 | 0.0487774 | 0.04933468 | 22808.03390186159 | - |
| `full_mlp_capacity_search_hd64_depth2` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 55.114693 | 0.062978 | 0.071832 | 0.0724648 | 1976516.5131778063 | - |
| `full_mlp_capacity_search_hd64_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.328718 | 0.015105 | 0.0164364 | 0.01646408 | 66887.84246575342 | - |
| `full_mlp_capacity_search_hd64_depth2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.069456 | 0.032558 | 0.0350786 | 0.035250119999999996 | 3861376.580751038 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 58.473644 | 0.048363 | 0.053649199999999994 | 0.054545039999999996 | 20345.300439051585 | - |
| `full_mlp_capacity_search_hd64_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 60.916417 | 0.080396 | 0.0847864 | 0.08509567999999999 | 1584797.8268459798 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.333739 | 0.014735 | 0.0171124 | 0.01730728 | 65674.54323355181 | - |
| `full_mlp_capacity_search_hd64_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.093914 | 0.046638 | 0.0490382 | 0.049265239999999995 | 2727547.8066680017 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 62.882539 | 0.049634 | 0.0532 | 0.0536408 | 20155.43874359057 | - |
| `full_mlp_capacity_search_hd64_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 64.982569 | 0.090808 | 0.0960142 | 0.09615644 | 1396111.8285574673 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.460786 | 0.015628 | 0.0190048 | 0.01937616 | 60885.02471932004 | - |
| `full_mlp_capacity_search_hd64_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.212846 | 0.055383 | 0.057517399999999996 | 0.05763388 | 2303931.8036186127 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 95.50395 | 0.055528 | 0.0611402 | 0.06202564 | 18034.655393804736 | - |
| `full_mlp_capacity_search_hd64_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 69.887804 | 0.112819 | 0.1172508 | 0.11759336000000001 | 1138026.6262667214 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 3.589226 | 0.016542 | 0.0190022 | 0.01909324 | 58637.26984871584 | - |
| `full_mlp_capacity_search_hd64_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.292687 | 0.063383 | 0.0654276 | 0.06554872 | 2011408.457343998 | - |
