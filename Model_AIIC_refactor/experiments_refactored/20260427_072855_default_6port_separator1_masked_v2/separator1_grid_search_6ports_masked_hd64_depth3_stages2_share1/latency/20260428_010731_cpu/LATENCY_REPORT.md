# Latency Report

- Device: `cpu`
- Runtime backends: `['pytorch', 'onnxruntime', 'openvino']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8]`

## Hardware Summary

- Runtime backend: `pytorch`
- Hostname: `sh14l07002s1404`
- CPU model: `Intel(R) Xeon(R) 6760P`
- CPU capability: `AVX512`
- CPU flag summary: `['avx2', 'avx512f', 'avx512bw', 'avx512vl', 'avx512_vnni', 'avx512_bf16', 'amx_bf16', 'amx_int8', 'amx_tile', 'fma']`
- Logical CPU count: `256`
- Physical CPU count: `128`
- mkldnn available: `True`
- mkldnn enabled: `True`
- oneDNN version: `None`
- torch.compile available: `True`
- Python: `3.11.9`
- PyTorch: `2.1.2+cu121`

## CPU Thread Scaling Highlights

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`121040.326` samples/s, p50=`1.056` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.540` ms, throughput=`1834.615` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`222698.922` samples/s, p50=`0.572` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.155` ms, throughput=`6426.165` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`152304.409` samples/s, p50=`0.815` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.474` ms, throughput=`2112.076` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`297111.027` samples/s, p50=`0.430` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.056` ms, throughput=`17795.178` samples/s

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`38063.161` samples/s, p50=`3.331` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.676` ms, throughput=`1446.900` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `78,480`
- MACs / sample: `153,600`
- FLOPs / sample estimate: `311,064`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.5612165 | 0.57525215 | 0.59322337 | 1775.5359623978416 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.5425005 | 0.55016875 | 0.55918814 | 1840.3017093679232 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.5437325 | 0.5538835 | 0.5988148099999999 | 1830.2400385433912 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.539511 | 0.56533265 | 0.66755184 | 1834.6146955058803 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.583515 | 0.59640305 | 0.6704424599999999 | 3406.7181299134245 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.5718435 | 0.6198684999999998 | 0.69207359 | 3460.5323516864282 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.5794765 | 0.59707165 | 0.6881390499999999 | 3419.768670432242 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.5697135 | 0.6238164499999999 | 0.69222359 | 3467.389668267549 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.610268 | 0.6821436499999999 | 0.71570413 | 6469.476379666785 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.586473 | 0.6730111999999999 | 0.6974553 | 6698.961915463656 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.597906 | 0.7004445499999999 | 0.7068346099999999 | 6549.9028583907075 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.595018 | 0.70122845 | 0.70827217 | 6605.76307248309 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.6249210000000001 | 0.7513142 | 0.75545791 | 12394.412772244052 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.6272465 | 0.72705055 | 0.75467204 | 12537.445824521208 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.6139805 | 0.73279175 | 0.73764407 | 12767.756613873482 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.6337305 | 0.64922835 | 0.6508653099999999 | 12597.960560257761 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.665173 | 0.67979975 | 0.6917296199999999 | 24025.63872010377 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.650474 | 0.67147605 | 0.7813799699999999 | 24360.236250443202 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.655672 | 0.675737 | 0.68301428 | 24332.161993193382 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.6593415 | 0.67955475 | 0.68443266 | 24178.356725843092 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.7134365 | 0.7397328 | 0.74608327 | 44676.47045272762 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.6984535000000001 | 0.72160795 | 0.72906678 | 45703.50621589108 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.8445125 | 0.8963665 | 0.901503 | 37465.10447483911 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.7160569999999999 | 0.74651625 | 0.7757010999999998 | 44429.67034489726 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.8206555 | 0.84395865 | 0.85154561 | 77595.49598883421 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.9717465000000001 | 1.08190065 | 1.13126411 | 64338.53780499307 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.971609 | 1.0403481 | 1.05303291 | 65183.753918944036 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.9832255000000001 | 1.0455424 | 1.07261959 | 64462.884768886855 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.056191 | 1.0770787 | 1.09184521 | 121040.32647602058 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 1.370545 | 1.42237985 | 1.44215522 | 92694.82377185985 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.3172615 | 1.3993056 | 1.41622356 | 96164.06214067603 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 1.233988 | 1.2926657 | 1.3519823699999998 | 103157.24847704038 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.578379 | 0.5824726 | 0.5831311 | 1730.1796587414406 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.5734925 | 0.58068715 | 0.59274357 | 1741.127007832843 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.5791175 | 0.5868649 | 0.5995693599999999 | 1723.9758773020164 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.5745145 | 0.5809713 | 0.59644555 | 1738.4932089935173 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.933624 | 0.9469181999999999 | 0.95152689 | 2139.435848737949 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.935283 | 0.9451051 | 0.94613753 | 2136.237020624407 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.905184 | 0.92174085 | 0.93015073 | 2203.5088013099416 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.895964 | 0.91609735 | 0.9560131299999999 | 2224.2263490171913 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 1.3608995 | 1.41822155 | 1.4279600399999999 | 2937.300055231521 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 1.438557 | 1.4942609 | 1.5216685399999998 | 2778.530991220648 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 1.3594145000000002 | 1.3954752000000001 | 1.40269117 | 2933.445590429188 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 1.4070725 | 1.4475696 | 1.45538433 | 2837.841843413251 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.5806015 | 1.63228215 | 1.6365957199999999 | 5039.539788218632 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.624024 | 1.6901515999999999 | 1.7542946099999999 | 4897.513170147541 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.7556855 | 1.8184116 | 1.86593498 | 4536.88210671391 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.659615 | 1.7313355 | 1.7672648899999999 | 4804.08029755993 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.5592755 | 1.61187255 | 1.62168988 | 10219.718061995007 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.648798 | 1.7210671499999999 | 1.74329708 | 9647.046290881333 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.7317575 | 1.80969335 | 1.8672858099999998 | 9149.155002914064 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.791754 | 1.8630300499999999 | 1.9343100299999998 | 8893.70571990679 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.6276514999999998 | 1.6980970499999999 | 1.72222958 | 19602.426285412872 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.7774655 | 1.83354025 | 1.8834389399999998 | 17982.808367764886 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.804074 | 1.8516818 | 1.86103068 | 17740.716042132295 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.894307 | 1.9437779999999998 | 2.01602808 | 16885.5182861562 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.759698 | 1.8243382499999998 | 1.83952146 | 36100.56091020886 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 2.243957 | 2.34431985 | 2.37050445 | 28262.41714375614 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.158993 | 2.2626233 | 2.28688626 | 29457.140072917104 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.222603 | 2.3470105 | 2.47325686 | 28620.697569042022 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.855977 | 1.9194536500000001 | 1.97084534 | 68879.78862514866 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.5867445 | 2.6886504999999996 | 2.7220050000000002 | 49497.20813317707 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.6230325 | 2.696471 | 2.7172983 | 48756.852994713336 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.695634 | 3.0528673499999996 | 3.06179267 | 46370.85226764516 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 222.36394 | 0.1556995 | 0.1591177 | 0.16194679 | 6399.992627208494 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 220.761189 | 0.158491 | 0.1622421 | 0.163422 | 6290.376353217213 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 218.700428 | 0.15599049999999998 | 0.1582687 | 0.16185961 | 6395.536018187881 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 219.562624 | 0.1551095 | 0.1591285 | 0.1620405 | 6426.165288682623 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 220.541905 | 0.173375 | 0.17531135 | 0.17648265999999999 | 11528.912494977718 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 220.091398 | 0.17258 | 0.17848995 | 0.1802685 | 11534.824963068375 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 217.842531 | 0.1741895 | 0.17831395 | 0.1796987 | 11447.444667059084 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 218.505205 | 0.1737915 | 0.1765745 | 0.1804603 | 11477.705188944261 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 219.280248 | 0.184919 | 0.1911362 | 0.19179498 | 21572.517906538145 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 220.181187 | 0.1815365 | 0.18337975 | 0.18451926 | 22047.916295526742 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 223.089189 | 0.184115 | 0.1899817 | 0.19174224999999998 | 21642.03298332395 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 219.211944 | 0.182864 | 0.18828609999999998 | 0.19145555 | 21804.232659448084 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 220.00588 | 0.203204 | 0.20785615 | 0.20997612 | 39301.54473739052 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 218.575988 | 0.20451799999999998 | 0.2092522 | 0.2124001 | 39024.382629388754 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 221.092752 | 0.2036655 | 0.2059711 | 0.206643 | 39287.269991450106 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 223.297568 | 0.20458300000000001 | 0.20729055000000002 | 0.20772078 | 39095.92625335431 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 221.047799 | 0.2356205 | 0.23922395 | 0.24035999 | 67777.39125837707 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 222.073072 | 0.22875 | 0.23239275 | 0.23457516 | 69809.06957686716 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 218.723882 | 0.23228749999999998 | 0.23512155 | 0.2357125 | 68805.70413048382 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 223.577733 | 0.2316695 | 0.2377843 | 0.24260529 | 68911.50487910683 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 221.739782 | 0.2775215 | 0.28262045 | 0.28538407 | 115102.69678424647 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 223.629562 | 0.278005 | 0.2827052 | 0.28503280999999997 | 114863.12300052917 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 222.096405 | 0.4115795 | 0.4666504 | 0.4940962299999999 | 76598.24879168656 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 224.649946 | 0.2794025 | 0.2848254 | 0.30218878999999993 | 114111.25447922334 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 222.365476 | 0.371598 | 0.37951409999999997 | 0.38054995999999996 | 171879.004915095 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 226.766701 | 0.5287955 | 0.56328285 | 0.5821981199999999 | 120031.28165238963 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 223.589058 | 0.566682 | 0.6128949499999999 | 0.62017104 | 111531.86895939476 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 224.220712 | 0.566025 | 0.63032875 | 0.6372867999999999 | 111306.2293853037 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 223.618372 | 0.572477 | 0.5849369 | 0.61017753 | 222698.92163614673 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 227.654809 | 0.9658735 | 1.0561991 | 1.0604613 | 130128.64965622248 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 230.404173 | 0.9571305000000001 | 1.0280170499999999 | 1.03751158 | 131890.11476171043 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 226.932546 | 0.7787204999999999 | 0.8276889 | 0.844552 | 162887.45708551785 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 221.043349 | 0.22687449999999998 | 0.22899125 | 0.23283514 | 4401.7343185353775 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 218.439645 | 0.224539 | 0.2258236 | 0.2264545 | 4452.188878592459 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 218.811194 | 0.22422999999999998 | 0.22930615000000001 | 0.23023632 | 4440.193304703559 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 217.372274 | 0.230496 | 0.23695254999999998 | 0.23808714 | 4324.956095208199 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 221.669299 | 0.4299385 | 0.43557935 | 0.43693416 | 4645.421266078907 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 221.909423 | 0.42954 | 0.434927 | 0.4370381 | 4653.3888420577005 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 222.126219 | 0.4372955 | 0.44506755000000003 | 0.44655487 | 4565.5300110093185 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 220.796548 | 0.420786 | 0.42696635 | 0.4488493199999999 | 4736.973394220495 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 227.266758 | 0.6865905 | 0.6980726 | 0.7130782499999999 | 5817.12009857459 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 230.326261 | 0.7487115 | 0.8702882000000001 | 0.87442666 | 5201.798709168448 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 230.343968 | 0.745328 | 0.8202760499999999 | 0.8674561899999998 | 5309.4731274424075 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 231.148106 | 0.7775505 | 0.7952311 | 0.79960425 | 5136.23332428757 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 229.552672 | 0.8255785 | 0.8340192 | 0.85087543 | 9685.12426171811 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 258.448805 | 1.2120225 | 1.2461064 | 1.3309537599999999 | 6575.789691515897 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 236.807041 | 0.9705155 | 1.02085965 | 1.023453 | 8180.2767759835915 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 236.795653 | 0.923497 | 1.0605001 | 1.06421631 | 8457.012024750544 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 232.129649 | 0.8996505 | 0.91428575 | 0.93233471 | 17753.565071209992 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 235.526281 | 1.0157815000000001 | 1.1837275999999999 | 1.20443662 | 15492.228275272497 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 234.937423 | 1.0255365 | 1.0724732999999997 | 1.20016014 | 15474.2275289796 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 238.028833 | 1.0463395 | 1.1711901999999998 | 1.20040223 | 15128.979848350133 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 230.399341 | 1.0026804999999999 | 1.01334005 | 1.02765264 | 31886.135248749426 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 236.87629 | 1.0785125 | 1.1159497999999999 | 1.29797447 | 29398.850828319977 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 232.776021 | 1.0631309999999998 | 1.13329 | 1.3288843499999998 | 29683.490873513896 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 236.137545 | 1.141353 | 1.2608235 | 1.30083004 | 27461.88363497095 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 234.990566 | 1.0900195 | 1.2525905 | 1.2769314299999999 | 56894.61071667244 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 238.888492 | 1.517611 | 1.62746155 | 1.64684121 | 41630.39347603859 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 239.300861 | 1.5406974999999998 | 1.6102627 | 1.65887994 | 41315.12658335058 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 241.242786 | 1.589362 | 1.63805515 | 1.65918417 | 40233.497618019785 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 232.289351 | 1.320123 | 1.4710159999999999 | 1.48415427 | 94397.82396496143 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 241.265302 | 1.9518464999999998 | 2.0500727 | 2.0752199 | 65304.03738435739 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 238.646683 | 1.9968599999999999 | 2.05820705 | 2.11716406 | 64182.49007402749 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 243.178128 | 2.0576825000000003 | 2.1777382999999997 | 2.20243661 | 61801.40572125546 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 499.653736 | 0.47982899999999995 | 0.48814805 | 0.55381589 | 2069.6460739641834 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 506.161232 | 0.4792305 | 0.5411465999999998 | 0.60405751 | 2056.9017177391934 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 505.476205 | 0.4737135 | 0.4785932 | 0.48098109 | 2112.076226689648 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 544.898597 | 0.4851995 | 0.5155225999999999 | 0.57584814 | 2041.6775811347359 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 493.709337 | 0.48956299999999997 | 0.4946411 | 0.49705729 | 4089.142488419038 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 521.932291 | 0.48717350000000004 | 0.5223401999999999 | 0.5893316799999999 | 4059.5920507828987 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 514.487141 | 0.4918145 | 0.5048842 | 0.5826880799999999 | 4032.656777194262 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 544.035716 | 0.48102500000000004 | 0.48717145 | 0.48919063 | 4154.415813733942 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 499.621623 | 0.5114405 | 0.5944722 | 0.6207340499999999 | 7633.200882031628 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 512.969162 | 0.502391 | 0.5768023 | 0.60591588 | 7829.198824774621 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 512.087095 | 0.504622 | 0.57736125 | 0.59171193 | 7813.194641836127 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 540.178346 | 0.5068515 | 0.56461385 | 0.58674931 | 7768.010540258143 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 496.192301 | 0.5115689999999999 | 0.5312610999999999 | 0.6133191199999999 | 15506.444148676252 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 514.166447 | 0.5154654999999999 | 0.5654266999999998 | 0.63155712 | 15333.510563159594 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 519.014003 | 0.5084415 | 0.5499571999999999 | 0.6251546299999999 | 15536.98661232243 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 545.240629 | 0.515061 | 0.5311766499999999 | 0.5755006699999999 | 15441.156743375113 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.437161 | 0.545488 | 0.64670485 | 0.7281669399999998 | 28743.351797463707 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 522.930267 | 0.5378575 | 0.63480225 | 0.64670205 | 29280.177221200647 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 513.222956 | 0.5376005 | 0.6012392499999999 | 0.63654333 | 29450.26768268618 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 541.685565 | 0.5353465 | 0.54681465 | 0.6055808199999999 | 29706.84542265786 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 498.213836 | 0.5838885 | 0.6597729999999999 | 0.6938609 | 53742.294278696114 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 511.743552 | 0.5879464999999999 | 0.7171155499999999 | 0.7450253699999999 | 52802.50447558978 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 514.92946 | 0.6872805 | 0.72458815 | 0.7736485699999999 | 46066.73174936139 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.291058 | 0.587611 | 0.6561578 | 0.7092158199999999 | 53601.1030034976 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 496.062169 | 0.6656245000000001 | 0.7066635499999999 | 0.82100914 | 95105.92913247077 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 524.92089 | 0.8156654999999999 | 0.84574605 | 0.8882190199999997 | 77942.27901713715 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 507.213893 | 0.826343 | 0.9491448 | 1.00188552 | 76141.81794300975 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 535.847578 | 0.828569 | 0.9663021999999998 | 1.02625313 | 75223.1423997593 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.858803 | 0.8153535000000001 | 0.9701436 | 0.99346538 | 152304.40853677632 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 510.754055 | 1.278086 | 1.3185650500000001 | 1.43085618 | 99370.22412534834 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 512.420493 | 1.155736 | 1.24360205 | 1.2785537599999999 | 109842.43291789609 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 539.495418 | 1.0642325000000001 | 1.1718456499999998 | 1.2204694399999998 | 118507.28007812444 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 496.441065 | 0.554622 | 0.6003871999999999 | 0.61443246 | 1791.5299759153875 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 506.373222 | 0.54928 | 0.55426295 | 0.55761295 | 1822.4600929534836 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 513.625065 | 0.5450425 | 0.55373315 | 0.55497909 | 1833.7910475788092 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 508.965658 | 0.5469585 | 0.5538265499999999 | 0.55636447 | 1826.2014990557623 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 494.798273 | 0.8397954999999999 | 0.9155940499999999 | 0.92379075 | 2348.112903843133 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 507.842997 | 0.8536265000000001 | 0.921848 | 0.94774193 | 2322.7216928107187 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 507.031975 | 0.845107 | 0.91514025 | 0.9515462899999999 | 2342.517744161971 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 501.03948 | 0.8236954999999999 | 0.8968129499999999 | 0.91282506 | 2404.008482207368 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 500.12673 | 1.1069105000000001 | 1.1325781 | 1.22810245 | 3601.767675539801 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.833502 | 1.1558555 | 1.1814745999999998 | 1.2427915499999997 | 3450.226429735017 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 506.926796 | 1.1472755000000001 | 1.2254659 | 1.30678572 | 3452.688934289063 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 509.947871 | 1.1740024999999998 | 1.1856913999999998 | 1.18943739 | 3405.039434527825 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.972485 | 1.4815494999999999 | 1.6266893 | 1.63691769 | 5323.4021362653075 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 509.558566 | 1.5552715 | 1.5910617999999999 | 1.59956919 | 5143.623667700205 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 512.686836 | 1.5742895 | 1.69228005 | 1.7544053099999999 | 5032.358315587141 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.453949 | 1.722453 | 1.7762579 | 1.99520542 | 4620.14064147026 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 493.619719 | 1.4612215 | 1.605937 | 1.63305993 | 10816.813703897 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 509.832054 | 1.5710455 | 1.6172441 | 1.7532017099999995 | 10125.744661501736 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 497.026912 | 1.654509 | 1.7204419 | 1.76887569 | 9648.421995155455 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 508.286498 | 1.830473 | 2.1167448500000003 | 2.1720468199999994 | 8682.101489642417 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 499.015187 | 1.4671375 | 1.62090695 | 1.6790635699999998 | 21416.888611444996 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 514.725428 | 1.6452055 | 1.73818395 | 1.76884583 | 19355.342026615843 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 501.873536 | 1.6165889999999998 | 1.6798955500000001 | 1.76567021 | 19687.08946266255 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 510.362161 | 1.791912 | 1.84847355 | 1.8851832599999998 | 17857.0151077379 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 491.421516 | 1.5530685 | 1.5971366 | 1.6998893299999998 | 40951.52192108094 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 508.795176 | 1.8957095000000002 | 1.9816126 | 2.01670318 | 33685.303348472844 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 511.286685 | 1.9905475 | 2.0873671 | 2.10965655 | 32101.105642331087 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 510.252053 | 2.119878 | 2.2571203 | 2.32156168 | 30037.51037361848 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.037734 | 1.683436 | 1.72201985 | 1.72825942 | 76066.68038481705 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 506.408439 | 2.151231 | 2.26785375 | 2.2771326899999997 | 59298.147863118975 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 503.113627 | 2.369138 | 3.3243078 | 3.41177854 | 46851.23912413453 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 500.748283 | 2.3298945 | 2.38111215 | 2.40493809 | 55058.18988251734 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 12.810388 | 0.0559655 | 0.05849665 | 0.06461647 | 17795.17750689563 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 12.933518 | 0.057037 | 0.05953749999999999 | 0.0607949 | 17490.816446824596 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 13.438112 | 0.0573735 | 0.060462249999999995 | 0.06296583 | 17338.2703480207 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 14.029366 | 0.0594505 | 0.06364064999999999 | 0.06673264999999999 | 16706.05955509759 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 13.113687 | 0.0567425 | 0.06228024999999999 | 0.06518264 | 34928.3723867465 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 13.324157 | 0.0602885 | 0.06592015 | 0.06895583 | 32893.08136794431 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 13.154211 | 0.059719499999999995 | 0.063029 | 0.06835859999999999 | 33099.98944110337 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 14.01425 | 0.0595995 | 0.06312755 | 0.06792070999999998 | 33328.75618415071 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 12.748074 | 0.0625845 | 0.0649367 | 0.06711023999999999 | 63325.7078626782 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 13.081548 | 0.0667235 | 0.07213595 | 0.07559608 | 59029.80385281627 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 13.552336 | 0.06676199999999999 | 0.07332815 | 0.07820921999999998 | 59071.438339312845 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 14.361495 | 0.06656100000000001 | 0.0711045 | 0.07353594999999999 | 59465.335277453356 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 13.104109 | 0.0780545 | 0.07943035 | 0.08016809 | 102956.81682232021 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 12.994079 | 0.08053350000000001 | 0.08232455 | 0.08656723 | 99916.64453929309 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 13.215033 | 0.07867299999999999 | 0.0833763 | 0.08608711999999999 | 100677.94007900702 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 14.103498 | 0.0792645 | 0.0845393 | 0.09223471 | 99246.84053886072 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 12.996298 | 0.10178699999999999 | 0.10378915 | 0.10751698999999999 | 157323.76542633265 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 13.055409 | 0.1476655 | 0.16275734999999997 | 0.16637882 | 107489.9248349828 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 13.0972 | 0.15330149999999998 | 0.16156725 | 0.16502376 | 104436.201817216 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 13.990242 | 0.146614 | 0.15463745 | 0.15666822 | 108914.08707055006 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 12.900594 | 0.146382 | 0.15312935 | 0.15633455999999998 | 216986.43007551265 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 13.127351 | 0.275112 | 0.29354615 | 0.29920467 | 117231.94567588868 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 12.938392 | 0.30705499999999997 | 0.31476335 | 0.32322825 | 104105.45726662924 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 13.842571 | 0.308367 | 0.3160025 | 0.32304109999999997 | 103655.78401866219 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 12.939492 | 0.240654 | 0.2473841 | 0.25237753 | 264694.01248081896 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 13.079836 | 0.5428265 | 0.58477845 | 0.6237544399999999 | 117295.97730146896 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 13.252846 | 0.471537 | 0.5428992499999999 | 0.5858929999999999 | 134325.4611896803 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 13.817687 | 0.4955965 | 0.52841885 | 0.5850140199999998 | 128115.39097033915 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 13.03524 | 0.43023100000000003 | 0.43704405 | 0.4384558 | 297111.02700186794 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 13.089774 | 1.0665545 | 1.10233995 | 1.11231803 | 119747.19122672205 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 13.373985 | 1.0060125 | 1.06486905 | 1.06996805 | 127035.30658532171 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 14.1353 | 0.968207 | 1.0489740499999998 | 1.0748029499999998 | 130722.7322848485 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 302.661987 | 0.6763705 | 0.80051945 | 0.9128721899999996 | 1446.89967886351 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 295.751311 | 0.692068 | 0.7770118500000001 | 0.80568098 | 1417.68675371906 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 294.742158 | 0.701341 | 0.7681709 | 0.7965802 | 1421.6331277822428 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 296.349135 | 0.7168765 | 0.7724185 | 0.77980716 | 1395.8552924635094 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 286.55753 | 1.0379934999999998 | 1.1554882999999998 | 1.17470204 | 1924.8480423900812 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 301.810277 | 0.9585600000000001 | 1.0360978 | 1.08929032 | 2081.491473565662 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 301.228681 | 1.0244300000000002 | 1.1436518 | 1.19902744 | 1930.74282607886 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 297.208915 | 1.0101865 | 1.0829341499999998 | 1.13859189 | 1973.9084828914442 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 305.692872 | 1.1272479999999998 | 1.28280545 | 1.34676153 | 3466.411229134132 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 298.151215 | 1.19097 | 2.6027606999999997 | 13.008942819999962 | 2231.249660013333 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 299.509134 | 1.145518 | 1.2609539 | 1.27854855 | 3481.745764038468 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 301.213646 | 1.1249354999999999 | 1.2440528999999998 | 1.2807582499999999 | 3531.9684735787655 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 300.274675 | 1.2952910000000002 | 1.4537315999999998 | 1.48195341 | 6153.503356374562 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 305.872624 | 1.30167 | 4.926122699999988 | 17.893319109999965 | 3771.1737025800126 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 299.02536 | 1.2586515 | 1.4358089499999998 | 1.47595976 | 6304.695797324456 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 294.288577 | 1.3033394999999999 | 1.4658528 | 1.49668875 | 6169.223179899869 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 299.862049 | 2.5074265000000002 | 2.7875498499999996 | 2.83306349 | 6340.007149784562 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 295.828959 | 2.3979885 | 2.69255495 | 2.75022653 | 6595.775165260858 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 302.500453 | 2.655272 | 3.0097236499999998 | 3.08838534 | 6008.832096879678 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 297.056039 | 2.559654 | 2.7808224 | 2.84915626 | 6254.821783473926 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 299.699737 | 2.6540545 | 2.9197045999999998 | 3.2074578599999994 | 11951.64668518166 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 296.427655 | 2.6349774999999998 | 2.8178324 | 2.88267214 | 12214.435095736359 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 296.884098 | 2.7601605 | 3.20188135 | 3.28713131 | 11475.818685622271 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 293.057293 | 2.802339 | 3.11800385 | 3.23879277 | 11529.5887438931 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 301.223659 | 3.2669379999999997 | 3.5809520499999996 | 3.7860924599999994 | 19281.579553263517 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 297.489445 | 3.3560915 | 3.6138099 | 3.8175837799999996 | 18964.637245426267 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 297.628356 | 3.4063600000000003 | 3.8097404 | 3.9636848 | 18940.38579861237 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 294.834565 | 3.304318 | 3.51091495 | 3.57833267 | 19527.20701955251 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 296.924095 | 3.4344935000000003 | 3.7173339 | 3.74183077 | 37084.09843188209 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 298.119308 | 3.3312155 | 3.8314871999999998 | 3.91887022 | 38063.16059321459 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 300.053694 | 3.518526 | 3.7891322499999998 | 3.85526547 | 36436.4230873565 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 298.076421 | 3.511932 | 3.86657605 | 3.941266 | 36217.2364166123 | - |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
