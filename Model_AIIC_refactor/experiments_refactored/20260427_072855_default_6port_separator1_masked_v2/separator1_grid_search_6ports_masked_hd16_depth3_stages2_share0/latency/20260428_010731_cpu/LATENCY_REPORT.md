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

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`169289.579` samples/s, p50=`0.755` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`bf16`, p50=`0.505` ms, throughput=`1980.266` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`398341.728` samples/s, p50=`0.321` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.149` ms, throughput=`6707.498` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`223279.123` samples/s, p50=`0.562` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.466` ms, throughput=`2126.931` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`700706.093` samples/s, p50=`0.182` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.051` ms, throughput=`19435.689` samples/s

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`41058.950` samples/s, p50=`3.058` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.610` ms, throughput=`1455.501` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `21,024`
- MACs / sample: `19,968`
- FLOPs / sample estimate: `41,496`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.5286105 | 0.6006611499999999 | 0.63307705 | 1865.08407071608 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.5459620000000001 | 0.6401357 | 0.6474309 | 1782.4639626783419 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.533631 | 0.5492695 | 0.6108966699999999 | 1864.3875796289258 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.535839 | 0.6328367 | 0.63922791 | 1826.8547279639758 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.5662985 | 0.6547864 | 0.66211467 | 3467.3331620701906 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.5738175 | 0.5817045 | 0.58703644 | 3482.7143830113755 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.5839025 | 0.66744865 | 0.6719879999999999 | 3368.7309475104894 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.572274 | 0.6267799999999999 | 0.66377293 | 3460.040357218719 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.5884054999999999 | 0.6755742499999999 | 0.68039115 | 6691.208905731405 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.5729470000000001 | 0.6574064000000001 | 0.66540641 | 6857.4383147708195 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.572917 | 0.6654579 | 0.67252697 | 6823.277488332451 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.5867255 | 0.6557353499999999 | 0.68057369 | 6720.415655020096 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.578651 | 0.67289615 | 0.67958218 | 13566.970586027668 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.575319 | 0.6328208499999999 | 0.67107726 | 13755.431117047956 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.5887575 | 0.67500325 | 0.68110238 | 13327.266317463413 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.572065 | 0.65888715 | 0.6684127599999999 | 13736.085431308618 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.5803305000000001 | 0.6381900499999998 | 0.6830915399999999 | 27264.585385916398 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.5838760000000001 | 0.6258277999999999 | 0.68848776 | 27131.748242337522 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.5883575 | 0.60899105 | 0.68139905 | 27029.558308636202 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.59711 | 0.69013215 | 0.70919946 | 26356.367876384422 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.634681 | 0.6496779500000001 | 0.6525954599999999 | 50333.7901019923 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.6176254999999999 | 0.63787155 | 0.6866655 | 51540.0837378183 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.614088 | 0.6857953999999999 | 0.7515234699999999 | 51371.54974223682 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.6125864999999999 | 0.65457875 | 0.7099150999999999 | 51775.98589000832 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.6590625 | 0.67488055 | 0.67919743 | 96906.10307369201 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.652937 | 0.66813995 | 0.66879496 | 97786.98279242462 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.663685 | 0.68424275 | 0.70995771 | 95834.26996358119 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.666886 | 0.68462285 | 0.7134628599999999 | 95547.863687118 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.764135 | 0.78661185 | 0.7917219 | 166830.85168531354 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.7530645 | 0.8110683999999999 | 0.85371764 | 168386.44921626503 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.782558 | 0.8072306499999999 | 0.8214099499999999 | 163317.1843260007 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.7545010000000001 | 0.77605355 | 0.77895455 | 169289.57945400092 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.5052385 | 0.5109300999999999 | 0.51248797 | 1980.2664076161363 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.5107675 | 0.51711255 | 0.5182447699999999 | 1957.326830206282 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.5117835 | 0.5191928 | 0.52111548 | 1950.2062421109283 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.5077395 | 0.51433185 | 0.5171527 | 1966.7771221810326 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.540619 | 0.5492045 | 0.55070515 | 3696.950607261107 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.5455265 | 0.5500855499999999 | 0.5529763200000001 | 3666.5922226225734 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.5315995 | 0.53908745 | 0.5463445299999999 | 3756.2253331030015 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.5491275 | 0.5560551499999999 | 0.5876131399999999 | 3636.5560432379234 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.567159 | 0.57363855 | 0.57569713 | 7048.0470654486935 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.5624885 | 0.56730305 | 0.56848985 | 7115.900333419068 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.5680855 | 0.5736333 | 0.57519716 | 7040.141230865204 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.5627085000000001 | 0.57593185 | 0.7002757399999999 | 7033.024904714821 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.612124 | 0.617677 | 0.61828624 | 13080.796351033374 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.616419 | 0.6234552 | 0.62542849 | 12974.351620563011 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.608071 | 0.61553745 | 0.6499958099999998 | 13130.488297911872 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.6005465 | 0.6056629 | 0.60686263 | 13315.241914144586 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.03416 | 1.0446932 | 1.0719174 | 15459.842383041943 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.1329365 | 1.173089 | 1.1815208499999998 | 14059.022588631591 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.1697585 | 1.21210905 | 1.23373317 | 13582.868512009149 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.1915205 | 1.2348356999999999 | 1.24760305 | 13383.333467166667 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.5792175 | 1.60629105 | 1.61874327 | 20267.306005590384 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.8571035 | 1.9121252 | 1.9255335 | 17196.207948110958 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.870705 | 1.9122506499999998 | 1.9415618799999999 | 17098.435834021253 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 2.0058415 | 2.0373465 | 2.04469592 | 15960.848039758474 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.599593 | 1.6418371999999999 | 1.8265353699999995 | 39831.67482745074 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.9964 | 2.0354503 | 2.05256586 | 32049.193268223196 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.9720145 | 2.0025017 | 2.0233155 | 32451.424628696375 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.0644995 | 2.09821295 | 2.1661883399999997 | 30949.931946902765 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.823055 | 1.89733785 | 1.9233541699999999 | 69919.84018072531 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.052087 | 2.08147625 | 2.10220073 | 62360.24241452934 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.13891 | 2.1617317 | 2.17834448 | 59803.19646403135 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.3307135 | 2.6431226 | 2.7748146499999993 | 54001.737590284676 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 314.48596 | 0.15087 | 0.15669895 | 0.16641334999999996 | 6582.449058097616 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 316.210873 | 0.153717 | 0.1573492 | 0.15835628000000002 | 6486.918480192843 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 312.909214 | 0.1489175 | 0.1506689 | 0.15182943999999998 | 6707.498339223412 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 310.796569 | 0.1510015 | 0.15509175 | 0.15660167 | 6601.312789871896 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 313.538943 | 0.166676 | 0.1689182 | 0.17161669 | 11977.206896188278 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 312.22727 | 0.166425 | 0.16799895 | 0.16825191 | 12012.427577074137 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 312.073858 | 0.172991 | 0.1781241 | 0.17890102 | 11540.101564433868 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 311.740794 | 0.169555 | 0.17125975000000002 | 0.17190736 | 11783.901164178733 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 312.777187 | 0.172658 | 0.1746964 | 0.17673533 | 23129.558836046588 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 313.593363 | 0.1707785 | 0.17261565 | 0.17317346 | 23428.97313505081 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 318.902589 | 0.1762065 | 0.1811007 | 0.18508944 | 22624.70312147373 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 315.263727 | 0.180748 | 0.1825068 | 0.18589182999999998 | 22119.227724859367 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 313.495624 | 0.1807645 | 0.18256109999999998 | 0.18521121 | 44226.88747087659 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 319.280654 | 0.17980000000000002 | 0.19018189999999996 | 0.21761755999999993 | 44065.078392325275 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 312.904701 | 0.181163 | 0.1868583 | 0.18916143 | 43949.91994522082 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 315.596737 | 0.17985299999999999 | 0.1829734 | 0.18422829 | 44370.99073433879 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 317.688623 | 0.1911775 | 0.19620855 | 0.19764310999999998 | 83457.32832019072 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 313.0087 | 0.18582700000000002 | 0.1894687 | 0.19126119 | 85902.54420786526 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 312.822505 | 0.1876335 | 0.19298875000000001 | 0.19597409 | 85117.13128030104 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 311.117126 | 0.18726500000000001 | 0.20345865 | 0.20518607 | 84819.65644646353 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 313.439814 | 0.225071 | 0.2310849 | 0.23216507 | 141864.45897195616 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 316.946604 | 0.22221200000000002 | 0.22514275 | 0.22624866 | 143846.39506549327 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 313.512234 | 0.221743 | 0.22428775 | 0.22631582 | 144292.5965272379 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 312.765356 | 0.2237515 | 0.229152 | 0.22966923 | 142716.36335871543 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 314.458923 | 0.25646800000000003 | 0.25941285 | 0.2614124 | 249329.96467773558 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 313.611352 | 0.2664205 | 0.27096585 | 0.27358444 | 239939.41529763737 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 319.252481 | 0.2554795 | 0.26233075 | 0.2815107699999999 | 249004.21657515244 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 313.673078 | 0.2639395 | 0.26913395 | 0.27237474 | 242035.3722594788 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 316.655107 | 0.320579 | 0.325567 | 0.32900356 | 398341.72828152205 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 316.485354 | 0.3236165 | 0.3302944 | 0.33290475999999997 | 395252.6207719086 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 320.527133 | 0.32793249999999996 | 0.33382235 | 0.33852249 | 390064.4965550296 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 317.15922 | 0.3271405 | 0.33417485 | 0.33547136 | 389651.86919045803 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 311.521616 | 0.151893 | 0.154442 | 0.15605513 | 6567.413780036561 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 316.773472 | 0.154661 | 0.15850495 | 0.17013738999999997 | 6429.207736445687 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 312.870172 | 0.15141900000000003 | 0.1572674 | 0.16043314 | 6575.74169434506 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 315.166792 | 0.156049 | 0.15895320000000002 | 0.17202565999999994 | 6376.049720945808 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 316.722853 | 0.1671255 | 0.16916530000000002 | 0.17157785 | 11932.485045016492 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 311.654377 | 0.16756949999999998 | 0.16950384999999998 | 0.17678860999999996 | 11911.913307000857 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 311.740128 | 0.175006 | 0.1772854 | 0.17816939999999998 | 11416.39828617029 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 314.223907 | 0.172335 | 0.1757561 | 0.18694074999999996 | 11562.82307250341 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 313.657491 | 0.1953925 | 0.2007458 | 0.20291685999999998 | 20381.224693524975 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 313.933826 | 0.19627050000000001 | 0.203085 | 0.20858854000000002 | 20253.386062348833 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 316.076305 | 0.1963355 | 0.20274375 | 0.20443695 | 20276.58683808223 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 309.862646 | 0.19663049999999999 | 0.20199185 | 0.20263363 | 20292.073966638818 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 312.114025 | 0.25267300000000004 | 0.2561929 | 0.2568833 | 31610.79774273616 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 313.360697 | 0.25221550000000004 | 0.2552958 | 0.25832032 | 31658.741560274168 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 310.963697 | 0.253236 | 0.2557502 | 0.25819873 | 31570.803762671534 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 312.304132 | 0.25506949999999995 | 0.2605583 | 0.26118842999999997 | 31286.77418105891 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 319.024547 | 0.5624505 | 0.5686303 | 0.5821902899999999 | 28412.158130707292 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 326.361287 | 0.6308925000000001 | 0.71080055 | 1.0753166699999988 | 24190.904945467653 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 320.638503 | 0.620305 | 0.688596 | 0.72709505 | 25443.567249192896 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 327.949372 | 0.7739579999999999 | 0.8254526 | 0.8934654099999998 | 20584.60430613969 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 327.75026 | 0.9221075 | 0.9770987 | 0.98325397 | 34404.48766976491 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 325.754071 | 1.056843 | 1.06633395 | 1.06727178 | 30278.774023037382 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 333.113532 | 1.12211 | 1.135557 | 1.14228614 | 28482.782522900554 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 334.527856 | 1.257221 | 1.3312145 | 1.35970955 | 25263.62035195571 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 322.921919 | 1.020872 | 1.08241905 | 1.1095886899999998 | 61933.287436896506 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 325.533516 | 1.303746 | 1.3505635 | 1.39065932 | 49098.29523200778 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 327.501357 | 1.2925015000000002 | 1.3845074499999999 | 1.39026325 | 49200.29229893655 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 329.757526 | 1.3804405000000002 | 1.4131443 | 1.41508433 | 46412.97200795948 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 330.064599 | 1.1436975 | 1.1763842 | 1.18555406 | 111467.64890101083 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 328.908147 | 1.4868115 | 1.5160566 | 1.53964999 | 85990.18368309393 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 330.837199 | 1.5406005 | 1.5712392499999999 | 1.58350531 | 82861.89554173095 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 331.739505 | 1.52349 | 1.5497659 | 1.55728016 | 83960.0106340601 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 492.284937 | 0.4684405 | 0.5209692499999999 | 0.5523910399999999 | 2103.0503946815707 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 505.010802 | 0.4664975 | 0.50643595 | 0.5317213199999999 | 2126.931077946962 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 513.126224 | 0.4745525 | 0.5247276 | 0.5608324499999999 | 2083.1906347748513 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 544.80283 | 0.469584 | 0.5304812999999999 | 0.55969154 | 2101.594702669954 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.877962 | 0.4860135 | 0.5052841499999999 | 0.56475327 | 4081.6964608058975 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 513.287573 | 0.48324049999999996 | 0.5219467999999998 | 0.56763907 | 4097.131684147694 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 514.654219 | 0.47663449999999996 | 0.5174611999999998 | 0.5590972499999999 | 4155.386523665965 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 547.606464 | 0.4818825 | 0.4993122 | 0.53130888 | 4127.409158407239 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 496.797097 | 0.485898 | 0.5114936499999999 | 0.56674847 | 8174.8181593448635 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 501.069781 | 0.485893 | 0.50182045 | 0.5399832999999999 | 8196.297388225248 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 504.948825 | 0.48386399999999996 | 0.4886106 | 0.49440469 | 8263.13183107944 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 534.459079 | 0.4875935 | 0.5299110999999999 | 0.5735385199999999 | 8118.938552706646 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.629824 | 0.48108949999999995 | 0.50096875 | 0.56580798 | 16463.00353588274 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 509.838236 | 0.4885695 | 0.54600785 | 0.6263577099999997 | 16090.214615305093 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 512.581043 | 0.4831305 | 0.49268755 | 0.5412563499999999 | 16473.547271997046 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 550.326297 | 0.48444750000000003 | 0.5256854999999998 | 0.5684636599999999 | 16373.14059963762 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 492.256753 | 0.4894925 | 0.51022295 | 0.5719291 | 32412.026871514998 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 502.281464 | 0.4806495 | 0.5195297999999998 | 0.5741619299999999 | 32911.704233045806 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 512.910946 | 0.4899345 | 0.5182910499999999 | 0.57612751 | 32366.92833035564 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 534.568972 | 0.487852 | 0.5410997999999999 | 0.5832337400000001 | 32406.965374899366 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 497.378999 | 0.499393 | 0.5817007999999999 | 0.59616251 | 62509.542472343055 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 514.67626 | 0.502712 | 0.58440055 | 0.59601667 | 62404.102445694785 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 503.519647 | 0.49393149999999997 | 0.5323183499999999 | 0.5620882599999999 | 64123.21403823876 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.843 | 0.4973975 | 0.5498666499999999 | 0.57771645 | 63523.522760478205 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 496.174075 | 0.5297765000000001 | 0.5399527 | 0.6010869399999998 | 120194.86291663202 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 511.169455 | 0.5345409999999999 | 0.5452232499999999 | 0.5679439299999999 | 119545.58332865112 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 508.046857 | 0.5228625 | 0.6095187 | 0.63157767 | 119766.67356073583 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 542.313344 | 0.5191429999999999 | 0.5760285499999999 | 0.6155322999999999 | 121870.4385465838 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 491.518584 | 0.5643735000000001 | 0.6658908499999999 | 0.6900976600000001 | 221608.67398510844 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 506.931665 | 0.5714965 | 0.5840205 | 0.6611743199999999 | 222689.8864483392 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 510.378449 | 0.57245 | 0.591139 | 0.6499432499999998 | 222388.98307216758 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 542.084356 | 0.5623855 | 0.6555704499999998 | 0.68854248 | 223279.12325544868 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 488.908034 | 0.470372 | 0.51548225 | 0.5291553 | 2106.0788089635726 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 512.742136 | 0.475387 | 0.48163039999999996 | 0.48370678 | 2104.382329905617 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 508.0282 | 0.4775625 | 0.51179115 | 0.52720407 | 2079.640844362466 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 523.329222 | 0.4824075 | 0.51730435 | 0.52783585 | 2056.5207584382747 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 494.062724 | 0.48515050000000004 | 0.52764765 | 0.55413777 | 4057.3541085011148 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 505.369091 | 0.487891 | 0.5296764 | 0.5463921399999999 | 4061.06196526728 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 506.496039 | 0.490934 | 0.49764359999999996 | 0.49950189 | 4072.2221631750012 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 510.797967 | 0.48334049999999995 | 0.52343925 | 0.53876952 | 4103.796841044928 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 490.199031 | 0.5125725 | 0.5181738 | 0.5232397400000001 | 7804.641092262643 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 499.959658 | 0.515201 | 0.51984305 | 0.5237019799999999 | 7769.025323332266 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 499.066485 | 0.5111355 | 0.5448712 | 0.56945785 | 7775.514973328623 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 505.597583 | 0.5071589999999999 | 0.5545062 | 0.6138868199999998 | 7754.785313227604 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.230202 | 0.5627675000000001 | 0.57111075 | 0.57313819 | 14193.897007521984 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 508.52919 | 0.556095 | 0.56172415 | 0.56373578 | 14387.32166195434 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 505.461705 | 0.562664 | 0.57058085 | 0.57231481 | 14218.128412306385 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 506.458334 | 0.553142 | 0.55849755 | 0.5756741299999999 | 14439.210832151575 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 496.938568 | 0.9058470000000001 | 0.9449995 | 0.9688494399999998 | 17602.495646627787 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 515.889434 | 0.972585 | 1.016822 | 1.0457538899999999 | 16329.964903843555 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 512.311876 | 0.967441 | 0.9970677 | 1.0218880300000002 | 16496.193627042012 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 511.29394 | 1.0027995 | 1.03522785 | 1.05026337 | 15928.03554261491 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 492.632393 | 1.424276 | 1.4857961499999999 | 1.51032951 | 22309.28389276256 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 513.291413 | 1.6160765000000001 | 1.6352065 | 1.66475966 | 19774.775197883704 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 508.526851 | 1.654989 | 1.7423065999999998 | 1.8612616899999999 | 19187.585401993885 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 508.60491 | 1.8075320000000001 | 1.87940375 | 1.9278642499999998 | 17623.70022870166 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.258326 | 1.514938 | 1.5554044 | 1.6194141999999998 | 42020.2776991466 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 508.494873 | 1.7268755 | 1.9146416499999999 | 1.94885986 | 36527.69344224598 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 512.097445 | 1.8018105000000002 | 1.9530210000000001 | 1.97899794 | 35072.21445677119 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 514.429855 | 1.8822695 | 1.9697077 | 2.04237601 | 33759.36526991794 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 496.907865 | 1.5388805 | 1.5849307 | 1.62600208 | 82819.36262347929 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 512.339159 | 1.872377 | 1.92699195 | 1.93171194 | 68296.6523574153 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 508.11032 | 1.9729325 | 2.02836105 | 2.04898798 | 64899.93165732977 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 516.160193 | 2.0136269999999996 | 2.0873653500000002 | 2.11500484 | 63291.868423919244 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 13.005707 | 0.051275 | 0.054329300000000004 | 0.05610704 | 19435.688666700094 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 13.447225 | 0.054837 | 0.059011999999999995 | 0.06822223999999998 | 17947.81135414442 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 14.121223 | 0.053315 | 0.058694649999999994 | 0.06021944 | 18599.100845068744 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 14.385231 | 0.05519 | 0.05798999999999999 | 0.06596490999999997 | 17947.772699355784 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 13.382379 | 0.053702 | 0.0552326 | 0.05825346999999999 | 37154.831614303126 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 13.296832 | 0.0549575 | 0.05910745 | 0.06105139 | 35986.26476246546 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 13.993943 | 0.054709 | 0.058802799999999995 | 0.0601798 | 36250.744952808775 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 14.400322 | 0.0550205 | 0.06097185 | 0.06412619 | 35827.96272172134 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 13.276309 | 0.054182 | 0.0570206 | 0.058329139999999995 | 73347.44538182479 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 13.449085 | 0.0569295 | 0.06169469999999999 | 0.06807635999999999 | 69332.26099436329 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 14.233468 | 0.056898500000000005 | 0.059751849999999995 | 0.06354673 | 69905.50523330088 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 15.468385 | 0.057258 | 0.06022684999999999 | 0.06332629000000001 | 69341.51563217792 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 13.268547 | 0.0595185 | 0.0616846 | 0.06515766999999999 | 133627.625908125 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 13.578437 | 0.0633525 | 0.0685963 | 0.07127014 | 125469.64857833477 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 13.674978 | 0.06320100000000001 | 0.0677487 | 0.0693148 | 125699.2018100685 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 14.702229 | 0.063291 | 0.0684953 | 0.07049743 | 126006.99275806312 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 13.337187 | 0.06762699999999999 | 0.06928134999999999 | 0.07086166999999999 | 236855.83355193154 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 13.49065 | 0.07159750000000001 | 0.07732699999999999 | 0.07963149 | 221747.3133927062 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 14.017154 | 0.0710195 | 0.07503789999999999 | 0.08169185999999998 | 223908.69003620325 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 14.36057 | 0.070488 | 0.0740919 | 0.07728690999999999 | 226656.82598280525 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 13.469339 | 0.086173 | 0.08876125 | 0.09122322 | 370244.3844340004 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 13.57353 | 0.089452 | 0.0954348 | 0.09823643 | 355586.53109337523 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 14.178127 | 0.08808350000000001 | 0.0954212 | 0.09780395 | 359237.58110069047 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 14.603239 | 0.088806 | 0.09569995 | 0.09862322999999999 | 355873.46918803017 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 13.122395 | 0.1199385 | 0.1230961 | 0.13084137999999998 | 530996.1537625475 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 13.400851 | 0.1446295 | 0.15310925 | 0.22950158999999973 | 432498.41766399227 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 13.646839 | 0.151247 | 0.15732854999999998 | 0.16141292 | 424233.54586992075 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 14.465318 | 0.14948899999999998 | 0.15588915 | 0.19286922999999986 | 425066.26384553727 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 13.30202 | 0.1819875 | 0.1880654 | 0.19292582 | 700706.0927708589 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 13.664493 | 0.203913 | 0.21623525000000002 | 0.21867617999999997 | 624962.404605348 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 13.807372 | 0.230804 | 0.23852695000000002 | 0.2393825 | 554209.8167358029 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 14.58322 | 0.2417925 | 0.2488438 | 0.25112234 | 528914.9104005746 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 308.586923 | 0.6362805 | 0.7326125499999999 | 0.9789427499999992 | 1531.3837185120706 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 299.018129 | 0.645792 | 1.1217356999999992 | 2.2787802199999994 | 1378.2887382039503 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 301.44546 | 0.651714 | 1.0752643 | 1.6870370099999996 | 1404.6180752153314 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 303.090793 | 0.6095550000000001 | 1.4570745999999986 | 2.025149699999999 | 1455.500940632038 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 299.232947 | 0.7812975 | 1.07584145 | 1.588165909999998 | 2544.747540558759 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 297.418456 | 0.98115 | 1.06294925 | 1.06920891 | 2050.8734752165888 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 296.776861 | 0.948374 | 1.0973812 | 1.2399384699999998 | 2085.1226726668715 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 301.163257 | 0.950413 | 1.0794795 | 1.10099536 | 2060.777140079451 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 294.433491 | 1.1009315 | 1.2204126999999998 | 1.2921486299999998 | 3623.174687617578 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 306.812213 | 1.0519075 | 1.1908716499999998 | 1.3097342299999997 | 3718.4855099358115 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 294.156931 | 1.094299 | 1.23263705 | 1.28215386 | 3682.154847093825 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 297.959109 | 1.0779655 | 1.172296 | 1.21516761 | 3705.1769232170072 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 299.199077 | 1.2230634999999999 | 1.3422414999999999 | 1.38127008 | 6582.446783345034 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 298.759967 | 1.2432555 | 1.4727146999999998 | 2.3139174199999997 | 6189.240228675097 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 300.350172 | 1.237141 | 1.4460241499999997 | 3.7990872999999947 | 6247.200765994275 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 299.625913 | 1.2153429999999998 | 1.3622655499999998 | 1.4354702099999999 | 6564.758635981016 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 300.100713 | 1.5117175 | 1.7599151 | 1.80319399 | 10545.54837207447 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 298.500282 | 1.5138539999999998 | 4.3679960999999885 | 16.203656489999965 | 7242.822055523039 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 298.826848 | 1.5211845 | 6.019433049999999 | 15.35366361999997 | 6900.922268580893 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 296.950786 | 1.5524365 | 1.7070865 | 1.7824513499999999 | 10289.390655597272 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 302.967424 | 1.3560915 | 1.52931985 | 1.5881469199999998 | 23658.16716452815 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 297.874041 | 1.3477375 | 1.4344808 | 1.5218209799999998 | 24208.3645406891 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 299.864927 | 1.3823555 | 1.5575901 | 1.66504367 | 23307.01689499434 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 297.954696 | 1.3818665 | 1.53243915 | 1.5804304599999999 | 22959.40601950281 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 297.331521 | 2.8766555 | 3.06808525 | 3.10694399 | 22457.912555976785 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 302.790947 | 3.176481 | 3.4854531499999997 | 3.54591106 | 20159.377137740514 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 297.792609 | 3.03538 | 3.3012495 | 3.47347821 | 20965.754870136836 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 300.211453 | 2.957343 | 3.2418035499999998 | 3.3822509 | 21494.256388769732 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 303.324125 | 3.3239885 | 9.165177449999979 | 19.56528988999998 | 30533.29894075263 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 304.525709 | 3.058287 | 3.4346961 | 3.5340795000000003 | 41058.94983549668 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 300.138176 | 3.266914 | 3.57469885 | 3.7020710799999996 | 39241.24887040386 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 304.840781 | 3.158142 | 3.4145453999999997 | 3.5320599799999997 | 40708.904351516 | - |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd16_depth3_stages2_share0` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
