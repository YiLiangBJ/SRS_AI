# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 128]`

## Hardware Summary

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

### full_mlp_capacity_search_hd512_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`362669.972` samples/s, p50=`0.351` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.078` ms, throughput=`12719.106` samples/s

### full_mlp_capacity_search_hd512_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`383657.647` samples/s, p50=`0.333` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.051` ms, throughput=`18903.063` samples/s

### full_mlp_capacity_search_hd512_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`415881.744` samples/s, p50=`0.304` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.079` ms, throughput=`12492.483` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `611,984`
- MACs / sample: `610,304`
- FLOPs / sample estimate: `1,222,360`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.150867 | 0.18247249999999998 | 0.18757861 | 6493.6954010090685 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.14078649999999998 | 0.15222755 | 0.15551503 | 7713.963755478456 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0865495 | 0.0943533 | 0.09858067999999999 | 11952.90936204454 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.078203 | 0.08525724999999999 | 0.09097659999999998 | 12719.105673891288 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.16412700000000002 | 0.21747364999999996 | 0.23876277999999995 | 5886.987038856116 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.160074 | 0.18267909999999998 | 0.19860092 | 12367.223937899716 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.1231075 | 0.13211355 | 0.13548323 | 16077.134231530668 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.09453 | 0.10299169999999999 | 0.10526334 | 20888.52255062229 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0735315 | 0.08595725 | 0.10762392999999991 | 26314.099831484506 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.255666 | 0.27559120000000004 | 0.28292533999999997 | 7728.321671295933 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1963095 | 0.2381365 | 0.24939210999999997 | 19955.885519470656 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.1749315 | 0.183494 | 0.19695268999999999 | 24885.51728840436 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.1164405 | 0.12409885 | 0.12709825 | 35159.06135150729 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0948905 | 0.10506889999999999 | 0.10690978999999999 | 41673.45596720132 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.2392625 | 0.2531848 | 0.25616816 | 16610.964565490387 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.248813 | 0.273875 | 0.28331029999999996 | 31716.145571716424 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1564665 | 0.16879095 | 0.17375396999999998 | 50515.96373913606 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.117028 | 0.15921185 | 0.16808367 | 62394.679730448755 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.107073 | 0.1170186 | 0.11996042999999999 | 73540.7943655985 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.3049225 | 0.33602339999999997 | 0.3900385799999998 | 25809.417198195486 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.30774500000000005 | 0.33106654999999996 | 0.34895886 | 51355.44347800633 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.26783 | 0.2914844 | 0.31382228999999995 | 58718.32861800689 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1863765 | 0.23349674999999998 | 0.2788518499999999 | 79838.67796729929 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1471595 | 0.21670145 | 0.21882664 | 94899.11334572162 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.40972 | 0.4383912 | 0.5052451999999997 | 38593.11128400825 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.43517150000000004 | 0.47391714999999995 | 0.54947217 | 72353.17720211078 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.338952 | 0.35446175 | 0.36788204999999996 | 93695.7461369976 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.23859049999999998 | 0.34185869999999996 | 0.36864481000000004 | 127139.02469588145 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.2152165 | 0.23653435 | 0.24178681999999999 | 147209.59158814952 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.475789 | 0.48648895 | 0.49083996999999996 | 67345.75601567018 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.7189255 | 0.73582125 | 0.74023572 | 89495.49932727353 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.5081964999999999 | 0.5147613 | 0.5226942 | 126025.5724790396 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.365993 | 0.3753052 | 0.3964787799999999 | 174168.910860297 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.297285 | 0.33370595 | 0.33850256 | 210291.96476374124 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.8471525 | 0.8731885 | 0.87966293 | 75537.62963673577 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.1259105 | 1.15346155 | 1.2545664199999995 | 113024.98915401609 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.7763070000000001 | 0.790681 | 0.81395259 | 164806.5543154635 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.5326850000000001 | 0.5410376 | 0.54521163 | 240034.16586308356 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.447413 | 0.4615833 | 0.5404853799999998 | 283806.7873546161 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 1.3353365 | 1.36375385 | 1.37296941 | 95940.32847360092 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.177153 | 0.19102914999999998 | 0.19869944 | 5581.181239149487 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1547175 | 0.18061565 | 0.18585958 | 6281.797674604137 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.13939600000000002 | 0.14752655 | 0.15097698999999998 | 7215.863122581062 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.13560299999999997 | 0.1591556 | 0.16345142 | 7044.732076158625 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.6060525 | 0.6354162999999999 | 0.64467061 | 1640.5810990751518 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1884255 | 0.20863964999999998 | 0.22108588999999995 | 10437.598400458919 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.15594049999999998 | 0.17075479999999998 | 0.20425076999999986 | 12610.41999003777 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.147073 | 0.172284 | 0.1795821 | 13358.937964431829 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.146422 | 0.15987355 | 0.17179595999999997 | 13521.959256173248 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.42418049999999996 | 0.46218899999999996 | 0.46886197 | 4656.207954442171 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1861665 | 0.1974549 | 0.21345118999999999 | 21266.494159954877 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.16135149999999998 | 0.21323545 | 0.21784234 | 23566.599859283833 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.149455 | 0.159565 | 0.16769587 | 26581.13919319863 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1487845 | 0.16034325 | 0.16998751999999995 | 26778.09926373616 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.352543 | 0.37388435 | 0.38437479999999996 | 11288.36350336619 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1998335 | 0.21197935 | 0.21801748999999998 | 39739.211425023284 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.183642 | 0.19448065 | 0.19671844 | 44611.08668034517 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1655325 | 0.17497155 | 0.18466685000000002 | 48362.225329763874 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.15673900000000002 | 0.16400185 | 0.16466772999999998 | 50909.26492621466 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.477864 | 0.51074195 | 0.51960619 | 16690.046640335335 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.2004485 | 0.21259820000000001 | 0.21668642999999999 | 79106.2966140033 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.19032500000000002 | 0.21360764999999998 | 0.21877703 | 83256.8584398206 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.188051 | 0.22750979999999998 | 0.23308157999999998 | 82447.07876602282 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1805155 | 0.19508655 | 0.19884191 | 88362.79643509134 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.623336 | 0.6728736 | 0.67833847 | 25666.098622433114 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.231405 | 0.25072289999999997 | 0.25979979999999997 | 136489.17926582642 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.2254725 | 0.2396022 | 0.24442571999999999 | 140963.77283899454 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.22001500000000002 | 0.27473264999999997 | 0.28375220999999995 | 141780.5600243508 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.2224685 | 0.2360302 | 0.23767803 | 143687.16717337348 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.4885815 | 0.5139497000000001 | 0.5251409 | 65266.38188224819 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.289584 | 0.30778925 | 0.3336149099999999 | 219847.03455548815 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.308913 | 0.3310688 | 0.3402398 | 206247.62734663073 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.3024285 | 0.31657955 | 0.31882064 | 210772.4203234145 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2908615 | 0.3084236 | 0.32699711 | 219196.98417829315 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.7717959999999999 | 0.81060235 | 0.83588179 | 82841.5222005442 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.3859175 | 0.40238155 | 0.40664205 | 331697.2367132452 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.402708 | 0.4218174 | 0.42658768999999996 | 317801.4180100646 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.396316 | 0.41851694999999994 | 0.44269499999999995 | 323341.9002096165 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.3509735 | 0.37601619999999997 | 0.38012567 | 362669.97179900965 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 1.330969 | 1.41706675 | 1.5641550499999994 | 95615.09045762742 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 1 | ok | 54.383037 | 0.121216 | 0.1442129 | 0.14812764999999997 | 8008.219636635043 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 2 | ok | 54.236741 | 0.0666065 | 0.0760261 | 0.10669388 | 14464.842765712652 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 4 | ok | 53.574326 | 0.051348 | 0.05988935 | 0.06111307 | 18903.062825463483 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 8 | ok | 53.693233 | 0.069102 | 0.07646485 | 0.07888495999999999 | 14261.49012605731 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 128 | ok | 68.447453 | 0.13689099999999998 | 0.16920525 | 0.18642371 | 7119.664327762007 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 1 | ok | 54.113577 | 0.12795250000000002 | 0.14893725 | 0.15529032999999998 | 15328.275717666038 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 2 | ok | 55.610706 | 0.0945455 | 0.1074363 | 0.11382003999999998 | 20964.874819780696 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 4 | ok | 54.199925 | 0.0751735 | 0.0838467 | 0.08787407 | 26138.84326208582 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 8 | ok | 54.796595 | 0.052026 | 0.05863 | 0.06055078 | 37747.857148519324 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 128 | ok | 86.220141 | 0.174118 | 0.2058928 | 0.21309292999999999 | 11364.257523621462 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 1 | ok | 57.3353 | 0.136467 | 0.1592352 | 0.16306880999999998 | 28884.207400740506 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 2 | ok | 54.323997 | 0.1004785 | 0.1151355 | 0.11784363999999999 | 39819.9263291543 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 4 | ok | 53.866761 | 0.07584850000000001 | 0.08556504999999999 | 0.08875672 | 52398.44722441495 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 8 | ok | 55.674005 | 0.1019695 | 0.11381045 | 0.11524567 | 38784.802330346065 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 128 | ok | 135.695384 | 0.1938585 | 0.2168706 | 0.22438444 | 20237.443905600016 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 1 | ok | 54.847798 | 0.1540215 | 0.1806904 | 0.18680554 | 50972.471934238376 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 2 | ok | 55.929854 | 0.113814 | 0.129583 | 0.13441661 | 69439.44142913313 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 4 | ok | 54.798547 | 0.0926895 | 0.09726844999999999 | 0.10039862 | 85839.24926709376 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 8 | ok | 54.837124 | 0.070814 | 0.0766366 | 0.07829857999999999 | 111448.24204329208 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 128 | ok | 69.814351 | 0.3053535 | 0.33059204999999997 | 0.341192 | 26059.83050425642 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 1 | ok | 55.579556 | 0.1918355 | 0.2163012 | 0.21984947999999999 | 82133.90454726148 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 2 | ok | 56.866289 | 0.1466805 | 0.1695159 | 0.17466014 | 107748.01134209443 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 4 | ok | 56.001763 | 0.11841399999999999 | 0.13289915 | 0.13694215 | 136638.3648760152 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 8 | ok | 55.993424 | 0.1005105 | 0.10676524999999999 | 0.10957083999999999 | 158104.2666017385 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 128 | ok | 78.155379 | 0.4982815 | 0.5304252 | 0.53355729 | 32052.012724168275 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 1 | ok | 58.050416 | 0.29530049999999997 | 0.3073907 | 0.31401847 | 107754.39701919012 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 2 | ok | 57.249253 | 0.245635 | 0.2553774 | 0.25998499999999997 | 130418.99057005488 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 4 | ok | 56.452074 | 0.2112385 | 0.2185402 | 0.22021109 | 151355.08204958998 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 8 | ok | 54.601931 | 0.185926 | 0.1988258 | 0.20825535999999997 | 171253.49531059765 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 128 | ok | 124.691194 | 35.144528 | 56.90673854999998 | 63.29610136 | 879.161209433723 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 1 | ok | 58.355339 | 0.49439049999999995 | 0.50581365 | 0.52595319 | 129332.31947572557 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 2 | ok | 57.513592 | 0.410571 | 0.421474 | 0.42506957 | 155954.7064644883 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 4 | ok | 57.981431 | 0.32974899999999996 | 0.339812 | 0.3568160799999999 | 193711.13178573878 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 8 | ok | 57.445179 | 0.285097 | 0.2962116 | 0.30540864999999995 | 223776.3646144952 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 128 | ok | 108.228489 | 26.0306275 | 41.49883204999997 | 53.65639232999999 | 2361.8543999146887 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 1 | ok | 60.276428 | 0.905363 | 0.9155601 | 0.92872165 | 141192.39314510048 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 2 | ok | 60.101569 | 0.7209954999999999 | 0.7402793 | 0.75018692 | 176794.49590115773 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 4 | ok | 59.619234 | 0.49726400000000004 | 0.5083386999999999 | 0.51164651 | 257489.67043977545 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 8 | ok | 56.862111 | 0.3534845 | 0.36748095 | 0.37329517999999995 | 361846.17082508956 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 128 | ok | 82.809551 | 11.832097000000001 | 13.716917749999999 | 15.04959164 | 10727.47764434306 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 1 | ok | 57.943772 | 0.136106 | 0.14904789999999998 | 0.15561911999999997 | 7231.453815814004 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 2 | ok | 56.619664 | 0.116171 | 0.13160354999999999 | 0.13766101 | 8477.3105630511 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 4 | ok | 55.706998 | 0.1033815 | 0.11843269999999999 | 0.12200243999999999 | 9527.72042619399 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 8 | ok | 53.760784 | 0.0942225 | 0.10923255 | 0.11554452 | 10256.553322179674 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 128 | ok | 77.433957 | 0.352202 | 0.38465155 | 0.39083994 | 2829.6623086660165 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 1 | ok | 56.70377 | 0.1452505 | 0.16391994999999998 | 0.17442175 | 13494.036512973638 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 2 | ok | 56.680308 | 0.11962400000000001 | 0.1320905 | 0.13738794000000001 | 16543.14999442496 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 4 | ok | 54.228387 | 0.10384399999999999 | 0.11464404999999998 | 0.11780372 | 19175.845184964448 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 8 | ok | 55.54276 | 0.105299 | 0.12425444999999999 | 0.13013247 | 18361.200433250884 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 128 | ok | 78.159948 | 0.45081550000000004 | 0.48798444999999996 | 0.49274751 | 4424.8263377345165 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 1 | ok | 55.922477 | 0.14928950000000002 | 0.16519489999999998 | 0.17437302999999996 | 26413.554379565416 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 2 | ok | 56.606144 | 0.1278865 | 0.1371286 | 0.15030723 | 30908.189232915924 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 4 | ok | 56.669635 | 0.115952 | 0.13612295 | 0.13833169 | 33830.41489620829 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 8 | ok | 53.487736 | 0.10553499999999999 | 0.12119545 | 0.12700783999999998 | 37126.20179835609 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 128 | ok | 89.029733 | 0.4065615 | 0.4430548 | 0.44669053000000003 | 9753.38897349141 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 1 | ok | 57.679989 | 0.1506405 | 0.1615095 | 0.17230378999999996 | 52348.18241875824 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 2 | ok | 54.962504 | 0.1380555 | 0.1472866 | 0.15391008999999997 | 57460.952768676994 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 4 | ok | 56.129377 | 0.125191 | 0.14467554999999999 | 0.15076971 | 62611.89920360794 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 8 | ok | 54.233819 | 0.1233655 | 0.1395592 | 0.15200931999999998 | 64138.26156769626 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 128 | ok | 82.704715 | 0.43756249999999997 | 0.47963369999999994 | 0.5529200499999998 | 18184.02919445887 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 1 | ok | 57.147272 | 0.1772555 | 0.1892731 | 0.19288998 | 89316.20739714598 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 2 | ok | 55.323931 | 0.1600955 | 0.17109925 | 0.1759069 | 99402.7756982579 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 4 | ok | 55.335721 | 0.142631 | 0.15821995 | 0.16223493 | 110645.30972457476 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 8 | ok | 56.258566 | 0.1322305 | 0.14574065 | 0.15333971 | 119662.59933491527 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 128 | ok | 78.305335 | 0.3481615 | 0.3840612 | 0.38610638999999997 | 45490.5450176708 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 1 | ok | 57.827671 | 0.1851125 | 0.2049917 | 0.21215989999999998 | 170325.56134248478 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 2 | ok | 57.656794 | 0.1918825 | 0.2037772 | 0.20640719 | 165164.30803432484 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 4 | ok | 56.876474 | 0.1823235 | 0.188268 | 0.19200368999999998 | 175833.31529210255 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 8 | ok | 56.657473 | 0.1825255 | 0.1931833 | 0.19682023999999998 | 175314.6075418153 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 128 | ok | 83.090936 | 0.714832 | 0.75182995 | 0.75832505 | 44747.38048135484 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 1 | ok | 57.058077 | 0.24725 | 0.26285885 | 0.26635255999999996 | 257395.41260243332 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 2 | ok | 57.927666 | 0.261985 | 0.2809017 | 0.28356678999999996 | 242878.38821044014 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 4 | ok | 58.529887 | 0.2466485 | 0.26591645 | 0.27024072 | 258207.19642821985 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 8 | ok | 57.489398 | 0.264773 | 0.276783 | 0.28044163 | 242441.48883015235 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 128 | ok | 83.317348 | 0.636049 | 0.6749552 | 0.67824063 | 100247.7309396013 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 1 | ok | 57.751463 | 0.3535985 | 0.36262835 | 0.37184246 | 360285.7065653063 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 2 | ok | 59.635577 | 0.365495 | 0.38471354999999996 | 0.39814322999999996 | 348552.62975879177 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 4 | ok | 62.207378 | 0.8360620000000001 | 0.86867915 | 0.87677978 | 153385.3451854509 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 8 | ok | 57.021616 | 0.333418 | 0.3442691 | 0.34514842 | 383657.64693533373 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 128 | ok | 155.60933 | 0.8542735 | 0.88916615 | 0.90387839 | 149535.34811186505 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 1 | ok | 499.313257 | 0.1427295 | 0.16837719999999998 | 0.17232392 | 6888.870396099026 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 2 | ok | 503.726279 | 0.1421305 | 0.1492321 | 0.15127691 | 7364.103207611891 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 4 | ok | 505.403311 | 0.09836349999999999 | 0.1029058 | 0.10530033 | 10124.576843310833 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 8 | ok | 508.401849 | 0.0794555 | 0.0858253 | 0.08821945 | 12492.482648566225 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 128 | ok | 671.446797 | 0.171309 | 0.1952696 | 0.1988578 | 5773.687389700919 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 1 | ok | 497.400715 | 0.15234350000000002 | 0.17624175 | 0.17876194999999998 | 12885.033801309171 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 2 | ok | 514.635793 | 0.1279805 | 0.13539845 | 0.13786193 | 15495.738904237416 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 4 | ok | 495.855353 | 0.09747449999999999 | 0.10565559999999999 | 0.12512240999999993 | 20168.060447710774 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 8 | ok | 511.822749 | 0.0987355 | 0.10560139999999998 | 0.10976671 | 20073.025667377922 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 128 | ok | 658.971646 | 0.27074 | 0.2984101 | 0.30663548 | 7300.064554470855 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 1 | ok | 508.048956 | 0.192489 | 0.20494949999999998 | 0.20939025 | 20660.43988555769 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 2 | ok | 512.362137 | 0.14757900000000002 | 0.2056367 | 0.20742858 | 25355.744261456264 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 4 | ok | 508.178717 | 0.13816699999999998 | 0.14503335 | 0.14731381999999998 | 28734.16657627377 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 8 | ok | 511.757431 | 0.115998 | 0.122929 | 0.12710037 | 34215.740164300565 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 128 | ok | 557.527846 | 0.2006405 | 0.2136592 | 0.3182250299999996 | 19483.762135217698 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 1 | ok | 500.398116 | 0.28722749999999997 | 0.29817735 | 0.30419981 | 27658.55213149595 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 2 | ok | 512.723928 | 0.1907505 | 0.28529525 | 0.29052316 | 43185.388828285395 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 4 | ok | 510.301386 | 0.182733 | 0.1904764 | 0.19364249 | 49358.780087878375 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 8 | ok | 518.801076 | 0.146521 | 0.15781275 | 0.16268139 | 54937.42558554025 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 128 | ok | 659.247218 | 0.40943799999999997 | 0.46059659999999997 | 0.46731916 | 19488.821845176317 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 1 | ok | 498.841748 | 0.352129 | 0.36977855 | 0.37280855 | 45411.505152417754 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 2 | ok | 506.991672 | 0.2557655 | 0.3910667 | 0.39600657 | 59586.72139707028 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 4 | ok | 504.596576 | 0.1780565 | 0.3004961 | 0.30356246 | 74378.3849719217 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 8 | ok | 507.529231 | 0.15702 | 0.2613085 | 0.26463787 | 82341.6567367775 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 128 | ok | 644.468871 | 0.482584 | 0.5122228 | 0.53311226 | 32989.5655653595 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 1 | ok | 498.119056 | 0.4616365 | 0.476968 | 0.4957183499999999 | 69046.52190660578 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 2 | ok | 514.561947 | 0.3419485 | 0.37741984999999995 | 0.3917149 | 92169.46646782641 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 4 | ok | 504.740257 | 0.236653 | 0.3037543 | 0.30811847 | 125699.21168525012 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 8 | ok | 514.861703 | 0.2180435 | 0.29732974999999984 | 0.36237531 | 138949.64317297415 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 128 | ok | 637.243516 | 0.7884990000000001 | 0.8388656 | 0.8846098499999999 | 40526.23416377244 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 1 | ok | 499.267612 | 0.699838 | 0.725175 | 0.7303957799999999 | 91489.54790525675 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 2 | ok | 505.720609 | 0.474066 | 0.4892514 | 0.5111765399999999 | 134657.45374032538 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 4 | ok | 512.841498 | 0.32925499999999996 | 0.3554924 | 0.36087741 | 191675.1298224639 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 8 | ok | 505.610663 | 0.2759495 | 0.32089175 | 0.32276546 | 224554.86384080743 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 128 | ok | 560.059533 | 1.1205705 | 1.1782985000000001 | 1.23387355 | 56857.83591862655 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 1 | ok | 502.120562 | 1.0877634999999999 | 1.1142645999999998 | 1.2611306999999996 | 116786.78034959504 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 2 | ok | 509.769428 | 0.722389 | 0.73963255 | 0.74243496 | 176874.2305107329 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 4 | ok | 506.788476 | 0.491012 | 0.51204805 | 0.6125860999999997 | 258314.74812536142 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 8 | ok | 508.294049 | 0.390131 | 0.43321165 | 0.44056859 | 324532.79827340436 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 128 | ok | 655.303015 | 1.0124355 | 1.0521442 | 1.08153548 | 126036.84009572184 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 1 | ok | 500.244369 | 0.17931350000000001 | 0.18972555 | 0.1947094 | 5545.144239182921 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 2 | ok | 510.4778 | 0.168414 | 0.17969544999999998 | 0.18983387999999998 | 5910.9682863547305 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 4 | ok | 510.437796 | 0.1619855 | 0.1896906 | 0.22523116999999987 | 6048.762217592364 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 8 | ok | 510.142482 | 0.147721 | 0.1555829 | 0.15902249 | 6881.889976196919 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 128 | ok | 650.27538 | 0.528235 | 0.56782165 | 0.5959421199999999 | 1888.3829701091638 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 1 | ok | 499.047587 | 0.175453 | 0.19074815 | 0.19414107 | 11265.957947784087 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 2 | ok | 506.68885 | 0.1627295 | 0.1697414 | 0.17680781 | 12237.228463548663 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 4 | ok | 514.965526 | 0.171728 | 0.1835448 | 0.18897601 | 12051.469899706462 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 8 | ok | 507.180955 | 0.1543515 | 0.16951975 | 0.17826039999999999 | 13114.700779577157 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 128 | ok | 639.758412 | 0.5108809999999999 | 0.5543897 | 0.56032069 | 3905.722574885102 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 1 | ok | 504.965929 | 0.19719150000000002 | 0.21099729999999997 | 0.21640245 | 20199.369799861615 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 2 | ok | 508.574126 | 0.1697265 | 0.23220095 | 0.23696724 | 20947.65296735549 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 4 | ok | 512.082277 | 0.1802935 | 0.19280445 | 0.19443261 | 23767.000684133112 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 8 | ok | 511.32521 | 0.1549225 | 0.1657043 | 0.17833746999999994 | 26010.667234739638 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 128 | ok | 550.201186 | 0.469254 | 0.5091471 | 0.51410705 | 8525.014737619227 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 1 | ok | 498.274655 | 0.182962 | 0.19430989999999998 | 0.20347156999999996 | 43183.276844209315 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 2 | ok | 511.540411 | 0.202929 | 0.2168559 | 0.22774840999999998 | 43496.070619350336 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 4 | ok | 508.11748 | 0.19363049999999998 | 0.20282485 | 0.20794129 | 42984.89704149998 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 8 | ok | 502.46158 | 0.174644 | 0.18477364999999998 | 0.19546223999999998 | 46983.71039522814 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 128 | ok | 652.811401 | 0.5238455 | 0.56390825 | 0.5868629799999999 | 15124.287423924363 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 1 | ok | 498.025374 | 0.1961165 | 0.20560575 | 0.21321804 | 80906.47615888414 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 2 | ok | 506.621017 | 0.199909 | 0.2775486 | 0.2833963 | 76336.78836818188 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 4 | ok | 510.015276 | 0.1894995 | 0.22649065000000002 | 0.23076189 | 81449.30900442472 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 8 | ok | 504.41333 | 0.193038 | 0.2034428 | 0.21001782 | 86595.96250654612 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 128 | ok | 547.349249 | 0.5756924999999999 | 0.6109338 | 0.7986438699999994 | 27548.300435152953 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 1 | ok | 499.981249 | 0.2290925 | 0.24232299999999998 | 0.25105077 | 139012.21396064913 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 2 | ok | 515.307167 | 0.223033 | 0.29772899999999985 | 0.32787083 | 133646.54514910234 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 4 | ok | 506.569336 | 0.19081199999999998 | 0.25738055 | 0.26348761 | 151859.88495854416 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 8 | ok | 511.629776 | 0.198096 | 0.2434639 | 0.25197551 | 154254.81043626348 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 128 | ok | 660.695434 | 0.705678 | 0.7538962 | 0.7830314899999999 | 45169.863671987514 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 1 | ok | 501.814489 | 0.247921 | 0.26778935 | 0.26992318 | 255455.34876001577 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 2 | ok | 519.080755 | 0.2525035 | 0.30003825 | 0.30717585999999997 | 242262.05531948045 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 4 | ok | 512.154972 | 0.23946 | 0.29714705 | 0.3139338199999999 | 261701.23467371255 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 8 | ok | 509.93842 | 0.2311875 | 0.30205945 | 0.30770029 | 262096.0393602727 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 128 | ok | 555.733228 | 0.7867915 | 0.83359025 | 0.9747973699999994 | 80310.54078357489 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 1 | ok | 499.455496 | 0.30661649999999996 | 0.3165507 | 0.32059877 | 415773.970717429 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 2 | ok | 511.331782 | 0.36732299999999996 | 0.38721465 | 0.39085138999999997 | 346837.5434082089 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 4 | ok | 514.251617 | 0.32180050000000004 | 0.34303975 | 0.34839033999999997 | 393802.58110517985 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 8 | ok | 510.39478 | 0.30368249999999997 | 0.3311773 | 0.33797974999999997 | 415881.7440260862 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 128 | ok | 556.119041 | 1.176208 | 1.2541859499999999 | 1.28121891 | 108212.31547116039 | - |
