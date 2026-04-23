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

### full_mlp_capacity_search_hd64_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2984437.093` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.028` ms, throughput=`35107.305` samples/s

### full_mlp_capacity_search_hd64_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4124721.259` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.018` ms, throughput=`55149.228` samples/s

### full_mlp_capacity_search_hd64_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3186954.203` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.026` ms, throughput=`37891.303` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.027663 | 0.0311351 | 0.03247231 | 35107.305479197166 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.02793 | 0.03139955 | 0.03284035 | 34954.32867415435 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.030851499999999997 | 0.03772324999999999 | 0.039477929999999994 | 31981.680893184384 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0280505 | 0.03290585 | 0.034848449999999996 | 34209.53063839774 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.027931499999999998 | 0.03227914999999999 | 0.03580306999999999 | 35140.7386583266 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0296835 | 0.035486699999999996 | 0.03840634999999999 | 63928.808878432974 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.029882 | 0.03437375 | 0.03835229 | 64397.305616733 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.029403 | 0.0347388 | 0.037209849999999996 | 65254.7283576168 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.029284499999999998 | 0.0361548 | 0.036964819999999995 | 65109.8892155235 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0294635 | 0.0326821 | 0.037047779999999995 | 66878.89569567428 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.029194499999999998 | 0.0354772 | 0.0368315 | 131329.76641687745 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.029335 | 0.03447984999999999 | 0.03589593 | 130767.3296134387 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.032039 | 0.03714355 | 0.04002231 | 121778.72440439547 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0292825 | 0.0323026 | 0.034234379999999995 | 134691.9258925024 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.029495 | 0.03548674999999999 | 0.04177131999999999 | 131966.2509509818 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0306355 | 0.03599944999999999 | 0.03925087 | 257446.14709314337 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0303555 | 0.035117749999999996 | 0.04050455999999998 | 252561.44662145394 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0297725 | 0.03581255 | 0.03877057999999999 | 255989.84232305663 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.029811499999999998 | 0.033329000000000004 | 0.0400615 | 260396.1536884139 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.029553 | 0.031671149999999995 | 0.03225895 | 268775.8402772692 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0300905 | 0.03483 | 0.037830539999999996 | 507112.89220428024 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0298595 | 0.0357182 | 0.05738850999999994 | 494286.0532247222 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.03052 | 0.03475205 | 0.036908819999999995 | 511727.18800153263 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0301215 | 0.034816 | 0.041170269999999995 | 512377.76587922743 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.0302795 | 0.032447649999999995 | 0.03431052 | 524593.9478907717 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.031993999999999995 | 0.03731365 | 0.04130911999999999 | 965580.0873246492 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.031821 | 0.0367848 | 0.04090389999999999 | 965806.2339170626 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.032682 | 0.038738249999999995 | 0.07776836999999986 | 907815.8404786006 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0322225 | 0.037856499999999994 | 0.041856769999999995 | 946510.894636182 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.034936499999999995 | 0.03675969999999999 | 0.0868693299999998 | 894143.4720261627 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0348435 | 0.04289305 | 0.04524772 | 1749283.8869087968 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.050473500000000004 | 0.056847949999999994 | 0.06287083 | 1265038.140899948 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.047723 | 0.0531441 | 0.054306890000000003 | 1311774.8600807644 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.034981 | 0.0404303 | 0.04537151999999999 | 1752221.6253717311 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.0349715 | 0.039038949999999996 | 0.048792089999999996 | 1786552.7292663574 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0410885 | 0.046752 | 0.05526391999999997 | 2984437.0931955767 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0689835 | 0.0774739 | 0.08139595 | 1864804.0382331447 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.072646 | 0.0856473 | 0.08894429999999999 | 1736915.8670521174 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0630115 | 0.0707087 | 0.10517572999999987 | 1948664.8600919528 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2030345 | 0.22109075 | 0.24337846999999996 | 623874.892808058 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.035593 | 0.04251115 | 0.04397498 | 27055.779276788195 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.035417500000000005 | 0.04008915 | 0.044497959999999996 | 27265.162429478656 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0356925 | 0.0412733 | 0.043370729999999996 | 27470.384178816817 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.034700999999999996 | 0.04048455 | 0.04273180999999999 | 27683.99056972545 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.035450999999999996 | 0.0388577 | 0.04430555999999999 | 27882.569769160204 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.056802000000000005 | 0.06607009999999999 | 0.07350401999999998 | 34476.38643340403 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.057869 | 0.06829325 | 0.07652764999999997 | 33316.35309870403 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.055848 | 0.0689076 | 0.0998929499999999 | 33583.756208796935 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.056329000000000004 | 0.06643095 | 0.06882863 | 34457.27212113798 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.1649025 | 0.22369114999999998 | 0.24122333999999998 | 11702.303692100219 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.053849 | 0.0632236 | 0.06670888 | 73006.26284225791 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0555065 | 0.0650959 | 0.06680512999999999 | 70250.9151937977 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.063068 | 0.07105304999999999 | 0.07252636 | 62030.852905618194 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.064466 | 0.0756453 | 0.1261550699999998 | 58468.02661347635 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.16795900000000002 | 0.18675015 | 0.20311184 | 23524.308373571665 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.052593 | 0.06083205 | 0.06422067 | 146457.8080685071 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.059909500000000004 | 0.07030725 | 0.07138145 | 130189.51687972182 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.058081 | 0.06610229999999999 | 0.06727501 | 133669.37815000245 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.05749 | 0.0667506 | 0.07165907999999999 | 134015.29248502545 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.1259225 | 0.15571854999999996 | 0.19289700999999992 | 61537.20239262797 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.055220000000000005 | 0.06419029999999999 | 0.06598864 | 287318.65254734916 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.061432 | 0.06969155 | 0.07454947999999999 | 255324.88016486328 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0580165 | 0.0701402 | 0.08920869999999995 | 261744.47457414176 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.058566 | 0.07341494999999998 | 0.08416325999999998 | 256116.95324553005 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1419585 | 0.15692545 | 0.16788134999999996 | 111481.46181739031 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.054055500000000006 | 0.060962749999999996 | 0.06366645 | 583026.2121297145 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0603615 | 0.07012625 | 0.07236585 | 511413.79704927024 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0671675 | 0.08434224999999998 | 0.08869965 | 459255.69577524945 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.0625855 | 0.08502064999999999 | 0.13019063999999986 | 472401.7020633325 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.138527 | 0.16017579999999998 | 0.16641223 | 230018.83854287668 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.064886 | 0.07662264999999999 | 0.08935318999999997 | 944942.343752307 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.07241049999999999 | 0.08426414999999998 | 0.08692865 | 866028.8544576264 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0821845 | 0.11166219999999999 | 0.14913176999999989 | 740338.5244174056 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.0797255 | 0.09488574999999999 | 0.09578988000000001 | 778398.0907840829 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.21733 | 0.24416409999999997 | 0.2538131 | 292575.78878661233 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.062918 | 0.07199525 | 0.07280755 | 1998526.0870108297 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.081961 | 0.10032565 | 0.10174000999999999 | 1525774.8671503055 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.084977 | 0.10076375 | 0.13569942999999987 | 1452602.5985698672 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.0847095 | 0.10612785 | 0.17205364999999997 | 1424434.5273136434 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.264722 | 0.28273729999999997 | 0.30421721999999995 | 479579.9359341153 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 1 | ok | 42.861864 | 0.018002999999999998 | 0.018798999999999996 | 0.02181363 | 54962.20798578897 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 2 | ok | 40.927702 | 0.0181495 | 0.020418999999999996 | 0.02291462 | 53711.981316824415 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 4 | ok | 40.794699 | 0.018025 | 0.0185811 | 0.021073009999999996 | 55115.67125927184 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 8 | ok | 41.22316 | 0.0175735 | 0.019711899999999997 | 0.021564979999999994 | 55149.22829684845 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 128 | ok | 44.442183 | 0.0180465 | 0.02074805 | 0.022565019999999995 | 54284.27807881426 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 1 | ok | 42.190168 | 0.019726 | 0.021245499999999997 | 0.025981749999999994 | 101484.00053989489 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 2 | ok | 41.398325 | 0.0195655 | 0.01997355 | 0.023158089999999996 | 101457.23019732418 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 4 | ok | 42.246088 | 0.019077999999999998 | 0.01955465 | 0.023410329999999993 | 104149.52539061279 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 8 | ok | 41.028814 | 0.0193295 | 0.01966275 | 0.02323016999999999 | 103091.61442499106 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 128 | ok | 48.190901 | 0.018282 | 0.019027549999999997 | 0.01992079 | 108874.34811484067 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 1 | ok | 41.607353 | 0.0192815 | 0.01990685 | 0.022249629999999992 | 207171.23220269632 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 2 | ok | 42.165163 | 0.0190975 | 0.0196731 | 0.024116679999999995 | 208251.7680575108 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 4 | ok | 41.779276 | 0.018975 | 0.01936005 | 0.01978647 | 212729.0693209582 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 8 | ok | 42.506371 | 0.018956 | 0.019979999999999998 | 0.02417410999999999 | 211478.6374854344 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 128 | ok | 46.387839 | 0.0192945 | 0.021369049999999997 | 0.02274726 | 202055.9189755765 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 1 | ok | 41.361427 | 0.01924 | 0.019823499999999997 | 0.02283755999999999 | 414877.07710801635 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 2 | ok | 41.460401 | 0.019542499999999997 | 0.01983395 | 0.023830969999999986 | 407068.3345613229 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 4 | ok | 41.394492 | 0.0195565 | 0.02013725 | 0.02524023 | 404309.53533715877 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 8 | ok | 42.419099 | 0.0194125 | 0.01980465 | 0.02757676 | 410277.8709450443 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 128 | ok | 46.19116 | 0.0187325 | 0.019268 | 0.02345330999999999 | 422278.5729517905 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 1 | ok | 41.90214 | 0.020003 | 0.0205779 | 0.02461589999999999 | 794873.0687068409 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 2 | ok | 43.004465 | 0.0203095 | 0.02101495 | 0.024262179999999994 | 785539.0123221612 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 4 | ok | 42.485676 | 0.020295 | 0.0206576 | 0.022417139999999995 | 785810.6176766134 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 8 | ok | 41.34311 | 0.020071 | 0.0232986 | 0.024737549999999997 | 758123.5305433755 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 128 | ok | 50.54354 | 0.020456000000000002 | 0.0209257 | 0.02869399999999999 | 771456.1234329797 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 1 | ok | 43.512597 | 0.021845999999999997 | 0.0226212 | 0.024987519999999992 | 1461958.915299583 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 2 | ok | 41.76417 | 0.0219115 | 0.02248425 | 0.023729969999999996 | 1464094.0094763483 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.028647 | 0.021676 | 0.0220197 | 0.025864069999999986 | 1469251.3246218974 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 8 | ok | 41.738875 | 0.021713 | 0.02239175 | 0.023590409999999996 | 1478684.3036736986 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 128 | ok | 45.038909 | 0.022095 | 0.027067599999999997 | 0.031690159999999995 | 1377116.455853089 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 1 | ok | 43.053224 | 0.024565 | 0.02539245 | 0.027426319999999997 | 2606274.9326848052 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 2 | ok | 42.953017 | 0.0362175 | 0.0378366 | 0.04013404999999999 | 1759825.6012829128 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 4 | ok | 44.019721 | 0.033083 | 0.03611604999999999 | 0.0404131 | 1908891.0178326212 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 8 | ok | 42.908255 | 0.024648999999999997 | 0.026061799999999996 | 0.030390829999999994 | 2594896.163608203 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 128 | ok | 48.416556 | 0.024697999999999998 | 0.02792249999999999 | 0.03089127 | 2573522.3156553796 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 1 | ok | 44.419472 | 0.0308055 | 0.03353339999999999 | 0.03593126 | 4124721.259071164 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 2 | ok | 43.204928 | 0.0543485 | 0.0570256 | 0.05744096 | 2348454.936819223 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 4 | ok | 43.51271 | 0.052892 | 0.05738695 | 0.06359230999999999 | 2396173.610266106 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 8 | ok | 45.111616 | 0.0438165 | 0.047251749999999995 | 0.054945709999999974 | 2886626.395221551 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 128 | ok | 82.503514 | 0.201403 | 0.23332129999999998 | 0.4058320799999998 | 604698.3169545125 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 1 | ok | 42.426588 | 0.022903 | 0.023949799999999997 | 0.025969659999999995 | 43361.11046069446 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 2 | ok | 41.374361 | 0.022717 | 0.023823749999999998 | 0.02478278 | 43798.60002154891 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 4 | ok | 42.376378 | 0.0230335 | 0.02400735 | 0.02492951 | 43183.785006935315 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 8 | ok | 41.919871 | 0.0229875 | 0.02402835 | 0.026699859999999995 | 43201.246960792276 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 128 | ok | 46.40204 | 0.0231935 | 0.02458255 | 0.027056439999999994 | 42730.11233746534 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 1 | ok | 42.19525 | 0.037667000000000006 | 0.0410037 | 0.04366832999999999 | 52526.49830523253 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 2 | ok | 42.79117 | 0.039678 | 0.04294525 | 0.04897235999999999 | 49734.02244794837 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 4 | ok | 40.837792 | 0.040187 | 0.0457152 | 0.05054550999999999 | 49175.668272743984 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 8 | ok | 42.354617 | 0.041671 | 0.045321299999999995 | 0.051672279999999994 | 47589.610045976326 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 128 | ok | 44.133449 | 0.1365255 | 0.14814235 | 0.1508851 | 14704.120020909259 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 1 | ok | 41.454666 | 0.0381785 | 0.0399024 | 0.04067854 | 104530.07199508707 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 2 | ok | 41.507956 | 0.040832499999999994 | 0.04503484999999999 | 0.049930119999999995 | 96395.80890302053 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 4 | ok | 40.993963 | 0.0401385 | 0.0430761 | 0.047716569999999986 | 98875.53794472362 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 8 | ok | 42.619819 | 0.0405955 | 0.04410715 | 0.052445490000000004 | 97078.47192660478 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 128 | ok | 46.153284 | 0.11587049999999999 | 0.1389927 | 0.14297909 | 33707.0205993715 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 1 | ok | 41.254687 | 0.038482 | 0.0431513 | 0.04879803999999999 | 205064.687655721 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 2 | ok | 41.62823 | 0.040544 | 0.04953915 | 0.053608949999999995 | 192400.84021446924 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 4 | ok | 41.614112 | 0.041788 | 0.048905399999999995 | 0.05219510999999999 | 187212.71623653764 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 8 | ok | 41.400191 | 0.041775 | 0.043123499999999995 | 0.050256139999999984 | 190257.48496728047 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 128 | ok | 46.067763 | 0.11760999999999999 | 0.15074879999999993 | 0.17547315999999996 | 66455.33868963363 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 1 | ok | 42.746937 | 0.039009 | 0.042210899999999996 | 0.04690674999999998 | 404588.84669926297 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 2 | ok | 41.460923 | 0.045792 | 0.050114549999999994 | 0.058406969999999996 | 344557.7321554627 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 4 | ok | 41.646632 | 0.0529825 | 0.05683409999999999 | 0.06777774999999998 | 302740.97898107226 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 8 | ok | 41.759098 | 0.0481855 | 0.057006249999999994 | 0.061865329999999996 | 322853.6387421138 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 128 | ok | 44.173653 | 0.13333699999999998 | 0.1490932 | 0.15518739 | 119224.62858920157 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 1 | ok | 43.077124 | 0.0405895 | 0.04420935 | 0.04959962 | 775166.0066451106 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 2 | ok | 43.241989 | 0.0490915 | 0.051476499999999994 | 0.05666617999999998 | 652824.8752390462 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 4 | ok | 42.961645 | 0.054783 | 0.06049179999999999 | 0.06406854000000001 | 579463.6629201927 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 8 | ok | 42.402857 | 0.0530635 | 0.05779145 | 0.06208717999999999 | 599079.9629469042 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 128 | ok | 43.406989 | 0.1684375 | 0.19341994999999998 | 0.21059074 | 188826.476493228 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 1 | ok | 43.119688 | 0.0424975 | 0.04752525 | 0.05062675 | 1485081.4288710968 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 2 | ok | 44.124834 | 0.052022 | 0.05685855 | 0.05897470999999999 | 1217814.95834502 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 4 | ok | 43.861813 | 0.055399500000000004 | 0.061184699999999995 | 0.06795285999999999 | 1147324.3320958964 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 8 | ok | 41.764674 | 0.06257499999999999 | 0.06539819999999999 | 0.07088505999999999 | 1023746.4384821168 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 128 | ok | 44.185544 | 5.992296 | 6.118556949999999 | 7.411273839999999 | 14179.376306308275 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 1 | ok | 42.6704 | 0.0468765 | 0.0510841 | 0.05567749999999999 | 2702348.932361051 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 2 | ok | 43.514949 | 0.0600865 | 0.06583955 | 0.07149517999999998 | 2118195.998661565 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 4 | ok | 44.963791 | 0.067595 | 0.0729157 | 0.07786560999999999 | 1910251.6040143934 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 8 | ok | 44.32756 | 0.067111 | 0.0718777 | 0.07583593 | 1898817.9265026918 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 128 | ok | 89.6505 | 0.17640450000000002 | 0.19515885 | 0.20976700999999995 | 715127.5854655682 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 1 | ok | 503.273346 | 0.0258785 | 0.029743299999999997 | 0.03198043 | 37891.303491153136 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 2 | ok | 510.316093 | 0.0302265 | 0.032986499999999995 | 0.03349819 | 33038.912570447224 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 4 | ok | 512.759703 | 0.026373 | 0.0300624 | 0.03145849999999999 | 36874.02652569972 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 8 | ok | 546.328773 | 0.026074 | 0.03124255 | 0.03324485999999999 | 36815.86911222213 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 128 | ok | 707.502041 | 0.0269605 | 0.0318714 | 0.033010229999999995 | 35696.437495537946 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 1 | ok | 498.583329 | 0.0277535 | 0.030594299999999998 | 0.03223532 | 70883.37699749356 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 2 | ok | 509.297808 | 0.0271215 | 0.0295069 | 0.03252703999999999 | 72981.67357194934 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 4 | ok | 508.589378 | 0.028374 | 0.031603650000000004 | 0.03205505 | 69896.24601241916 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 8 | ok | 539.505027 | 0.027984000000000002 | 0.029921649999999998 | 0.03102199 | 71224.51319825841 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 128 | ok | 840.875746 | 0.027968 | 0.03028845 | 0.03491050999999999 | 70918.36445231165 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 1 | ok | 503.822455 | 0.0285105 | 0.0313174 | 0.03326037 | 138802.8118673628 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 2 | ok | 508.321319 | 0.029785 | 0.03243855 | 0.03655766999999999 | 133035.8651388828 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 4 | ok | 507.040344 | 0.0272985 | 0.029633 | 0.038365929999999986 | 143350.37052487023 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 8 | ok | 539.788829 | 0.028346 | 0.0298486 | 0.03439692999999999 | 141061.4308425176 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 128 | ok | 839.450509 | 0.027251 | 0.03003655 | 0.03224555999999999 | 144007.76489868335 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 1 | ok | 497.740795 | 0.029156 | 0.03418524999999999 | 0.038380449999999997 | 271468.7681968909 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 2 | ok | 509.873059 | 0.027859000000000002 | 0.03043325 | 0.034651289999999994 | 284465.4822472204 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 4 | ok | 518.665197 | 0.0293385 | 0.03303925 | 0.03596681999999999 | 267709.484344684 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 8 | ok | 536.78135 | 0.029323500000000002 | 0.03461379999999999 | 0.03941321 | 265148.25101584924 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 128 | ok | 744.842635 | 0.029806 | 0.033549100000000005 | 0.03614442999999999 | 262327.0772697522 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 1 | ok | 502.1163 | 0.0322195 | 0.035852699999999994 | 0.03952839999999999 | 496723.79611226707 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 2 | ok | 508.304136 | 0.03024 | 0.03227005 | 0.03313362 | 533199.3225702607 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 4 | ok | 510.013012 | 0.030677999999999997 | 0.0347443 | 0.0359824 | 514991.39964362595 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 8 | ok | 542.318858 | 0.030588 | 0.0344096 | 0.03522911 | 515550.28548597055 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 128 | ok | 760.72909 | 0.028389 | 0.03237705 | 0.035282469999999996 | 550539.90760564 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 1 | ok | 503.534984 | 0.032604 | 0.037094499999999996 | 0.03796122 | 961751.1565057657 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 2 | ok | 515.556406 | 0.033103 | 0.0361299 | 0.04045353999999999 | 954497.3229332895 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 4 | ok | 527.833758 | 0.032191 | 0.03790734999999999 | 0.04004859 | 979296.4489486641 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 8 | ok | 539.300352 | 0.030349 | 0.03362925 | 0.04097713999999999 | 1021937.8112519188 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 128 | ok | 732.392318 | 0.030673 | 0.034489599999999995 | 0.03970102 | 1014225.1415002546 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 1 | ok | 500.165477 | 0.0365515 | 0.04074205 | 0.04090888 | 1728907.3305670817 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 2 | ok | 510.676552 | 0.06031 | 0.06475955 | 0.06901084999999998 | 1055937.2878844726 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 4 | ok | 515.054494 | 0.048531500000000005 | 0.05074585 | 0.05653279999999999 | 1315308.1314814892 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 8 | ok | 546.478435 | 0.034344 | 0.03708445 | 0.043999939999999994 | 1840190.91980793 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 128 | ok | 812.170498 | 0.0339305 | 0.0373118 | 0.038315 | 1864026.2734503243 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 1 | ok | 497.581915 | 0.039912500000000004 | 0.0439055 | 0.04591729 | 3186954.202970141 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 2 | ok | 508.882148 | 0.086787 | 0.090845 | 0.09695180999999999 | 1527217.400351451 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 4 | ok | 515.52527 | 0.09572349999999999 | 0.10133835 | 0.10594674 | 1340323.6797916468 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 8 | ok | 538.896524 | 0.059228 | 0.06394064999999999 | 0.06628595 | 2155977.565705943 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 128 | ok | 861.020428 | 5.9969475 | 7.183943349999998 | 7.92671201 | 29614.063874407788 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 1 | ok | 498.834615 | 0.032732 | 0.0351198 | 0.037729969999999995 | 30416.696576418304 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 2 | ok | 500.896882 | 0.033759 | 0.03625635 | 0.04283751999999999 | 29387.511365620023 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 4 | ok | 519.119793 | 0.03314 | 0.0364795 | 0.041853959999999996 | 29732.75604214202 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 8 | ok | 547.391847 | 0.032992 | 0.03774429999999999 | 0.040357979999999995 | 29975.40817513312 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 128 | ok | 739.448308 | 0.0321495 | 0.0353017 | 0.03731784999999999 | 30700.480585323083 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 1 | ok | 501.088839 | 0.0532935 | 0.05730635 | 0.06431865999999999 | 37441.810745874565 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 2 | ok | 501.094511 | 0.051991499999999996 | 0.05801225 | 0.06144851 | 37608.92956336409 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 4 | ok | 509.804988 | 0.052545499999999995 | 0.0566307 | 0.058311289999999995 | 37781.81455476398 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 8 | ok | 538.793788 | 0.052731 | 0.05904225 | 0.0636735 | 37315.39607121121 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 128 | ok | 735.186765 | 0.138536 | 0.1560325 | 0.18094955999999993 | 14192.737661295143 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 1 | ok | 496.215987 | 0.055642 | 0.0597276 | 0.06473214999999999 | 71277.13297711399 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 2 | ok | 514.62903 | 0.0622525 | 0.0675748 | 0.0696637 | 63922.49525295569 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 4 | ok | 520.310886 | 0.054196 | 0.059123 | 0.0603398 | 73613.49894898327 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 8 | ok | 545.04999 | 0.0580155 | 0.06276335 | 0.06627857 | 68295.9578012936 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 128 | ok | 817.725658 | 0.1367625 | 0.17579525 | 0.19847957999999993 | 27990.290168340605 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 1 | ok | 496.797144 | 0.0496015 | 0.054202549999999995 | 0.057970429999999996 | 159208.92270486406 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 2 | ok | 509.100485 | 0.0613475 | 0.06900115 | 0.06953631 | 128732.14605820559 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 4 | ok | 517.068446 | 0.0552305 | 0.0627011 | 0.06905301999999999 | 142656.4557930825 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 8 | ok | 548.299277 | 0.062422000000000005 | 0.0714273 | 0.07387993999999999 | 126253.0616367447 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 128 | ok | 704.046458 | 0.1222665 | 0.1326985 | 0.13731840999999997 | 65122.84284653249 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 1 | ok | 496.956577 | 0.050274 | 0.05516975 | 0.056155659999999996 | 316398.8745692031 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 2 | ok | 517.32178 | 0.062118 | 0.0695194 | 0.07317395 | 254105.06736325336 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 4 | ok | 522.435012 | 0.064325 | 0.06994795 | 0.07576434999999998 | 246356.0096854865 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 8 | ok | 544.613075 | 0.06386800000000001 | 0.07432505 | 0.08325873999999998 | 244468.44309660845 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 128 | ok | 742.42603 | 0.153854 | 0.16891389999999998 | 0.17730585 | 103257.14332917552 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 1 | ok | 496.700307 | 0.060160000000000005 | 0.0662421 | 0.07004605 | 526288.4363246909 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 2 | ok | 505.257062 | 0.060005 | 0.06554905 | 0.06818729 | 531203.9139104377 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 4 | ok | 506.679495 | 0.06475500000000001 | 0.0713405 | 0.07352764 | 488598.85118195124 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 8 | ok | 545.064559 | 0.060172500000000004 | 0.0709469 | 0.07224171 | 524559.8942487253 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 128 | ok | 696.718515 | 0.177899 | 0.2019769 | 0.21345576 | 177849.43133755887 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 1 | ok | 497.347817 | 0.062004500000000004 | 0.0668556 | 0.06967642 | 1019505.3678550593 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 2 | ok | 508.468396 | 0.069489 | 0.07913854999999999 | 0.08255335 | 906629.3588685492 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 4 | ok | 514.734974 | 0.078065 | 0.08325655 | 0.08507926 | 830462.8299110911 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 8 | ok | 547.737371 | 0.0722615 | 0.08121219999999998 | 0.0844124 | 876637.3942830092 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 128 | ok | 724.952691 | 0.16389700000000001 | 7.608773149999996 | 9.10984371 | 56660.278515369675 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 1 | ok | 538.028847 | 0.0602365 | 0.06626575 | 0.07351913999999998 | 2103743.480038761 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 2 | ok | 513.08406 | 0.08065 | 0.09036699999999999 | 0.0969444 | 1573785.8426651866 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 4 | ok | 512.164908 | 0.094256 | 0.10390764999999999 | 0.10622672999999999 | 1356494.6313545678 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 8 | ok | 542.915716 | 0.09221599999999999 | 0.10371544999999999 | 0.10880025 | 1405807.6991255218 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 128 | ok | 765.618201 | 0.172904 | 5.9752103 | 6.00583904 | 192840.2977405987 | - |
