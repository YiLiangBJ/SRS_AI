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

### full_mlp_capacity_search_hd32_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2267650.827` samples/s, p50=`0.052` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.035` ms, throughput=`28527.838` samples/s

### full_mlp_capacity_search_hd32_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3525646.323` samples/s, p50=`0.036` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.020` ms, throughput=`50057.165` samples/s

### full_mlp_capacity_search_hd32_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2509025.632` samples/s, p50=`0.050` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.032` ms, throughput=`30237.424` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `5,552`
- MACs / sample: `5,376`
- FLOPs / sample estimate: `11,000`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.034856 | 0.041039549999999994 | 0.042308719999999994 | 27564.90017922698 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.034905 | 0.039726649999999995 | 0.0406561 | 27673.494824779737 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.035037 | 0.041700549999999996 | 0.04461311 | 27338.583434896173 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0349605 | 0.043551549999999994 | 0.12107924999999975 | 25251.415720622383 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.034695000000000004 | 0.0361562 | 0.041774269999999995 | 28527.83803491123 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.036971000000000004 | 0.04222475 | 0.044397879999999994 | 52156.189009856476 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.036618 | 0.04291004999999999 | 0.08339129999999984 | 50862.086942634174 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.037058999999999995 | 0.04254095 | 0.04697725999999999 | 52132.91390666332 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037063 | 0.043666699999999996 | 0.08325267999999986 | 49844.932415256146 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.037335 | 0.041347199999999994 | 0.04736380999999998 | 52279.05303814489 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.037843 | 0.042354249999999996 | 0.04399673999999999 | 103369.69706993426 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0386955 | 0.04292005 | 0.04626737999999999 | 102821.52548071634 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.036878999999999995 | 0.041307 | 0.0423881 | 105301.12172019912 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0374315 | 0.04319725 | 0.04494217 | 104038.13621921252 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.037096500000000004 | 0.04658035 | 0.05057585999999999 | 103535.26333417716 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.037428 | 0.043365999999999995 | 0.04439726 | 205933.98786019144 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.037321 | 0.04174175 | 0.04448802999999999 | 207893.83277747722 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.037433499999999995 | 0.04366075 | 0.04650447999999999 | 203880.66456941425 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0370895 | 0.0450849 | 0.05057113999999999 | 203998.57608993887 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.037479 | 0.0408125 | 0.04559729 | 209873.38863147335 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.040511000000000005 | 0.04419185 | 0.04772834999999999 | 397271.53906966955 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.037925 | 0.04462885 | 0.0470917 | 401636.4677880022 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.042921 | 0.0488925 | 0.09295721999999984 | 361676.73333574453 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0386025 | 0.04741349999999999 | 0.049612369999999996 | 394674.65488168143 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.038016999999999995 | 0.04158675 | 0.043779969999999994 | 414183.2926743401 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.039802500000000005 | 0.044779 | 0.04695192 | 778909.4683261333 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.040961 | 0.047489 | 0.048200929999999996 | 750434.5485057441 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.050788 | 0.056466749999999996 | 0.05796192 | 620195.4235779694 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.040057499999999996 | 0.045991849999999994 | 0.04977486 | 769849.9802966521 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.039946 | 0.04364344999999999 | 0.046613169999999995 | 793218.7727120843 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.044 | 0.050005 | 0.052512249999999996 | 1399057.559851245 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.060924000000000006 | 0.07159194999999999 | 0.07601684999999998 | 1042335.4307076578 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0553385 | 0.062496 | 0.06600326 | 1129350.912621414 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0532775 | 0.0639748 | 0.06746303 | 1161427.118089191 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.1091325 | 0.11996095 | 0.127634 | 577767.6106727399 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.052038 | 0.06100245 | 0.11559775999999995 | 2267650.8271256397 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.076936 | 0.09433304999999999 | 0.14251423999999985 | 1586509.1197006258 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.0835645 | 0.10570824999999999 | 0.14414582999999986 | 1453064.9794765925 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.06991 | 0.07856534999999999 | 0.08328248999999999 | 1799758.8323164696 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.277489 | 0.30267605 | 0.30611142999999996 | 459778.07518075564 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.06318850000000001 | 0.07333229999999999 | 0.07531594 | 15480.758036990343 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.06532299999999999 | 0.07299499999999999 | 0.07479954 | 15168.977862697086 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.06283 | 0.07187235 | 0.07232476 | 15437.030038916755 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.070465 | 0.0769295 | 0.08183170999999999 | 14080.853953101183 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.14554299999999998 | 0.1622787 | 0.16908712999999997 | 6869.0952398681175 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.066908 | 0.08621779999999997 | 0.16404639999999981 | 27766.069751699146 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0763165 | 0.09767065 | 0.1658593999999998 | 24337.970690268656 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.07585 | 0.09114475 | 0.0981256 | 25181.495629751433 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0677295 | 0.08125699999999998 | 0.12156908999999985 | 28006.301417819006 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.15195750000000002 | 0.25280764999999994 | 0.27200081 | 11945.397587626956 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.06571650000000001 | 0.07558285 | 0.07853418999999999 | 59581.52913015437 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.06962850000000001 | 0.08025675 | 0.08306568 | 56463.010235049864 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.06651299999999999 | 0.07474539999999999 | 0.08824474999999998 | 58680.17719066305 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.07591300000000001 | 0.08891285 | 0.09086373 | 50707.45777399838 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.1408005 | 0.15474369999999998 | 0.15900416 | 28166.939252221277 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.08578 | 0.10770779999999998 | 0.17800770999999982 | 88622.29333668917 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.081597 | 0.1002701 | 0.11596796999999993 | 93016.8744237314 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0814935 | 0.0927186 | 0.13849708999999985 | 95138.08698713138 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.081846 | 0.09795164999999999 | 0.10231380999999999 | 94098.05252318046 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.145029 | 0.16225949999999997 | 0.17085135999999998 | 54631.58372044363 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.076417 | 0.08981679999999999 | 0.1685841799999998 | 196135.15673650714 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.08269 | 0.09508654999999999 | 0.10902697999999995 | 188861.3361184085 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.091173 | 0.10437039999999999 | 0.10713889 | 171818.83336300787 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.08744099999999999 | 0.10553369999999998 | 0.17168835999999996 | 171377.03590563123 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1546925 | 0.17150015 | 0.18565983 | 101939.63084091785 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.076382 | 0.08911324999999999 | 0.10373363999999996 | 402130.18411781813 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.08587 | 0.0987108 | 0.10282669 | 367497.42522116454 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.09816849999999999 | 0.1183136 | 0.14935852999999988 | 312261.2299384358 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.09618850000000001 | 0.1166257 | 0.15549977999999987 | 317246.30209779117 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.1472075 | 0.16027755 | 0.16529343999999999 | 214810.36943355846 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.0839765 | 0.0968014 | 0.18458131999999972 | 716613.2677811347 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.09813250000000001 | 0.1141094 | 0.11599387 | 638107.8825067909 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.103498 | 0.12305169999999999 | 0.17257922999999983 | 593187.3178034422 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1154935 | 0.13577129999999998 | 0.18093827999999984 | 525772.2896643847 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.252012 | 0.260209 | 0.26822777 | 253659.1521728205 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.086974 | 0.11721619999999995 | 0.1845455399999999 | 1357279.7060810798 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.114541 | 0.13723635 | 0.20846568999999998 | 1058466.1997838414 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.118463 | 0.14123639999999998 | 0.17928109999999986 | 1048865.4962748068 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.115703 | 0.1337893 | 0.139321 | 1090637.0513264022 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.32678799999999997 | 0.3513263 | 0.35757379 | 389237.9359806986 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 1 | ok | 46.452538 | 0.020028999999999998 | 0.02062905 | 0.022321349999999997 | 50057.1652827529 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 2 | ok | 46.770938 | 0.020229499999999997 | 0.02114735 | 0.024593549999999992 | 49369.45334191704 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 4 | ok | 46.343774 | 0.0202125 | 0.0214328 | 0.02264353 | 49306.30953120547 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 8 | ok | 44.772769 | 0.020108 | 0.021600549999999996 | 0.028907799999999977 | 48707.355492582356 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 128 | ok | 51.619694 | 0.020161 | 0.02068905 | 0.023935909999999994 | 49678.1847193878 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 1 | ok | 48.254875 | 0.021337000000000002 | 0.0222225 | 0.02816392 | 93419.2666027056 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 2 | ok | 46.521328 | 0.0216605 | 0.022028100000000002 | 0.028348839999999993 | 91328.03752486406 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 4 | ok | 45.218102 | 0.0224115 | 0.0241417 | 0.02726528999999999 | 89966.32560432631 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.170703 | 0.0213145 | 0.02177735 | 0.029324969999999995 | 92732.5500523939 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 128 | ok | 52.558736 | 0.0204185 | 0.021460499999999997 | 0.024121699999999996 | 97088.22698450764 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 1 | ok | 46.120811 | 0.0215165 | 0.024658249999999996 | 0.02900619999999999 | 180437.34403447076 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.936871 | 0.0215855 | 0.0220923 | 0.025129169999999992 | 186124.0777551947 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 4 | ok | 45.093724 | 0.021525000000000002 | 0.023336449999999998 | 0.028040139999999984 | 180636.65388662342 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 8 | ok | 45.160953 | 0.021462000000000002 | 0.022067800000000002 | 0.02553547999999999 | 184978.96789135074 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 128 | ok | 49.48793 | 0.021397 | 0.02187225 | 0.027155449999999994 | 185471.80782153158 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.268983 | 0.0215065 | 0.023391449999999998 | 0.025051159999999996 | 363218.99199465336 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 2 | ok | 45.521115 | 0.021987 | 0.02450935 | 0.026182379999999998 | 356799.7559489669 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 4 | ok | 46.291879 | 0.0221615 | 0.0239547 | 0.026237139999999992 | 353163.7735196037 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 8 | ok | 45.387382 | 0.0215465 | 0.0223499 | 0.024107779999999995 | 372498.78705082467 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 128 | ok | 50.905298 | 0.021920000000000002 | 0.0225689 | 0.025355339999999994 | 363827.86939670966 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 1 | ok | 48.899431 | 0.0228815 | 0.0234947 | 0.027143579999999987 | 700449.3382504876 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 2 | ok | 45.228227 | 0.023025999999999998 | 0.023671150000000002 | 0.024427599999999997 | 698870.625069887 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 4 | ok | 46.634382 | 0.0227055 | 0.028404549999999997 | 0.030597219999999994 | 661283.8164011613 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 8 | ok | 46.072257 | 0.022940000000000002 | 0.0282164 | 0.029398909999999997 | 646292.8239684156 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 128 | ok | 52.40428 | 0.022544500000000002 | 0.0284968 | 0.03003915 | 659740.4910778346 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 1 | ok | 47.239594 | 0.02481 | 0.02517835 | 0.028084469999999993 | 1288274.8469972326 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 2 | ok | 45.722238 | 0.02477 | 0.0306491 | 0.03157544 | 1234055.6158014652 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 4 | ok | 45.947552 | 0.031670500000000004 | 0.03487959999999999 | 0.03795195999999999 | 1002549.6089743227 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.733567 | 0.025083 | 0.02591375 | 0.029672189999999987 | 1277489.9876722216 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 128 | ok | 51.383782 | 0.024017 | 0.025824699999999996 | 0.028085699999999998 | 1319631.1630899166 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 1 | ok | 48.700915 | 0.027585 | 0.02904235 | 0.031109369999999994 | 2289090.6944886562 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 2 | ok | 46.437688 | 0.0412415 | 0.04480099999999999 | 0.04752191 | 1539780.7159787857 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 4 | ok | 46.617165 | 0.03701 | 0.0413455 | 0.041805 | 1712192.253507185 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 8 | ok | 47.873331 | 0.037089 | 0.0393362 | 0.04385694999999999 | 1718083.9069228044 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 128 | ok | 65.067551 | 0.0999245 | 0.11759249999999999 | 0.15471543999999993 | 634639.3285198584 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 1 | ok | 48.582127 | 0.035930000000000004 | 0.039882399999999985 | 0.04299738 | 3525646.322585621 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 2 | ok | 47.06275 | 0.0595655 | 0.06201564999999999 | 0.06392291 | 2163105.5977455033 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 4 | ok | 48.80312 | 0.0589495 | 0.06299245 | 0.06392347 | 2165090.886455868 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 8 | ok | 49.153089 | 0.050777 | 0.0534261 | 0.05853508 | 2516913.4618296307 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 128 | ok | 95.200747 | 0.2712875 | 0.30474995 | 0.31142203 | 466552.72574317653 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 1 | ok | 46.56605 | 0.037298 | 0.0393701 | 0.04088831 | 26803.371006364727 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 2 | ok | 46.522337 | 0.041194999999999996 | 0.04402575 | 0.052402229999999994 | 23986.452451655307 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 4 | ok | 44.405952 | 0.041464 | 0.043250449999999996 | 0.049139999999999996 | 23927.39471348141 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 8 | ok | 45.065786 | 0.0420445 | 0.04616779999999999 | 0.05037331 | 23535.891057067478 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 128 | ok | 48.972498 | 0.1341655 | 0.16229055 | 0.1944833799999999 | 7206.359583507007 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 1 | ok | 45.325253 | 0.042015 | 0.0460793 | 0.05050821999999999 | 46802.78476569356 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 2 | ok | 45.448475 | 0.0433435 | 0.04520415 | 0.053026029999999995 | 45819.51874843068 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 4 | ok | 46.702536 | 0.044268 | 0.04910895 | 0.05491158999999998 | 44416.215471944495 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 8 | ok | 45.241136 | 0.044337 | 0.0466445 | 0.050639119999999996 | 44836.03238775636 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 128 | ok | 50.479854 | 0.11624699999999999 | 0.13092789999999999 | 0.14740960999999997 | 16994.4788337165 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 1 | ok | 47.273892 | 0.042194 | 0.0450959 | 0.051778569999999996 | 93627.3481738922 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 2 | ok | 45.610251 | 0.0445735 | 0.04689165 | 0.05412548999999999 | 88497.1023836252 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 4 | ok | 45.312765 | 0.0444735 | 0.047675550000000004 | 0.05564086999999998 | 88921.7306480749 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 8 | ok | 46.194973 | 0.045727000000000004 | 0.04984459999999999 | 0.056841829999999996 | 86692.2986896459 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 128 | ok | 48.375868 | 0.12187100000000001 | 6.0017435500000005 | 6.00720913 | 1713.5643086076884 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 1 | ok | 46.375935 | 0.054432999999999995 | 0.057458800000000004 | 0.06747268999999997 | 146873.1801495022 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 2 | ok | 45.684033 | 0.055193 | 0.061940449999999994 | 0.06440755 | 144030.73615909636 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 4 | ok | 46.598151 | 0.05636 | 0.06061585 | 0.06589958 | 141249.5713958318 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 8 | ok | 46.283574 | 0.059862 | 0.06561974999999999 | 0.06740402 | 132959.71652988435 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 128 | ok | 48.889462 | 5.997142500000001 | 8.257401449999998 | 8.72524337 | 1587.921809396251 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 1 | ok | 47.008278 | 0.05639 | 0.06006565 | 0.06448892999999999 | 288088.96187142585 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 2 | ok | 45.325068 | 0.0633085 | 0.06863925 | 0.07482031 | 249757.42310281142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 4 | ok | 45.369812 | 0.0626055 | 0.06969315000000001 | 0.07189003999999999 | 252513.85433043906 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 8 | ok | 45.566652 | 0.0599575 | 0.06657525 | 0.07157487 | 263170.44727462326 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 128 | ok | 52.188987 | 0.13574 | 0.14771349999999997 | 0.15932111999999998 | 117236.4725597045 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 1 | ok | 47.277049 | 0.054938 | 0.06284405 | 0.06342024 | 570576.985278044 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 2 | ok | 46.855357 | 0.065604 | 0.07118315 | 0.07800352999999999 | 479459.7926516194 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 4 | ok | 46.123892 | 0.0661535 | 0.07198375 | 0.07643413999999998 | 477409.148293128 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 8 | ok | 46.832079 | 0.0706945 | 0.07770854999999999 | 0.08972300999999996 | 447567.22040056146 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 128 | ok | 52.597376 | 0.140575 | 0.1693203 | 0.1839641 | 224073.0657452782 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 1 | ok | 47.207962 | 0.057176000000000005 | 0.065033 | 0.07115506999999999 | 1100645.8039254532 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 2 | ok | 47.937505 | 0.0743145 | 0.08037055 | 0.08326019 | 864285.3265148423 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 4 | ok | 47.561973 | 0.076205 | 0.08457119999999999 | 0.08648375999999999 | 832401.0008581534 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 8 | ok | 47.850472 | 0.081635 | 0.08802125 | 0.09184938 | 782413.4962416034 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 128 | ok | 62.594447 | 0.2229045 | 0.26197935 | 0.33918545999999977 | 277699.7470415648 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 1 | ok | 47.520199 | 0.0636115 | 0.07262795 | 0.07909245999999998 | 1964728.216689875 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.659952 | 0.0863815 | 0.09168325 | 0.09381745999999999 | 1488028.8082377275 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 4 | ok | 48.814 | 0.0842605 | 0.0945143 | 0.09696827999999999 | 1481169.133083278 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 8 | ok | 47.542936 | 0.0908235 | 0.0968931 | 0.10258147999999997 | 1403922.5596316108 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 128 | ok | 110.164729 | 0.33983450000000004 | 0.36596715 | 0.37720939 | 375725.8774446766 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 1 | ok | 500.018093 | 0.0329225 | 0.03554255 | 0.03624563 | 30091.586753442778 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 2 | ok | 507.727041 | 0.032468 | 0.03641005 | 0.03767532 | 30237.424255252245 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 4 | ok | 514.555266 | 0.035053 | 0.03921215 | 0.041904979999999994 | 28201.139551647004 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 8 | ok | 542.743663 | 0.0358305 | 0.0403161 | 0.0410123 | 27408.32873330216 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 128 | ok | 685.808865 | 0.035167000000000004 | 0.03889595 | 0.04012428 | 28144.548147722853 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 1 | ok | 496.044898 | 0.035078 | 0.038730049999999995 | 0.042071889999999994 | 56066.35116261989 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 2 | ok | 511.993882 | 0.036086 | 0.03973965 | 0.042243579999999996 | 54915.90727119562 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 4 | ok | 513.656647 | 0.0334795 | 0.037109949999999996 | 0.0386843 | 59312.19209558278 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 8 | ok | 541.824048 | 0.034156 | 0.03839445 | 0.04340857999999998 | 57066.58357772099 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 128 | ok | 776.457258 | 0.0335825 | 0.03687195 | 0.040595209999999986 | 58610.56619565262 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 1 | ok | 490.9933 | 0.035819500000000004 | 0.04056125 | 0.04132911 | 110156.72548117835 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 2 | ok | 511.975352 | 0.037956000000000004 | 0.04146335 | 0.042885669999999994 | 103998.64385768409 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 4 | ok | 512.369988 | 0.0350455 | 0.03975495 | 0.0408641 | 111064.03230186316 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 8 | ok | 546.063107 | 0.033345 | 0.0349419 | 0.03838135999999999 | 119270.82587890673 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 128 | ok | 816.557483 | 0.0355395 | 0.0392422 | 0.04218594999999999 | 110375.94599084211 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 1 | ok | 492.992955 | 0.033679 | 0.037705499999999996 | 0.039962489999999996 | 235219.1242556785 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 2 | ok | 512.369727 | 0.0346935 | 0.039032700000000004 | 0.04349865999999999 | 224349.3028906286 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 4 | ok | 514.2993 | 0.0344905 | 0.038958799999999995 | 0.03975234 | 225160.69437306537 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 8 | ok | 537.73932 | 0.0364965 | 0.04152035 | 0.04253348999999999 | 217046.6248707216 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 128 | ok | 765.386606 | 0.0343425 | 0.03869525 | 0.03921693 | 227323.10000511477 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 1 | ok | 495.014429 | 0.035284499999999996 | 0.0408358 | 0.043372509999999996 | 445361.14613690967 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 2 | ok | 503.876338 | 0.0346505 | 0.0370422 | 0.04185224 | 456753.9637679918 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 4 | ok | 509.116267 | 0.0358655 | 0.04101315 | 0.04183634 | 433933.37278510915 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 8 | ok | 545.443937 | 0.038761000000000004 | 0.04340745 | 0.04448971 | 404410.7053579869 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 128 | ok | 666.545951 | 0.038132 | 0.0440807 | 0.04957450999999998 | 407927.8742725499 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 1 | ok | 495.885238 | 0.037264 | 0.04145309999999999 | 0.04416874999999999 | 849770.1902741678 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 2 | ok | 517.548713 | 0.041012 | 0.04538035 | 0.04576428 | 763014.0389814335 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 4 | ok | 510.489341 | 0.0461955 | 0.0492777 | 0.05611218 | 683888.4013809416 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 8 | ok | 546.164623 | 0.0368565 | 0.03953555 | 0.04037897 | 862792.2978531572 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 128 | ok | 844.080511 | 0.040070999999999996 | 0.04564895 | 0.0459978 | 782036.6188646783 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 1 | ok | 498.130798 | 0.042803 | 0.0459115 | 0.04851475999999999 | 1485297.1824376604 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 2 | ok | 507.529948 | 0.0696865 | 0.07326805 | 0.07813226 | 911403.8552952708 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 4 | ok | 506.896788 | 0.0567835 | 0.06117355 | 0.0665148 | 1121906.3432584647 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 8 | ok | 538.109336 | 0.051571 | 0.0558196 | 0.057892110000000004 | 1227351.0580724988 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 128 | ok | 738.996344 | 0.15748 | 0.1713633 | 0.19400205999999992 | 404664.721999762 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 1 | ok | 548.466527 | 0.0502915 | 0.057982099999999995 | 0.06125058 | 2509025.6316570034 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 2 | ok | 512.288092 | 0.0976555 | 0.10317475 | 0.1042015 | 1307315.8415835516 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 4 | ok | 521.137622 | 0.1061015 | 0.11386245 | 0.11676535 | 1209190.0712647506 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 8 | ok | 546.866444 | 0.0670395 | 0.07453325 | 0.07720595 | 1893663.682962826 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 128 | ok | 699.565988 | 4.833966 | 6.041398699999999 | 7.39892314 | 40117.43048885728 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 1 | ok | 533.517752 | 0.055524500000000004 | 0.06179079999999999 | 0.06720334 | 17713.524133968094 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 2 | ok | 507.285732 | 0.062835 | 0.06969639999999999 | 0.07233418 | 15684.094728167975 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 4 | ok | 507.470691 | 0.06663749999999999 | 0.07217314999999999 | 0.07383551000000001 | 14853.773512038091 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 8 | ok | 548.657999 | 0.0618065 | 0.06789215 | 0.07125626999999998 | 15880.70919435892 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 128 | ok | 862.038209 | 0.14268799999999998 | 0.17643484999999998 | 0.2037714 | 6804.100423078964 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 1 | ok | 511.525292 | 0.0598015 | 0.06629565 | 0.07170969999999999 | 32925.713006315156 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 2 | ok | 512.552879 | 0.07300000000000001 | 0.08107125 | 0.08387847 | 27132.970276102395 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 4 | ok | 524.62795 | 0.07430300000000001 | 0.0843651 | 0.08604905 | 26538.38365851257 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 8 | ok | 535.954218 | 0.0740595 | 0.08344114999999999 | 0.08856106 | 26590.143778225436 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 128 | ok | 828.973568 | 0.14459650000000002 | 0.17137845 | 0.18002605 | 13616.880356370099 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 1 | ok | 497.567335 | 0.0706185 | 0.0759469 | 0.07984514 | 56424.58825567335 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 2 | ok | 511.409958 | 0.072597 | 0.0805002 | 0.08343986 | 54352.90690216695 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 4 | ok | 508.988126 | 0.0653465 | 0.07243435 | 0.0741207 | 60457.04315484197 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 8 | ok | 542.777462 | 0.0765395 | 0.08432189999999999 | 0.08665071 | 51790.74261192109 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 128 | ok | 785.641376 | 0.151949 | 0.20163964999999992 | 0.26012221999999985 | 25185.671921615136 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 1 | ok | 539.829947 | 0.0696885 | 0.07522495 | 0.07717626 | 114087.46287166131 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 2 | ok | 505.709173 | 0.0853865 | 0.09082635 | 0.09568578999999998 | 93441.52616175849 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 4 | ok | 503.292221 | 0.0761615 | 0.0855318 | 0.08815089 | 102753.27397619205 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 8 | ok | 539.659942 | 0.0762565 | 0.0841154 | 0.08863822 | 103333.78025193293 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 128 | ok | 752.149763 | 0.156581 | 0.21022905 | 0.24613609999999989 | 49232.387082750894 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 1 | ok | 545.209377 | 0.0850825 | 0.0948002 | 0.10080826 | 185510.60169899886 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 2 | ok | 512.664637 | 0.07525799999999999 | 0.08347755 | 0.08645512 | 209571.66171990221 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 4 | ok | 502.35194 | 0.0781125 | 0.08625074999999999 | 0.09162084 | 203402.1035845553 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 8 | ok | 540.983916 | 0.0898 | 0.09737405 | 0.10030019 | 177001.80187834313 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 128 | ok | 728.46164 | 0.16138550000000002 | 0.24303819999999993 | 0.5257893699999997 | 87792.13105960493 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 1 | ok | 508.01017 | 0.07263 | 0.08167375 | 0.08850344999999998 | 430638.76378683274 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 2 | ok | 502.881778 | 0.0917975 | 0.09913759999999999 | 0.10510542999999999 | 345686.88632475643 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 4 | ok | 510.88683 | 0.08411099999999999 | 0.09266275 | 0.09492010999999999 | 376957.9727200227 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 8 | ok | 532.83651 | 0.09649350000000001 | 0.10765419999999999 | 0.1140052 | 328589.0550271661 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 128 | ok | 766.550325 | 0.17275000000000001 | 0.35261685 | 0.36095958 | 145554.63411385793 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 1 | ok | 499.469818 | 0.0831855 | 0.0924357 | 0.09580613999999998 | 757181.9297585987 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 2 | ok | 509.330917 | 0.1073125 | 0.1195769 | 0.12328408 | 590855.0411013538 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 4 | ok | 505.247473 | 0.10155800000000001 | 0.1082911 | 0.10933836 | 630483.2338746032 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 8 | ok | 552.245542 | 0.106919 | 0.11797704999999999 | 0.12298670999999999 | 594199.3516542324 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 128 | ok | 798.807714 | 0.2418215 | 0.29048820000000003 | 0.29806615000000003 | 256707.27989757378 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 1 | ok | 496.181412 | 0.088752 | 0.097575 | 0.0979095 | 1422162.80742049 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 2 | ok | 506.223051 | 0.1111935 | 0.1181715 | 0.12509972 | 1150361.402211462 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 4 | ok | 509.460072 | 0.118867 | 0.12983845 | 0.13213378 | 1075131.5356238114 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 8 | ok | 538.349232 | 0.116391 | 0.1237342 | 0.12523704 | 1101148.3945428461 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 128 | ok | 727.894454 | 0.2578895 | 0.27050514999999997 | 0.27145025 | 493665.38604208943 | - |
