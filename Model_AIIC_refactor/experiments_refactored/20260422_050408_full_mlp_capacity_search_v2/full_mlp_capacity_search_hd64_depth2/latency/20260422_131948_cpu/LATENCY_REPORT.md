# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd64_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1484576.755` samples/s, p50=`0.085` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.056` ms, throughput=`17649.247` samples/s

### full_mlp_capacity_search_hd64_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2161266.989` samples/s, p50=`0.059` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.033` ms, throughput=`29249.550` samples/s

### full_mlp_capacity_search_hd64_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2088498.833` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.044` ms, throughput=`22611.594` samples/s

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
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.055900000000000005 | 0.0600374 | 0.0614983 | 17649.247330198355 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0574135 | 0.06035065 | 0.0623762 | 17247.99302353178 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.05627 | 0.059244 | 0.06139868 | 17592.13617439296 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0567995 | 0.060385549999999996 | 0.06638923999999999 | 17386.496603547886 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.05764 | 0.062427399999999994 | 0.06884941999999998 | 17062.464659370075 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.068841 | 0.0728997 | 0.07900589999999998 | 28755.124522628987 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.067843 | 0.07089115 | 0.07641097999999999 | 29252.767311787695 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0727775 | 0.0766653 | 0.08240689999999998 | 27267.11024801618 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.06790650000000001 | 0.07076249999999999 | 0.07781028999999998 | 29210.564058913034 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.059237 | 0.06241794999999999 | 0.06581501 | 33493.10661626177 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.06875049999999999 | 0.07209205 | 0.07787110999999998 | 57761.03221275006 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0674115 | 0.0700445 | 0.07126156 | 59082.763726255376 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.068801 | 0.07191929999999999 | 0.07983405999999997 | 57571.91595881996 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.070033 | 0.0736018 | 0.07772905 | 56621.31016615806 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.0693825 | 0.0717785 | 0.07386656 | 57681.95267253466 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.070644 | 0.07390405 | 0.08124361999999999 | 112166.03264816712 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0721225 | 0.07439645 | 0.08206546999999997 | 110209.14113189747 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0688985 | 0.0712566 | 0.07849951999999998 | 115259.39270402282 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.06837399999999999 | 0.0717553 | 0.08063405999999997 | 115637.41656760394 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.0629715 | 0.06591235 | 0.06818205000000001 | 125952.51590150512 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.068981 | 0.07385349999999999 | 0.07776836999999999 | 228966.55943399467 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.074488 | 0.07830025 | 0.08340931999999998 | 213529.83772533073 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.07277549999999999 | 0.0764843 | 0.08102735999999998 | 218849.08905434486 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.07326 | 0.07624475 | 0.08393853999999998 | 216877.34041148683 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.0657375 | 0.0686782 | 0.07211177999999999 | 241905.61145423068 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.072497 | 0.07653979999999999 | 0.08529847999999997 | 436259.6082770445 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0712275 | 0.0745742 | 0.08318563999999996 | 444815.9893555534 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0758845 | 0.07910060000000001 | 0.08579844999999998 | 418877.44509891135 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0703855 | 0.0739049 | 0.08099920999999997 | 450919.6506274547 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.06335299999999999 | 0.0920165 | 0.09537416 | 468248.37064199767 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.079764 | 0.08268745 | 0.09172462999999997 | 795577.9786998914 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.094534 | 0.09910105 | 0.10526080999999998 | 673453.6766993026 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.09262999999999999 | 0.09644325 | 0.10503315999999997 | 686782.5268789511 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0773125 | 0.08184634999999998 | 0.09023760999999998 | 820940.7621973198 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.064029 | 0.06766604999999999 | 0.07071514 | 992832.6788394159 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.085367 | 0.08879115 | 0.10320411999999995 | 1484576.755285847 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.11655850000000001 | 0.1207178 | 0.12576547999999999 | 1093193.7416024785 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1169105 | 0.12477639999999998 | 0.13084742 | 1086565.668124136 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1051875 | 0.1117513 | 0.11229873 | 1210266.1583296664 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.120615 | 0.1257686 | 0.12895812 | 1056206.0024187118 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0694205 | 0.0722684 | 0.07422801 | 14355.420104191639 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0709195 | 0.0741676 | 0.07721449 | 14019.342767603319 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0726175 | 0.0759881 | 0.07946690999999999 | 13691.65302065249 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.072591 | 0.0749544 | 0.07995457999999998 | 13715.40050203852 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.0726755 | 0.07531099999999999 | 0.07908884999999999 | 13672.97364479638 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1171265 | 0.12207105 | 0.12788818999999998 | 16964.774511851338 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.11693100000000001 | 0.12334655 | 0.12595742999999998 | 16983.268762945496 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.12603799999999998 | 0.13804055 | 0.14034905 | 15605.490698269243 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.136202 | 0.1527903 | 0.15640651 | 14518.257361954238 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.217216 | 0.24066244999999997 | 0.24552293999999997 | 9158.237958680045 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.12023 | 0.1283837 | 0.14637414999999998 | 32803.75498022508 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.12470300000000001 | 0.1411335 | 0.14244557 | 31304.7245716418 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.129961 | 0.13710265 | 0.14507977 | 30729.504611192635 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1336725 | 0.14566115 | 0.15014265999999998 | 29931.583882141393 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.229055 | 0.24598275 | 0.25370708999999997 | 17554.515548034422 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.11533850000000001 | 0.1201965 | 0.12395167999999998 | 69074.3212066455 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.12833450000000002 | 0.14639385 | 0.15502392999999998 | 60702.61455266271 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1269475 | 0.1481788 | 0.16556697999999995 | 60962.91534415242 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.137272 | 0.14955675 | 0.15541228999999998 | 57832.11683475571 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.229961 | 0.2462991 | 0.24959381 | 35100.05974030167 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1177765 | 0.1265664 | 0.13056416999999998 | 134266.37273519902 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.128842 | 0.14872435 | 0.15043689 | 121230.72829208482 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1303395 | 0.15707725 | 0.16805914999999996 | 118370.879343231 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.141261 | 0.1967187 | 0.20558016999999998 | 107683.87728722239 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.2407935 | 0.24947125 | 0.2544977 | 66624.60433313102 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.127414 | 0.1349476 | 0.14343516999999997 | 249269.63995493206 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.142579 | 0.1507772 | 0.16186585999999997 | 222228.54956286948 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.14271 | 0.1675883 | 0.16978638 | 217116.07432426012 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.148851 | 0.21352079999999998 | 0.24913171 | 199605.67898117268 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.2446125 | 0.26115685 | 0.26528166999999997 | 130689.22885513623 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1356175 | 0.14920395 | 0.15724372 | 463734.0415711483 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1655925 | 0.18894805 | 0.19806634999999997 | 377223.5708619783 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.166262 | 0.1772372 | 0.1904537 | 381016.3826328447 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1720955 | 0.23824759999999998 | 0.25451887 | 352051.2762683885 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.24974999999999997 | 0.26684235 | 0.27233625 | 256502.88930465514 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1664005 | 0.1768768 | 0.18056711 | 763760.609261385 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.21264650000000002 | 0.24166899999999997 | 0.2580905399999999 | 590320.133375455 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.20826450000000002 | 0.24698035 | 0.25666024 | 601361.0491871055 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.21303499999999997 | 0.2229806 | 0.22811689999999998 | 600368.757747923 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.278041 | 0.8134292499999998 | 1.1714028599999995 | 344628.2737128215 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 1 | ok | 52.90934 | 0.038236 | 0.041874449999999994 | 0.04952420999999999 | 25628.986588863896 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 2 | ok | 53.147653 | 0.037877999999999995 | 0.04339234999999999 | 0.04927091999999999 | 25784.38683181051 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 4 | ok | 53.479538 | 0.038617 | 0.04322135 | 0.05103219 | 25410.86833002823 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 8 | ok | 53.763293 | 0.038555 | 0.04542144999999999 | 0.05029191 | 25221.736899955966 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 64 | ok | 51.48313 | 0.033492499999999994 | 0.03932099999999999 | 0.042462639999999996 | 29249.55014191882 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 1 | ok | 53.494565 | 0.041557 | 0.044129299999999996 | 0.04696855999999999 | 47752.57290862832 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 2 | ok | 53.542578 | 0.03931 | 0.04315215 | 0.04436579 | 51294.17775176592 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 4 | ok | 53.771984 | 0.0376615 | 0.0415458 | 0.041692690000000004 | 52114.819370036064 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 8 | ok | 53.951682 | 0.037096000000000004 | 0.0388568 | 0.040958379999999996 | 53574.34645994112 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 64 | ok | 51.762211 | 0.0375615 | 0.0390467 | 0.04280886999999999 | 52922.73756619973 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 1 | ok | 53.276133 | 0.041260000000000005 | 0.04318705 | 0.04696229999999999 | 96149.9631745641 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 2 | ok | 53.486517 | 0.0417395 | 0.0431166 | 0.04497391 | 96048.42377332956 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 4 | ok | 53.296078 | 0.0410915 | 0.04228805 | 0.04435848 | 98767.91958711059 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 8 | ok | 53.955501 | 0.041794 | 0.04393955 | 0.04791544 | 94557.81926974887 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 64 | ok | 51.917957 | 0.036429 | 0.03774595 | 0.04172233999999999 | 108847.68399340383 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 1 | ok | 53.694004 | 0.040084499999999995 | 0.041717199999999996 | 0.044914739999999995 | 198223.52080653186 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 2 | ok | 53.715827 | 0.042066 | 0.045454299999999996 | 0.052258159999999984 | 186943.57996020906 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 4 | ok | 53.869638 | 0.042535500000000004 | 0.04443945 | 0.04591912 | 186990.07222959018 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 8 | ok | 53.726107 | 0.041434 | 0.04323845 | 0.04551114 | 192067.79224795182 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 64 | ok | 52.491444 | 0.0367845 | 0.0390287 | 0.04058713 | 215231.6188161938 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 1 | ok | 53.609101 | 0.0429485 | 0.0440289 | 0.04582539 | 371084.3641747733 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 2 | ok | 53.771114 | 0.0434745 | 0.04504125 | 0.049251489999999995 | 365505.7594570046 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 4 | ok | 54.023174 | 0.0436245 | 0.0457116 | 0.04767452999999999 | 364210.1601887337 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 8 | ok | 53.584795 | 0.043193499999999996 | 0.0447079 | 0.04666185 | 368481.05201340345 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 64 | ok | 52.433613 | 0.039342 | 0.0412467 | 0.04256723 | 403880.4836872673 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 1 | ok | 54.385335 | 0.044506000000000004 | 0.0474749 | 0.04964884 | 709050.7671486144 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 2 | ok | 54.512055 | 0.046765 | 0.0495511 | 0.05352229999999999 | 678860.9900593536 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 4 | ok | 54.330598 | 0.0463425 | 0.04827935 | 0.05403198999999999 | 684072.3167049604 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 8 | ok | 54.120527 | 0.0442445 | 0.0462758 | 0.05136430999999999 | 715347.2384914042 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 64 | ok | 53.076778 | 0.040481500000000004 | 0.0420989 | 0.043851709999999995 | 785081.8742259584 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 1 | ok | 54.193549 | 0.048277 | 0.049370700000000003 | 0.052055489999999996 | 1321427.2735982134 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 2 | ok | 54.765457 | 0.0608575 | 0.0627988 | 0.06906797999999999 | 1050512.5844841918 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 4 | ok | 54.173289 | 0.056709999999999997 | 0.05941035 | 0.06246453 | 1122204.9644243494 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 8 | ok | 54.997893 | 0.0491365 | 0.05205915 | 0.060600719999999976 | 1294104.1826660405 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 64 | ok | 52.456062 | 0.0438725 | 0.04675715 | 0.049402839999999996 | 1445863.0469485288 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 1 | ok | 54.956025 | 0.058679999999999996 | 0.062098 | 0.06566039 | 2161266.9887404745 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 2 | ok | 54.931189 | 0.078748 | 0.08172565 | 0.08430836 | 1619643.440557947 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 4 | ok | 55.175812 | 0.07499249999999999 | 0.08611205 | 0.08895442999999999 | 1623555.35283663 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 8 | ok | 55.401845 | 0.074475 | 0.07819865 | 0.08164897 | 1709956.1663423984 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 64 | ok | 62.460009 | 0.0952845 | 0.100022 | 0.10179155999999999 | 1331580.6403155182 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 1 | ok | 53.436653 | 0.0447475 | 0.052688849999999995 | 0.05829245999999999 | 21760.19998494194 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 2 | ok | 52.878893 | 0.0438765 | 0.04866405 | 0.050832169999999996 | 22428.907093186728 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 4 | ok | 53.405835 | 0.0452235 | 0.04942325 | 0.05089704 | 21854.417236666293 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 8 | ok | 53.634059 | 0.0441305 | 0.049506299999999996 | 0.05036258 | 22399.52692199141 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 64 | ok | 52.246919 | 0.04571 | 0.051089999999999997 | 0.052815219999999996 | 21562.98894077423 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 1 | ok | 53.610029 | 0.0766505 | 0.08171229999999999 | 0.08423885999999998 | 25933.307313581663 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 2 | ok | 53.638906 | 0.0759695 | 0.07994425000000001 | 0.0830634 | 26164.98962165687 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 4 | ok | 53.693606 | 0.07924049999999999 | 0.0901134 | 0.09508369 | 24679.5179497836 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 8 | ok | 54.272828 | 0.09295300000000001 | 0.10421405 | 0.10899765 | 21307.03760798673 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 64 | ok | 52.053995 | 0.188369 | 0.20260455 | 0.20723457 | 10637.60982268381 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 1 | ok | 53.503653 | 0.0757375 | 0.08146885 | 0.08314865 | 52364.149722718736 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 2 | ok | 53.649699 | 0.084213 | 0.09964275 | 0.10124947999999999 | 45999.514245129576 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 4 | ok | 53.655427 | 0.0832195 | 0.09394969999999998 | 0.09809765000000001 | 47318.205229939274 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 8 | ok | 54.106676 | 0.0972625 | 0.10797065 | 0.10918734 | 40896.99376423087 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 64 | ok | 54.261925 | 0.1813415 | 0.1913061 | 0.19569284 | 22124.400414921005 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 1 | ok | 53.945347 | 0.0813975 | 0.0834083 | 0.08737221 | 98035.15491637724 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 2 | ok | 53.567719 | 0.0833515 | 0.09810305 | 0.10030712 | 93125.04365236421 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 4 | ok | 53.664291 | 0.084646 | 0.11209405 | 0.11762249 | 89435.99870496673 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 8 | ok | 53.408331 | 0.09634100000000001 | 0.1088291 | 0.11304 | 82226.37785199622 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 64 | ok | 53.915248 | 0.18051699999999998 | 0.19288324999999998 | 0.19894595999999998 | 44156.1586967928 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 1 | ok | 53.758351 | 0.079486 | 0.08605094999999999 | 0.08884471 | 198817.6315452007 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 2 | ok | 53.780527 | 0.091723 | 0.1075807 | 0.11357360999999998 | 169572.35970536803 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 4 | ok | 54.037015 | 0.08770649999999999 | 0.11567114999999999 | 0.12082588999999999 | 171932.19767818463 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 8 | ok | 53.452545 | 0.09898599999999999 | 0.17221329999999993 | 0.19270415 | 146256.0017521469 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 64 | ok | 52.517998 | 0.19188349999999998 | 0.203406 | 0.9956325399999969 | 71710.86251943029 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 1 | ok | 53.936794 | 0.0840245 | 0.09514829999999999 | 0.10109860999999999 | 374733.3245427141 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 2 | ok | 54.211797 | 0.10381499999999999 | 0.11003195 | 0.11387473999999999 | 305827.1736236296 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 4 | ok | 53.887027 | 0.09718850000000001 | 0.12318459999999999 | 0.13079764 | 311742.62373113446 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 8 | ok | 54.146873 | 0.10994799999999999 | 0.16839359999999998 | 0.17854372999999998 | 270260.9572237714 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 64 | ok | 53.783631 | 0.2005565 | 0.21293995 | 0.2926331799999997 | 158745.74983058852 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 1 | ok | 54.102904 | 0.096648 | 0.10403454999999999 | 0.10575359 | 657248.8385180682 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 2 | ok | 54.562285 | 0.126129 | 0.1460706 | 0.15476818 | 493480.2773667584 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 4 | ok | 54.58529 | 0.12321299999999999 | 0.13208435 | 0.13454563 | 515731.5862066298 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 8 | ok | 54.409151 | 0.1258235 | 0.19539645 | 0.21263457999999996 | 469479.98636160634 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 64 | ok | 52.604815 | 0.201979 | 0.22050385 | 0.22585464 | 320180.13334301877 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 1 | ok | 54.86299 | 0.1216935 | 0.12625225 | 0.13016474 | 1045415.2904073836 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 2 | ok | 55.05009 | 0.16574499999999998 | 0.1898741 | 0.19956998999999997 | 752066.1254140771 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 4 | ok | 55.696893 | 0.162197 | 0.19822725 | 0.20323982 | 762142.2962537609 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 8 | ok | 56.112855 | 0.16450700000000001 | 0.17841025 | 0.19297435999999998 | 775599.6354681713 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 64 | ok | 62.380282 | 0.2228595 | 0.8144558499999995 | 1.0449449699999995 | 407003.5129490712 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 1 | ok | 1372.854783 | 0.048771999999999996 | 0.054606749999999996 | 0.05994229999999998 | 20118.01225991667 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 2 | ok | 1361.991226 | 0.044711 | 0.04903245 | 0.05471669999999998 | 22054.509926734918 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 4 | ok | 1377.846378 | 0.0443695 | 0.04607875 | 0.04633914 | 22455.82266008082 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 8 | ok | 1440.363395 | 0.0441075 | 0.045769699999999996 | 0.04638676 | 22611.593868640204 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 64 | ok | 1542.22218 | 0.0445585 | 0.0485521 | 0.05004683 | 22156.795666485275 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 1 | ok | 1395.558176 | 0.045885 | 0.048682649999999994 | 0.057054189999999984 | 43038.79732384758 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 2 | ok | 1366.266054 | 0.049554 | 0.050903449999999996 | 0.055058589999999984 | 40212.95170705991 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 4 | ok | 1366.069087 | 0.046176999999999996 | 0.049469849999999996 | 0.05853408999999998 | 42752.92953761424 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 8 | ok | 1400.648276 | 0.0514575 | 0.0533203 | 0.057905589999999986 | 38663.705026861615 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 64 | ok | 1590.938068 | 0.0462435 | 0.04793665 | 0.050893379999999995 | 43027.556568328626 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 1 | ok | 1301.391561 | 0.053115499999999996 | 0.0554945 | 0.05856549999999999 | 74867.86757221849 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 2 | ok | 1330.014327 | 0.0483335 | 0.05320075 | 0.057296569999999984 | 81292.22114735842 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 4 | ok | 1356.883777 | 0.0475655 | 0.04897505 | 0.052795059999999984 | 83925.42722238727 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 8 | ok | 1406.282472 | 0.045835 | 0.04762385 | 0.04904227 | 86716.99431960328 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 64 | ok | 1604.295665 | 0.0466715 | 0.04941395 | 0.05655105999999998 | 84930.19799352407 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 1 | ok | 1360.45486 | 0.048509 | 0.0501755 | 0.05119963 | 164699.06187414358 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 2 | ok | 1307.554649 | 0.0511595 | 0.05348255 | 0.05626975999999999 | 155487.56429896684 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 4 | ok | 1312.591355 | 0.050218 | 0.05275605 | 0.05667878999999999 | 158110.70358077265 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 8 | ok | 1369.263355 | 0.0476285 | 0.0500382 | 0.055123629999999986 | 166673.88920186542 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 64 | ok | 1625.224986 | 0.047175999999999996 | 0.0494374 | 0.05199345 | 168387.58781412704 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 1 | ok | 1387.242107 | 0.050179 | 0.0526233 | 0.05595950999999999 | 316698.23092368204 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 2 | ok | 1366.800843 | 0.0466115 | 0.049221499999999994 | 0.05532657999999998 | 340434.8033308141 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 4 | ok | 1355.319314 | 0.046669 | 0.0488098 | 0.04967609 | 340444.5098854447 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 8 | ok | 1396.548027 | 0.047644500000000006 | 0.04916185 | 0.05002585 | 335174.1376283664 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 64 | ok | 1579.948622 | 0.0473345 | 0.05010585 | 0.05352907999999999 | 336049.3118760247 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 1 | ok | 1388.953831 | 0.048550499999999996 | 0.05027585 | 0.05144853 | 657678.6656028714 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 2 | ok | 1367.615505 | 0.0540075 | 0.0560311 | 0.05920915999999999 | 589775.2145491648 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 4 | ok | 1334.906195 | 0.054326 | 0.0561894 | 0.06133023999999999 | 585467.2394487825 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 8 | ok | 1385.5104 | 0.048505 | 0.0503853 | 0.05180492 | 657147.3398264967 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 64 | ok | 1552.298652 | 0.049474000000000004 | 0.050871 | 0.051294809999999996 | 646102.5679750378 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 1 | ok | 1362.687675 | 0.051954 | 0.0535212 | 0.05482897 | 1227529.9680343524 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 2 | ok | 1372.651331 | 0.0809135 | 0.0827079 | 0.0851672 | 788223.3519542889 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 4 | ok | 1391.791215 | 0.06694449999999999 | 0.0687802 | 0.07600134999999997 | 952159.3486277895 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 8 | ok | 1346.919326 | 0.0533995 | 0.05517335 | 0.05657554 | 1194686.0365096051 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 64 | ok | 1633.141188 | 0.057254 | 0.05943145 | 0.062329869999999996 | 1112530.7466993474 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 1 | ok | 1365.827013 | 0.060837 | 0.06395805 | 0.07047945999999998 | 2088498.832724949 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 2 | ok | 1371.23047 | 0.10042799999999999 | 0.1029886 | 0.10763731999999998 | 1268923.8177793676 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 4 | ok | 1357.756051 | 0.1108735 | 0.11439225 | 0.12428389999999997 | 1149033.6088739866 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 8 | ok | 1373.801826 | 0.0768895 | 0.07974315 | 0.08438922999999998 | 1658840.7398844408 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 64 | ok | 1539.003382 | 0.09693750000000001 | 0.10193785 | 0.10710576999999999 | 1309412.2805276685 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 1 | ok | 1377.462649 | 0.0531165 | 0.055453199999999994 | 0.05632535 | 18712.97404173667 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 2 | ok | 1369.915882 | 0.053225 | 0.05518145 | 0.0553181 | 18756.524926108672 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 4 | ok | 1355.87398 | 0.0541725 | 0.05626415 | 0.06329508999999997 | 18311.550523215934 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 8 | ok | 1357.917563 | 0.057291 | 0.05931955 | 0.06321473 | 17328.559892701556 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 64 | ok | 1609.725667 | 0.0571435 | 0.0591779 | 0.06602152999999998 | 17373.01886779341 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 1 | ok | 1383.384908 | 0.092159 | 0.09664655 | 0.10370362 | 21579.843045485562 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 2 | ok | 1316.527658 | 0.093948 | 0.1042255 | 0.10843889999999999 | 21046.623321847488 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 4 | ok | 1322.203661 | 0.10043550000000001 | 0.11208719999999998 | 0.11827683 | 19547.25813542223 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 8 | ok | 1409.047262 | 0.107684 | 0.1164949 | 0.12114145999999999 | 18436.618960071446 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 64 | ok | 1592.284494 | 0.20970149999999999 | 0.22204935 | 0.23461618 | 9476.263050827927 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 1 | ok | 1336.290009 | 0.095579 | 0.09888839999999999 | 0.10635225 | 41581.1999752176 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 2 | ok | 1336.025778 | 0.09676 | 0.1174499 | 0.12177027 | 39986.092836911324 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 4 | ok | 1415.779496 | 0.09751950000000001 | 0.10560834999999999 | 0.10944411 | 40648.811947011025 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 8 | ok | 1411.6177 | 0.120666 | 0.12825005 | 0.13002608 | 33306.85993083164 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 64 | ok | 1582.468479 | 0.214437 | 0.22492004999999998 | 0.22975552999999999 | 18578.924105652397 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 1 | ok | 1342.235677 | 0.09112100000000001 | 0.09441754999999999 | 0.10018678999999998 | 87318.43768104933 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 2 | ok | 1378.858183 | 0.10734350000000001 | 0.12521179999999998 | 0.13098421999999998 | 72293.49859145154 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 4 | ok | 1382.047955 | 0.10187550000000001 | 0.12424985 | 0.12924671 | 74761.85545468105 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 8 | ok | 1463.491456 | 0.111192 | 0.1211664 | 0.12396864999999999 | 72083.9302032929 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 64 | ok | 1548.506845 | 0.212145 | 0.22648944999999998 | 0.23470657999999997 | 37926.591186429105 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 1 | ok | 1369.869548 | 0.09552150000000001 | 0.0993257 | 0.10380339999999999 | 166783.6236827742 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 2 | ok | 1358.672459 | 0.1132675 | 0.1277848 | 0.12881787 | 138391.6400378086 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 4 | ok | 1316.831389 | 0.1142685 | 0.1402769 | 0.14585136 | 134057.06272932136 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 8 | ok | 1381.91522 | 0.119756 | 0.18192395 | 0.19685213999999995 | 124585.3837362191 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 64 | ok | 1637.593463 | 0.2167165 | 0.23427984999999998 | 0.238958 | 74925.55443739572 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 1 | ok | 1383.339956 | 0.1022455 | 0.10972574999999998 | 0.11278847 | 311030.7442227011 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 2 | ok | 1336.198504 | 0.12001999999999999 | 0.12590029999999997 | 0.13052457 | 266018.7337042743 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 4 | ok | 1397.199051 | 0.12398300000000001 | 0.1464198 | 0.15325303999999998 | 249245.7977937385 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 8 | ok | 1405.533633 | 0.124997 | 0.17620844999999993 | 0.20159416 | 243047.51199097687 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 64 | ok | 1554.794826 | 0.215581 | 0.22830785 | 0.23404579 | 147951.0034360696 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 1 | ok | 1324.622905 | 0.11804100000000001 | 0.12222475000000001 | 0.12901467 | 539489.0836069835 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 2 | ok | 1338.063742 | 0.142405 | 0.16197119999999998 | 0.16408122 | 439765.3412139283 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 4 | ok | 1380.297594 | 0.13479400000000002 | 0.144944 | 0.14910004 | 471076.8389625297 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 8 | ok | 1510.948657 | 0.14684049999999998 | 0.20339235 | 0.22677860999999996 | 414244.14747454115 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 64 | ok | 1578.564992 | 0.219691 | 0.23545539999999998 | 0.23836865 | 288656.8328500135 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 1 | ok | 1407.25514 | 0.1470105 | 0.15448335 | 0.15844019 | 867239.1077139428 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 2 | ok | 1382.453036 | 0.1868515 | 0.20861275 | 0.21265478 | 673511.7915076899 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 4 | ok | 1415.65548 | 0.1587385 | 0.1892998 | 0.19382505 | 785337.3588417452 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 8 | ok | 1430.44334 | 0.1700345 | 0.18529305 | 0.1893978 | 762808.0537751073 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 64 | ok | 1603.554158 | 0.242769 | 0.8877400999999998 | 1.2505262499999996 | 363031.91549706104 | - |
