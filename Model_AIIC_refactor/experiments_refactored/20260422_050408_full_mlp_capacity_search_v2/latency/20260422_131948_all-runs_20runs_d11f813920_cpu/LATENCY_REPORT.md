# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd128_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1499365.581` samples/s, p50=`0.084` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.056` ms, throughput=`17817.848` samples/s

### full_mlp_capacity_search_hd128_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2376399.384` samples/s, p50=`0.053` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.035` ms, throughput=`27702.443` samples/s

### full_mlp_capacity_search_hd128_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2144726.118` samples/s, p50=`0.060` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.044` ms, throughput=`22487.191` samples/s

### full_mlp_capacity_search_hd128_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`829699.401` samples/s, p50=`0.153` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.072` ms, throughput=`13898.718` samples/s

### full_mlp_capacity_search_hd128_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1381366.914` samples/s, p50=`0.093` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.041` ms, throughput=`24190.431` samples/s

### full_mlp_capacity_search_hd128_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1182789.161` samples/s, p50=`0.107` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.058` ms, throughput=`17146.712` samples/s

### full_mlp_capacity_search_hd128_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`588677.506` samples/s, p50=`0.216` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.082` ms, throughput=`12074.194` samples/s

### full_mlp_capacity_search_hd128_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1069971.452` samples/s, p50=`0.119` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.045` ms, throughput=`21730.825` samples/s

### full_mlp_capacity_search_hd128_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`778022.706` samples/s, p50=`0.164` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.073` ms, throughput=`13719.954` samples/s

### full_mlp_capacity_search_hd128_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`455899.651` samples/s, p50=`0.280` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.096` ms, throughput=`10344.133` samples/s

### full_mlp_capacity_search_hd128_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`863968.439` samples/s, p50=`0.141` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.047` ms, throughput=`20873.707` samples/s

### full_mlp_capacity_search_hd128_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`593344.657` samples/s, p50=`0.215` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.078` ms, throughput=`12702.803` samples/s

### full_mlp_capacity_search_hd256_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1438477.659` samples/s, p50=`0.088` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.055` ms, throughput=`17801.139` samples/s

### full_mlp_capacity_search_hd256_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2232192.683` samples/s, p50=`0.057` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.034` ms, throughput=`28471.696` samples/s

### full_mlp_capacity_search_hd256_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2118346.736` samples/s, p50=`0.060` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.044` ms, throughput=`22413.263` samples/s

### full_mlp_capacity_search_hd256_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`590264.164` samples/s, p50=`0.215` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.076` ms, throughput=`13039.761` samples/s

### full_mlp_capacity_search_hd256_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`959537.935` samples/s, p50=`0.133` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.041` ms, throughput=`24183.785` samples/s

### full_mlp_capacity_search_hd256_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`740064.490` samples/s, p50=`0.173` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.059` ms, throughput=`16808.209` samples/s

### full_mlp_capacity_search_hd256_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`398080.877` samples/s, p50=`0.331` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.096` ms, throughput=`10392.277` samples/s

### full_mlp_capacity_search_hd256_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`530134.458` samples/s, p50=`0.241` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.051` ms, throughput=`19515.097` samples/s

### full_mlp_capacity_search_hd256_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`435164.501` samples/s, p50=`0.288` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.075` ms, throughput=`13244.016` samples/s

### full_mlp_capacity_search_hd256_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`334041.937` samples/s, p50=`0.391` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.111` ms, throughput=`8925.307` samples/s

### full_mlp_capacity_search_hd256_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`436732.685` samples/s, p50=`0.276` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.064` ms, throughput=`15496.462` samples/s

### full_mlp_capacity_search_hd256_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`356939.708` samples/s, p50=`0.360` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.095` ms, throughput=`10477.568` samples/s

### full_mlp_capacity_search_hd32_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1512361.666` samples/s, p50=`0.084` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.056` ms, throughput=`17651.659` samples/s

### full_mlp_capacity_search_hd32_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2180938.124` samples/s, p50=`0.058` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.034` ms, throughput=`28751.412` samples/s

### full_mlp_capacity_search_hd32_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2100533.503` samples/s, p50=`0.060` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.044` ms, throughput=`22725.052` samples/s

### full_mlp_capacity_search_hd32_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1133097.130` samples/s, p50=`0.112` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.069` ms, throughput=`14233.635` samples/s

### full_mlp_capacity_search_hd32_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2010019.949` samples/s, p50=`0.064` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.039` ms, throughput=`25229.462` samples/s

### full_mlp_capacity_search_hd32_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1549461.598` samples/s, p50=`0.082` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.054` ms, throughput=`18237.545` samples/s

### full_mlp_capacity_search_hd32_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`997599.371` samples/s, p50=`0.128` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.078` ms, throughput=`12640.985` samples/s

### full_mlp_capacity_search_hd32_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1817519.869` samples/s, p50=`0.070` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.041` ms, throughput=`24004.935` samples/s

### full_mlp_capacity_search_hd32_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1426545.288` samples/s, p50=`0.089` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.065` ms, throughput=`15199.384` samples/s

### full_mlp_capacity_search_hd32_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`853982.132` samples/s, p50=`0.149` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.090` ms, throughput=`10977.185` samples/s

### full_mlp_capacity_search_hd32_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1689489.737` samples/s, p50=`0.075` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.045` ms, throughput=`21684.561` samples/s

### full_mlp_capacity_search_hd32_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1249736.628` samples/s, p50=`0.102` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.072` ms, throughput=`13705.757` samples/s

### full_mlp_capacity_search_hd512_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1474861.674` samples/s, p50=`0.086` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.055` ms, throughput=`17836.796` samples/s

### full_mlp_capacity_search_hd512_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2189041.521` samples/s, p50=`0.058` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.036` ms, throughput=`27028.970` samples/s

### full_mlp_capacity_search_hd512_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2156090.146` samples/s, p50=`0.059` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.043` ms, throughput=`22864.773` samples/s

### full_mlp_capacity_search_hd512_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`467789.820` samples/s, p50=`0.278` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.081` ms, throughput=`11918.911` samples/s

### full_mlp_capacity_search_hd512_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`588636.573` samples/s, p50=`0.226` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.043` ms, throughput=`22620.964` samples/s

### full_mlp_capacity_search_hd512_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`503296.791` samples/s, p50=`0.233` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.064` ms, throughput=`15565.452` samples/s

### full_mlp_capacity_search_hd512_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`313901.675` samples/s, p50=`0.415` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.106` ms, throughput=`9345.997` samples/s

### full_mlp_capacity_search_hd512_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`346368.478` samples/s, p50=`0.348` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.070` ms, throughput=`14091.906` samples/s

### full_mlp_capacity_search_hd512_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`331190.016` samples/s, p50=`0.414` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.093` ms, throughput=`10680.000` samples/s

### full_mlp_capacity_search_hd512_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`256275.978` samples/s, p50=`0.512` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.138` ms, throughput=`7042.828` samples/s

### full_mlp_capacity_search_hd512_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`287534.107` samples/s, p50=`0.442` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.087` ms, throughput=`11402.064` samples/s

### full_mlp_capacity_search_hd512_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`258354.457` samples/s, p50=`0.480` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.120` ms, throughput=`8207.418` samples/s

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

### full_mlp_capacity_search_hd64_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1035385.430` samples/s, p50=`0.122` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.069` ms, throughput=`14333.321` samples/s

### full_mlp_capacity_search_hd64_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1658864.818` samples/s, p50=`0.077` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.039` ms, throughput=`25115.960` samples/s

### full_mlp_capacity_search_hd64_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1469563.053` samples/s, p50=`0.087` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.055` ms, throughput=`18170.024` samples/s

### full_mlp_capacity_search_hd64_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`838156.663` samples/s, p50=`0.151` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.080` ms, throughput=`12386.460` samples/s

### full_mlp_capacity_search_hd64_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1512666.216` samples/s, p50=`0.084` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.042` ms, throughput=`23611.301` samples/s

### full_mlp_capacity_search_hd64_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1118263.337` samples/s, p50=`0.114` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.067` ms, throughput=`14839.543` samples/s

### full_mlp_capacity_search_hd64_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`686385.595` samples/s, p50=`0.185` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.090` ms, throughput=`10999.133` samples/s

### full_mlp_capacity_search_hd64_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1302336.901` samples/s, p50=`0.098` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.045` ms, throughput=`21823.035` samples/s

### full_mlp_capacity_search_hd64_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`971902.895` samples/s, p50=`0.131` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.074` ms, throughput=`13426.001` samples/s

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

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.059389 | 0.0634818 | 0.06391499 | 16728.741449940244 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.057356500000000005 | 0.06151585 | 0.06853368999999998 | 17225.697985282364 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.056963 | 0.06075585 | 0.0642501 | 17391.89718466925 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.055594000000000005 | 0.05902285 | 0.06317376999999999 | 17817.848138480316 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.059024 | 0.061835 | 0.06669439 | 16796.881692507548 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0716105 | 0.0747447 | 0.07836283999999999 | 27806.98283391529 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.07082250000000001 | 0.0747935 | 0.08751300999999997 | 27888.487325658607 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.07130149999999999 | 0.0750692 | 0.07855144999999998 | 27884.256911043365 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.08324899999999999 | 0.08799309999999999 | 0.09104841999999999 | 23834.70915219463 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.067 | 0.0704243 | 0.0748764 | 29637.392431084172 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.06769900000000001 | 0.07081950000000001 | 0.07863144999999998 | 58577.28086019565 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.06801 | 0.07144485 | 0.07550865 | 58322.3857355109 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.067277 | 0.070335 | 0.07543227 | 59001.2446312555 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0725855 | 0.07579935 | 0.07969614 | 54681.90578471679 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.06995599999999999 | 0.0734876 | 0.08464861999999997 | 56475.90893033782 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0681415 | 0.07193845 | 0.07704715999999999 | 116435.54695743727 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.072019 | 0.0750156 | 0.08277574999999997 | 110030.51971540606 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.072508 | 0.07527640000000001 | 0.07972944999999998 | 109776.32250966238 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.068328 | 0.07133864999999999 | 0.07452737 | 116236.08944599556 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.0684305 | 0.0718655 | 0.07626156999999999 | 116091.09668356761 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.075134 | 0.0783556 | 0.08272545999999999 | 211786.22087079083 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.06887099999999999 | 0.07394179999999999 | 0.07792742999999999 | 229727.2849967982 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.06980700000000001 | 0.07326965 | 0.07641500999999999 | 227381.5086535718 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.069548 | 0.07276505 | 0.08056691999999997 | 227944.58097375074 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.07130600000000001 | 0.0735659 | 0.07777197 | 223473.98692956517 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.07088549999999999 | 0.0739873 | 0.07635971999999999 | 448612.0503925917 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.071821 | 0.074202 | 0.08155747999999997 | 442553.87886028 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.074859 | 0.0783843 | 0.08150412 | 424886.44909647893 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0717555 | 0.0742324 | 0.07550431 | 444297.0859109507 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.07397200000000001 | 0.07696895 | 0.07971208999999999 | 430151.4670853474 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.07896349999999999 | 0.0831227 | 0.08894761999999999 | 804448.3985317812 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.09469 | 0.0993096 | 0.10683218 | 671470.4783807489 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0927585 | 0.09634855 | 0.10032956999999999 | 686387.8773605039 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.07602 | 0.0798208 | 0.08519935999999999 | 834655.0879139596 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.073896 | 0.0779834 | 0.08242855999999998 | 859255.7824557691 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.084383 | 0.09126 | 0.09625505999999999 | 1499365.5809385653 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1197555 | 0.1234503 | 0.13289939999999997 | 1062918.2853334383 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.117028 | 0.12296829999999999 | 0.12615653999999998 | 1090430.230951419 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.10891400000000001 | 0.1125236 | 0.11852279999999998 | 1169903.8576665719 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.154226 | 0.15913744999999999 | 0.16305540999999998 | 829767.7091228552 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.073327 | 0.0756507 | 0.08120635999999999 | 13548.751252243334 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.07339599999999999 | 0.07639575 | 0.08066300999999998 | 13548.821008693467 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.068834 | 0.0723693 | 0.07406207000000001 | 14440.274734890998 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.071283 | 0.07444775 | 0.07891246999999998 | 13945.655454173739 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.073631 | 0.0763569 | 0.07943673 | 13512.899954812863 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1121305 | 0.12197969999999998 | 0.13396162999999997 | 17571.514747157322 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.12001200000000001 | 0.12471259999999999 | 0.12771781 | 16584.12196361311 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.12558350000000001 | 0.13887839999999999 | 0.14534185000000002 | 15637.077198842795 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.130206 | 0.1455597 | 0.15498594999999998 | 15136.914909043791 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.2284225 | 0.24526465 | 0.25496433 | 8687.045955254936 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.11213100000000001 | 0.11809389999999999 | 0.11978398 | 35459.83606208592 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.11968200000000001 | 0.13656669999999999 | 0.13861499 | 32608.176402408375 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.122027 | 0.1319394 | 0.13294343 | 32356.647915989197 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1397445 | 0.1546488 | 0.15660826 | 28605.77184379647 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.2346125 | 0.2482548 | 0.25076252 | 17106.3043916331 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1167265 | 0.12522525 | 0.13044493 | 67787.74031601289 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.127073 | 0.14351609999999998 | 0.14888778 | 61599.95724962966 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.127304 | 0.15946335 | 0.17004425999999995 | 60345.29578246727 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.13993899999999998 | 0.1521757 | 0.16118659 | 56753.94666945139 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.3516415 | 0.55814705 | 2.159508659999994 | 18111.50438230488 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1170745 | 0.13024075 | 0.13243828 | 134796.0047812143 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.130952 | 0.1460978 | 0.14856911 | 119494.26442467545 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.13611499999999999 | 0.16820714999999997 | 0.19016270999999996 | 113202.9603140137 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1437195 | 0.20722529999999997 | 0.21636536 | 105549.37725207736 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.2271815 | 0.2477999 | 0.25358222999999996 | 70402.20781323702 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1269105 | 0.1324101 | 0.13972443 | 250334.97949443595 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.145743 | 0.15096665 | 0.15621248 | 218467.94167288276 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1393075 | 0.1698915 | 0.175504 | 221659.11571035304 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1493835 | 0.2167798 | 0.22475978 | 202058.16446322238 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.24415599999999998 | 0.2652806 | 0.26778471 | 130425.59256626293 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1385185 | 0.143862 | 0.15018734999999997 | 459643.71292147535 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.17327199999999998 | 0.19263175000000002 | 0.19446579 | 365353.44177784637 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.165943 | 0.174068 | 0.17902536 | 386829.33162419236 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1729155 | 0.23129104999999997 | 0.24119909999999997 | 354549.0695358255 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.26027199999999995 | 0.2811846 | 0.28241416 | 245976.83743736715 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1579715 | 0.16626 | 0.16858766 | 808769.4875535431 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.20812599999999998 | 0.23959095 | 0.24642792 | 613102.6943181092 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2025595 | 0.23498734999999998 | 0.24317121 | 634397.4458365834 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.20778000000000002 | 0.21948445 | 0.22929493999999997 | 611394.1515755007 | - |
| `full_mlp_capacity_search_hd128_depth2` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.294374 | 0.84045695 | 1.01427283 | 324805.26655499457 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 1 | ok | 53.538027 | 0.0379285 | 0.04382264999999999 | 0.050370259999999986 | 25768.706276947625 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 2 | ok | 53.108935 | 0.0352965 | 0.04171295 | 0.046862169999999995 | 27608.21730980009 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 4 | ok | 53.023114 | 0.0352945 | 0.040895249999999994 | 0.04599614 | 27702.442524357368 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 8 | ok | 53.427543 | 0.035460500000000006 | 0.04313715 | 0.04691471999999999 | 27397.770698193828 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 1 | 64 | ok | 52.835433 | 0.038112999999999994 | 0.04464034999999998 | 0.04968896999999999 | 25669.999830578 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 1 | ok | 53.728342 | 0.042575 | 0.045008849999999996 | 0.04718048 | 47166.91900968337 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 2 | ok | 53.851224 | 0.036699999999999997 | 0.038501049999999995 | 0.043047649999999986 | 53918.23838331554 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 4 | ok | 53.595899 | 0.0383005 | 0.03898 | 0.04060349 | 52147.18647677442 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 8 | ok | 53.607034 | 0.037486 | 0.0387667 | 0.042540869999999995 | 52950.956823789806 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 2 | 64 | ok | 53.905377 | 0.036764 | 0.0442382 | 0.046330149999999994 | 52709.08904989759 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 1 | ok | 53.885891 | 0.037363 | 0.044119349999999995 | 0.04593436 | 101236.14393552474 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 2 | ok | 53.949043 | 0.0398425 | 0.04164225 | 0.04652404999999999 | 99336.97535797322 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 4 | ok | 53.844857 | 0.0382645 | 0.04021825 | 0.0427766 | 103922.71890931028 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 8 | ok | 53.58288 | 0.039163500000000004 | 0.04005465 | 0.04154244 | 101737.47255631677 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 4 | 64 | ok | 53.48443 | 0.041181999999999996 | 0.0432245 | 0.049098679999999985 | 96078.5999810725 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 1 | ok | 53.696718 | 0.040059 | 0.04218525 | 0.04350574 | 198217.62709714248 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 2 | ok | 53.591998 | 0.0367525 | 0.0388877 | 0.041671139999999995 | 215861.50325950873 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 4 | ok | 53.857489 | 0.0373575 | 0.0388999 | 0.04080316 | 212722.84712513047 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 8 | ok | 53.743076 | 0.038692 | 0.0414437 | 0.04365811 | 204480.57843466027 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 8 | 64 | ok | 53.595935 | 0.039342 | 0.040903300000000004 | 0.04233003999999999 | 202203.71721203538 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 1 | ok | 53.970991 | 0.040135500000000005 | 0.0414221 | 0.04599320999999999 | 395180.18487517 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 2 | ok | 53.813827 | 0.039583999999999994 | 0.0442595 | 0.046218199999999994 | 389801.62021043437 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 4 | ok | 53.600557 | 0.041004 | 0.04268945 | 0.04645057999999999 | 387104.0167848302 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 8 | ok | 53.494346 | 0.03916 | 0.040722999999999995 | 0.04443390999999999 | 406273.05917011615 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 16 | 64 | ok | 53.598442 | 0.042642 | 0.0440588 | 0.045718709999999996 | 373246.0932797974 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 1 | ok | 54.383593 | 0.0442 | 0.0459133 | 0.04797177 | 724250.6269294489 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 2 | ok | 53.671019 | 0.0426255 | 0.0464386 | 0.048139839999999996 | 732278.5166417159 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 4 | ok | 54.066518 | 0.0414525 | 0.0459823 | 0.0483758 | 758735.7754824967 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 8 | ok | 54.522041 | 0.0432375 | 0.04509585 | 0.046255239999999996 | 735092.4401715155 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 32 | 64 | ok | 53.925468 | 0.044783500000000004 | 0.045867649999999996 | 0.0478474 | 712566.0125607573 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 1 | ok | 53.978933 | 0.047823000000000004 | 0.05042939999999999 | 0.05554187999999998 | 1329194.5537914652 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 2 | ok | 54.674975 | 0.061395500000000006 | 0.06368245 | 0.06933272999999998 | 1037581.8635878154 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 4 | ok | 54.897056 | 0.0560725 | 0.0596241 | 0.0625028 | 1144729.1839046783 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 8 | ok | 54.035301 | 0.0438805 | 0.04581635 | 0.04999496999999999 | 1447431.9842664143 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 64 | 64 | ok | 55.160439 | 0.042825 | 0.04388305 | 0.04596093 | 1487548.0605859733 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 1 | ok | 55.191551 | 0.053378 | 0.0560477 | 0.06216796999999998 | 2376399.3836214095 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 2 | ok | 55.365887 | 0.085757 | 0.0902924 | 0.09215902 | 1480462.6353206462 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 4 | ok | 55.205375 | 0.08210300000000001 | 0.08770635 | 0.08872363 | 1588815.5330550591 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 8 | ok | 55.532544 | 0.0660145 | 0.06937404999999999 | 0.07298503 | 1925811.714592297 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `fp32` | 128 | 64 | ok | 63.609731 | 0.1214405 | 0.13159185 | 0.13476098 | 1042674.7213288889 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 1 | ok | 53.338286 | 0.044161 | 0.0496979 | 0.05078367 | 22325.50496943415 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 2 | ok | 53.204 | 0.0435855 | 0.048538149999999995 | 0.05004291 | 22636.76681775954 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 4 | ok | 53.254486 | 0.042795 | 0.047528299999999996 | 0.04908634 | 23039.414909450494 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 8 | ok | 53.158511 | 0.0431895 | 0.04661435 | 0.057076199999999994 | 22726.27070533708 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 1 | 64 | ok | 53.247231 | 0.0446015 | 0.05016945 | 0.05150054 | 22048.392694220984 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 1 | ok | 53.769395 | 0.0768205 | 0.08267455 | 0.08392507999999999 | 25828.84772344536 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 2 | ok | 53.320501 | 0.07498350000000001 | 0.0777046 | 0.08108220999999999 | 26505.590426604827 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 4 | ok | 53.320725 | 0.07710449999999999 | 0.08050399999999999 | 0.08607811999999998 | 25799.14795733956 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 8 | ok | 53.66991 | 0.09492300000000001 | 0.1057181 | 0.10859864999999999 | 20906.152074277048 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 2 | 64 | ok | 53.411955 | 0.1989415 | 0.2594312 | 0.3085728899999999 | 9228.58191487722 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 1 | ok | 53.69866 | 0.076958 | 0.08170175 | 0.0837744 | 51676.034670485176 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 2 | ok | 54.025539 | 0.0849335 | 0.1016884 | 0.10553622 | 45691.70617856959 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 4 | ok | 54.035446 | 0.08474799999999999 | 0.09539529999999999 | 0.09812425 | 46657.6337487729 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 8 | ok | 53.275366 | 0.091211 | 0.10132065 | 0.10209781999999999 | 44121.90352481062 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 4 | 64 | ok | 53.732708 | 0.1764715 | 0.19534985 | 0.2317363499999999 | 22282.06938480432 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 1 | ok | 53.785479 | 0.08014199999999999 | 0.08591449999999999 | 0.08971276999999998 | 98527.16664934231 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 2 | ok | 53.832403 | 0.08839 | 0.10693255 | 0.11107816 | 87581.35760738241 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 4 | ok | 53.625457 | 0.0844765 | 0.11173585 | 0.11544346 | 89784.055878005 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 8 | ok | 53.506826 | 0.09177099999999999 | 0.10248039999999999 | 0.10878829999999999 | 87041.08404447713 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 8 | 64 | ok | 54.240315 | 0.190747 | 0.20560079999999997 | 0.22048865999999998 | 41748.188702647894 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 1 | ok | 53.878328 | 0.0783905 | 0.08514719999999999 | 0.08707019 | 202155.43175219992 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 2 | ok | 53.76687 | 0.093098 | 0.1111875 | 0.11653671999999998 | 166508.13704452218 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 4 | ok | 53.932469 | 0.0880325 | 0.11777694999999999 | 0.12385186 | 170429.10213262198 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 8 | ok | 53.733962 | 0.099366 | 0.1752007 | 0.1809298 | 145632.85302404797 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 16 | 64 | ok | 53.600715 | 0.2129275 | 0.22235164999999998 | 0.22503256 | 75362.45809729576 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 1 | ok | 53.963658 | 0.08773449999999999 | 0.09449024999999998 | 0.09894 | 361429.2629847416 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 2 | ok | 53.554157 | 0.101142 | 0.10812155 | 0.11087801 | 313922.0916001119 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 4 | ok | 53.92709 | 0.104852 | 0.13056964999999998 | 0.13389355 | 292663.9036755294 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 8 | ok | 54.235921 | 0.1100145 | 0.18625835 | 0.20185220999999998 | 262289.7501755702 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 32 | 64 | ok | 54.655399 | 0.19545649999999998 | 0.21230155 | 0.21354379999999998 | 163491.7421853757 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 1 | ok | 54.138227 | 0.09489800000000001 | 0.1027394 | 0.10403907999999999 | 667034.2303206663 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 2 | ok | 54.315886 | 0.1255095 | 0.14533795 | 0.14744952 | 500867.98857207084 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 4 | ok | 54.526341 | 0.119476 | 0.1290528 | 0.13636736 | 532124.8777359949 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 8 | ok | 54.726099 | 0.1268325 | 0.19223215 | 0.20903289 | 467548.6192120403 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 64 | 64 | ok | 54.345712 | 0.20551999999999998 | 0.219132 | 0.22766656 | 311275.05362928717 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 1 | ok | 55.351044 | 0.126104 | 0.1324527 | 0.14709888999999995 | 1008031.6496737207 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 2 | ok | 55.409975 | 0.174551 | 0.19649244999999999 | 0.19943908999999999 | 717377.7243680098 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 4 | ok | 55.218621 | 0.1598255 | 0.1987445 | 0.20110928 | 769820.9035724741 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 8 | ok | 55.524794 | 0.1644965 | 0.17215675 | 0.17490644 | 781730.5658116979 | - |
| `full_mlp_capacity_search_hd128_depth2` | `jit` | `bf16` | 128 | 64 | ok | 64.977935 | 0.26804 | 0.8177721499999999 | 0.8742986499999998 | 362292.5874936599 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 1 | ok | 1388.457008 | 0.048886 | 0.05159955 | 0.057333809999999985 | 20269.421145870914 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 2 | ok | 1365.35398 | 0.043942 | 0.048905449999999996 | 0.051957449999999995 | 22487.191295837893 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 4 | ok | 1381.485139 | 0.0440695 | 0.046168799999999996 | 0.05436444999999997 | 22432.33845912267 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 8 | ok | 1407.827993 | 0.0445365 | 0.046679599999999995 | 0.050857879999999994 | 22326.432084333184 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 1 | 64 | ok | 1485.059741 | 0.0447155 | 0.0511448 | 0.05640844 | 21827.36042166968 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 1 | ok | 1404.627691 | 0.050596 | 0.0539623 | 0.05781264999999999 | 39166.02989229734 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 2 | ok | 1308.987668 | 0.045994999999999994 | 0.04855825 | 0.05110426999999999 | 43183.91554608005 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 4 | ok | 1357.051553 | 0.0463935 | 0.0484372 | 0.05162943 | 42829.94584153348 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 8 | ok | 1322.034251 | 0.049796 | 0.05214895 | 0.05318994 | 39921.609926747835 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 2 | 64 | ok | 1490.023776 | 0.0530395 | 0.05473284999999999 | 0.0610713 | 37527.59216213721 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 1 | ok | 1368.482811 | 0.04623 | 0.047946350000000006 | 0.04884696 | 86250.0215625054 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 2 | ok | 1376.400167 | 0.050428 | 0.05265295 | 0.05532536999999999 | 78941.67631070814 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 4 | ok | 1340.028671 | 0.051372 | 0.0539334 | 0.05663105999999999 | 77339.99901004802 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 8 | ok | 1327.869828 | 0.046115 | 0.049082249999999994 | 0.052020839999999985 | 86211.61589449078 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 4 | 64 | ok | 1537.915709 | 0.045188000000000006 | 0.0471575 | 0.04815534 | 88302.24799862955 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 1 | ok | 1369.743533 | 0.0506975 | 0.05311965 | 0.05598778999999999 | 156692.96366329349 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 2 | ok | 1324.077683 | 0.0463125 | 0.04910675 | 0.056125399999999985 | 170897.44657852632 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 4 | ok | 1325.019628 | 0.047358 | 0.04924925 | 0.052076639999999993 | 168197.82755685927 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 8 | ok | 1357.234368 | 0.049178 | 0.05201765 | 0.05249744 | 161818.38556224 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 8 | 64 | ok | 1507.292709 | 0.0473435 | 0.048616 | 0.05005227 | 168644.7287981955 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 1 | ok | 1371.555094 | 0.048105499999999995 | 0.05064525 | 0.053477889999999986 | 329602.64341320016 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 2 | ok | 1354.689547 | 0.0480935 | 0.049892599999999995 | 0.05074028 | 331273.63947986724 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 4 | ok | 1352.002663 | 0.051513500000000004 | 0.053561349999999994 | 0.05728057 | 309577.8750884232 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 8 | ok | 1346.910455 | 0.0475795 | 0.04920485 | 0.05006308 | 335380.5521202339 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 16 | 64 | ok | 1559.467464 | 0.04684 | 0.048754549999999994 | 0.04910182 | 340566.97590148076 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 1 | ok | 1378.553453 | 0.048728999999999995 | 0.05110705 | 0.05167934 | 651926.8717329805 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 2 | ok | 1313.315634 | 0.0530625 | 0.0556658 | 0.06103390999999998 | 598229.912467747 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 4 | ok | 1354.204588 | 0.048908 | 0.05082275 | 0.054803759999999986 | 651059.9459376098 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 8 | ok | 1387.13138 | 0.048537 | 0.0512974 | 0.053224719999999996 | 654217.7212043004 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 32 | 64 | ok | 1559.553074 | 0.049155500000000005 | 0.05115265 | 0.05431078999999999 | 645637.7084703632 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 1 | ok | 1378.133301 | 0.053376 | 0.05541195 | 0.06259143999999997 | 1186444.1342387386 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 2 | ok | 1383.434228 | 0.0737845 | 0.07679654999999999 | 0.07974059999999998 | 876149.6726485785 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 4 | ok | 1431.322514 | 0.06534000000000001 | 0.0673015 | 0.07014395 | 976446.8806794117 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 8 | ok | 1347.153469 | 0.052951 | 0.0551388 | 0.061608509999999984 | 1204199.9483699272 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 64 | 64 | ok | 1502.908384 | 0.052955 | 0.055161 | 0.05841608999999999 | 1202047.2367000047 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 1 | ok | 1378.004488 | 0.0595085 | 0.06142125 | 0.06228788999999999 | 2144726.1184746707 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 2 | ok | 1316.315383 | 0.0903465 | 0.0926266 | 0.10311975999999998 | 1408248.9502494251 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 4 | ok | 1374.076333 | 0.1060295 | 0.1102529 | 0.11883685 | 1197511.5709555545 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 8 | ok | 1333.740078 | 0.0767235 | 0.0786187 | 0.08195296999999999 | 1665731.4278753907 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `fp32` | 128 | 64 | ok | 1499.368084 | 0.0888815 | 0.09239889999999999 | 0.09405994999999999 | 1436185.1332398318 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 1 | ok | 1373.532184 | 0.058646500000000004 | 0.0605813 | 0.06546981999999998 | 16940.58462635169 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 2 | ok | 1321.528875 | 0.056617 | 0.0593662 | 0.06828665999999996 | 17466.077384503817 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 4 | ok | 1355.126113 | 0.052749000000000004 | 0.055618100000000004 | 0.061302979999999986 | 18779.98459290064 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 8 | ok | 1339.25124 | 0.052917 | 0.0551652 | 0.057373959999999995 | 18860.01044090178 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 1 | 64 | ok | 1572.120777 | 0.056968000000000005 | 0.05890745 | 0.05916499 | 17488.83425377068 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 1 | ok | 1404.551559 | 0.091308 | 0.09848634999999999 | 0.10269323 | 21730.04633063178 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 2 | ok | 1388.193225 | 0.096241 | 0.10622424999999999 | 0.10784027 | 20530.154280003382 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 4 | ok | 1424.358946 | 0.102348 | 0.11278395 | 0.11396011 | 19273.71997443534 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 8 | ok | 1433.754434 | 0.109753 | 0.1217265 | 0.12653514999999999 | 18108.124699065604 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 2 | 64 | ok | 1460.440429 | 0.20554699999999998 | 0.21530505 | 0.22048836 | 9708.162914623503 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 1 | ok | 1318.439986 | 0.090818 | 0.0967459 | 0.10009432 | 43791.081508340016 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 2 | ok | 1362.907874 | 0.0972895 | 0.1127154 | 0.11384213 | 40034.66201036858 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 4 | ok | 1402.867772 | 0.10348450000000001 | 0.11367039999999999 | 0.11914654 | 38016.09943795097 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 8 | ok | 1352.589424 | 0.113393 | 0.12651745 | 0.13235462 | 34706.89936717175 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 4 | 64 | ok | 1511.820642 | 0.211883 | 0.22516404999999998 | 0.24063409 | 18971.460378437 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 1 | ok | 1372.136634 | 0.09053800000000001 | 0.09736389999999999 | 0.10268213 | 87385.12678926508 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 2 | ok | 1316.619545 | 0.0982445 | 0.11383845000000001 | 0.11747977999999999 | 79026.03572016327 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 4 | ok | 1359.116711 | 0.102731 | 0.1267862 | 0.13037026000000002 | 74676.0320367645 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 8 | ok | 1406.916011 | 0.113024 | 0.12301669999999999 | 0.13017181 | 70409.25909920858 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 8 | 64 | ok | 1551.957207 | 0.215431 | 0.22753159999999997 | 0.23224618 | 36982.8332160207 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 1 | ok | 1408.957316 | 0.1018385 | 0.10884175 | 0.11020742 | 156168.56053581432 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 2 | ok | 1367.239567 | 0.1061155 | 0.12245275 | 0.12861879999999998 | 146101.56286668067 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 4 | ok | 1373.534169 | 0.10884450000000001 | 0.1362352 | 0.14029494 | 139848.306541894 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 8 | ok | 1391.088277 | 0.115207 | 0.17811445 | 0.18889320999999998 | 129918.32197495339 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 16 | 64 | ok | 1531.551902 | 0.21445 | 0.22742245 | 0.22997646 | 74675.96930108241 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 1 | ok | 1320.114044 | 0.10626050000000001 | 0.1156805 | 0.11671974 | 299453.92704815726 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 2 | ok | 1421.928139 | 0.1222775 | 0.13115970000000002 | 0.13414521 | 259812.84706327858 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 4 | ok | 1395.980927 | 0.1208835 | 0.14798804999999998 | 0.15338117 | 255587.09393800126 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 8 | ok | 1342.988047 | 0.1230485 | 0.1729851 | 0.19025738999999997 | 244819.57944586617 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 32 | 64 | ok | 1511.663618 | 0.2172945 | 0.23475115 | 0.2504461 | 146124.51306289216 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 1 | ok | 1403.307501 | 0.11959800000000001 | 0.13206925 | 0.13635230999999998 | 527653.1472036856 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 2 | ok | 1315.155422 | 0.14246799999999998 | 0.16216825 | 0.16644403 | 437029.45073182625 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 4 | ok | 1391.431013 | 0.1312975 | 0.1376982 | 0.14474375 | 484843.48949132033 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 8 | ok | 1463.129799 | 0.13764549999999998 | 0.20709829999999996 | 0.22305135999999998 | 435752.69128351635 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 64 | 64 | ok | 1483.994387 | 0.220199 | 0.23785635 | 0.24663617999999998 | 289229.71799198654 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 1 | ok | 1316.255507 | 0.140008 | 0.147535 | 0.15653358 | 907695.7137331951 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 2 | ok | 1366.852853 | 0.18620199999999998 | 0.20599895 | 0.21268302 | 675747.944749581 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 4 | ok | 1390.621202 | 0.16259099999999999 | 0.1960785 | 0.20396661 | 759755.9189140998 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 8 | ok | 1328.403049 | 0.17010150000000002 | 0.1780174 | 0.19161802999999997 | 750892.8291729373 | - |
| `full_mlp_capacity_search_hd128_depth2` | `compile` | `bf16` | 128 | 64 | ok | 1546.144516 | 0.2647715 | 0.62302395 | 0.7280357 | 395960.4853708377 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0715205 | 0.0739833 | 0.07983170999999999 | 13898.718371381538 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0733515 | 0.07654665 | 0.08199347999999998 | 13521.772758495728 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.074188 | 0.07815884999999999 | 0.08151180999999999 | 13369.491568931226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0722495 | 0.0753617 | 0.07771815 | 13771.244209880371 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.073265 | 0.0753668 | 0.08332596999999997 | 13567.976102995051 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.09957250000000001 | 0.10223195 | 0.11177054999999998 | 20241.530032965355 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.088337 | 0.0924539 | 0.09817775999999999 | 22446.200385760396 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.08596200000000001 | 0.08824295 | 0.09314971999999999 | 23154.37088532358 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0859805 | 0.08893995 | 0.09343193 | 23135.562830404757 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.08833350000000001 | 0.09074385 | 0.09671783999999999 | 22544.574569223907 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.087729 | 0.09197055 | 0.10000968999999998 | 45212.13647464243 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0888295 | 0.0925031 | 0.09871484999999999 | 44646.255440425266 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.087906 | 0.09076705 | 0.09503252 | 45310.42740193408 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.086918 | 0.09013195 | 0.09660366999999999 | 45722.37465553906 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.09109 | 0.10888555 | 0.11145775 | 41206.951612737066 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0909395 | 0.0943138 | 0.10014071999999999 | 87271.55307364956 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.093303 | 0.09728695 | 0.10362394999999998 | 85203.5331349085 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.119721 | 0.12512245 | 0.13334608999999997 | 66399.48620077578 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.096371 | 0.1005365 | 0.10782069999999998 | 82305.27189958098 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.09342249999999999 | 0.09776874999999999 | 0.10262959999999999 | 85109.20788009133 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0947355 | 0.099981 | 0.10496824999999999 | 167281.9855702559 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1305715 | 0.13620365 | 0.13996357 | 121712.48862939111 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.122039 | 0.1260104 | 0.12754189999999999 | 130755.1683023321 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.113764 | 0.1171559 | 0.12115436 | 139923.49682810923 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.15366600000000002 | 0.1587858 | 0.16174903999999998 | 104038.2309287194 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.1036115 | 0.11001214999999999 | 0.11556474 | 306135.6081249921 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.1430285 | 0.14810555 | 0.15276938999999998 | 222699.16961047132 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.12735000000000002 | 0.13142715 | 0.13538125 | 249992.7345861511 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1182665 | 0.1228638 | 0.12694665 | 268575.88309428963 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.15071099999999998 | 0.1573543 | 0.16320144 | 211527.5088881215 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.11647350000000001 | 0.12158939999999999 | 0.12753354 | 547092.9021876536 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.166217 | 0.17237834999999999 | 0.17975954 | 383593.1917482871 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1523875 | 0.15808904999999998 | 0.16098476 | 418640.54593867 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.13701000000000002 | 0.14188355 | 0.1472434 | 464578.03030819964 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.1701495 | 0.1759387 | 0.18190122999999997 | 376365.56010487425 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.15330549999999998 | 0.16060534999999998 | 0.16914501999999998 | 829699.4012033493 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.2349215 | 0.24325724999999998 | 0.24643218 | 568168.0982135375 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.2049085 | 0.21740795000000002 | 0.22654488999999997 | 613324.4350179887 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1768135 | 0.1830185 | 0.18717171999999999 | 720585.3675233077 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.24771549999999998 | 0.26256685 | 0.27208963999999997 | 518246.48709857767 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1378055 | 0.14507565 | 0.1523472 | 7216.539037362764 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1415095 | 0.14925295 | 0.15076787 | 7015.330461350581 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.152177 | 0.1588618 | 0.15970072 | 6553.17871763467 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.16100399999999998 | 0.16767325 | 0.17116046 | 6236.652005545132 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.2555205 | 0.27694585 | 0.28536519 | 3885.0346754884927 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.18026399999999998 | 0.18964105 | 0.19279367 | 11022.485429376511 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.17940699999999998 | 0.1842847 | 0.18740426 | 11115.12119983732 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.186744 | 0.19560809999999998 | 0.20149369999999997 | 10654.34739991306 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.2061075 | 0.21921715 | 0.23167205 | 9648.979765124534 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.3488605 | 0.47489644999999997 | 0.4963931199999999 | 5232.0270583696965 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1752395 | 0.18265284999999998 | 0.18555901 | 22720.37657660953 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.193908 | 0.2011919 | 0.20524626999999998 | 20546.97485348209 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.186114 | 0.1953665 | 0.19898583 | 21362.83797611891 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.211369 | 0.2251291 | 0.22787 | 18972.57438482376 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.3710555 | 0.40704914999999997 | 0.41285611 | 10673.495996344967 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.18182 | 0.19200885 | 0.19509751 | 43645.14662696122 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.209858 | 0.21610059999999998 | 0.24082315999999993 | 37865.276859544814 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.2129 | 0.23102209999999998 | 0.23293327 | 37234.54561566949 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.207774 | 0.22173585 | 0.22611472999999999 | 38418.314164035255 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.376795 | 0.4181588 | 0.43735785 | 21004.23891796663 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.20177499999999998 | 0.2065553 | 0.21318722999999998 | 78976.39484406555 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.235993 | 0.24594834999999998 | 0.24904677 | 67469.02518137993 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.2217945 | 0.2331892 | 0.23588874999999998 | 71790.6333863238 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.2316165 | 0.2436084 | 0.25248528 | 69050.33605503864 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.38429599999999997 | 0.41323545 | 0.43685804 | 42116.46835513344 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.22803800000000002 | 0.2375448 | 0.24018066999999999 | 139634.75910604082 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.28871800000000003 | 0.2974076 | 0.30045872 | 113506.99951553794 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.250448 | 0.2620404 | 0.27117053 | 127188.39558516361 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.2482435 | 0.25852559999999997 | 0.26001857 | 128986.31189196558 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.38550549999999995 | 0.433415 | 0.45145434999999995 | 82323.97282321008 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.2689695 | 0.2787253 | 0.28133205 | 237020.11204533247 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.334503 | 0.39063255 | 0.39477126999999995 | 192260.23961513827 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.287397 | 0.30039075 | 0.30650214000000003 | 224377.83881789903 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2804675 | 0.3058049 | 0.30908775 | 227484.10330422793 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.401379 | 0.54321195 | 0.6582047899999997 | 150279.51520204093 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.3673285 | 0.37645435 | 0.38089597999999997 | 347572.887663791 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.427747 | 0.5362209 | 0.55109367 | 291714.9268750444 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.3640445 | 0.42464445 | 0.42790045000000004 | 349132.373321137 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.3365415 | 0.385465 | 0.39090159 | 381628.2036122421 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.4452275 | 0.5379837 | 0.6439076899999998 | 278620.8840893152 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 1 | ok | 58.179314 | 0.0405595 | 0.0468533 | 0.04831658 | 24190.430942851075 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 2 | ok | 58.178849 | 0.043552 | 0.04500615 | 0.046822329999999995 | 23000.16974125269 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 4 | ok | 58.86282 | 0.046994 | 0.04932304999999999 | 0.05346506999999999 | 21141.702685080807 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 8 | ok | 58.53695 | 0.04253 | 0.050947799999999994 | 0.05523631999999999 | 22678.699077249094 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 64 | ok | 58.227127 | 0.0428035 | 0.050162099999999994 | 0.051387709999999996 | 22866.059313643218 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 1 | ok | 58.518144 | 0.0447775 | 0.04762555 | 0.04944307 | 44345.111394919826 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 2 | ok | 59.469285 | 0.042379 | 0.045060699999999995 | 0.04873684999999999 | 46826.10330493955 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 4 | ok | 58.964866 | 0.043302 | 0.0472329 | 0.05061653999999999 | 45681.958550018084 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 8 | ok | 58.711841 | 0.0486375 | 0.0525705 | 0.05563941999999999 | 40623.769861468885 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 64 | ok | 67.753679 | 0.0438165 | 0.045668749999999994 | 0.04977407 | 45269.04525320109 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 1 | ok | 58.423607 | 0.043233999999999995 | 0.04515255 | 0.04700157 | 92029.46599442209 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 2 | ok | 58.585985 | 0.0423005 | 0.0444606 | 0.045870179999999997 | 93902.97381327768 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 4 | ok | 58.869901 | 0.0437485 | 0.047372899999999996 | 0.049663259999999994 | 90484.47195976699 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 8 | ok | 58.41096 | 0.0421595 | 0.045281499999999995 | 0.04837858999999999 | 93946.68619505117 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 64 | ok | 58.442931 | 0.0487555 | 0.05353055 | 0.06178921999999999 | 80463.34009761813 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 1 | ok | 58.29193 | 0.044308 | 0.04621965 | 0.048940309999999994 | 179408.64220399928 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 2 | ok | 58.361778 | 0.044145500000000004 | 0.04760249999999999 | 0.05112846999999999 | 179037.24511810194 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 4 | ok | 59.315057 | 0.0489395 | 0.05158405 | 0.05247133 | 162855.90631551133 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 8 | ok | 58.656934 | 0.043459 | 0.046248199999999996 | 0.04933028999999999 | 182312.5803884534 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 64 | ok | 59.040235 | 0.05007 | 0.0524166 | 0.05504302 | 159317.67426465932 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 1 | ok | 59.495944 | 0.050956 | 0.0532253 | 0.0554999 | 315266.58745593653 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 2 | ok | 59.176971 | 0.057482000000000005 | 0.06139265 | 0.06411034 | 276124.5777451184 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 4 | ok | 58.914773 | 0.050814 | 0.054176249999999995 | 0.05681028999999999 | 312345.5353719606 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 8 | ok | 60.121828 | 0.0509205 | 0.053670949999999995 | 0.056806369999999995 | 312037.67230817775 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 64 | ok | 65.484562 | 0.087177 | 0.0916281 | 0.09666446 | 183145.6548235529 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 1 | ok | 58.511004 | 0.054624 | 0.057837599999999996 | 0.06200859999999998 | 580452.3247296998 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 2 | ok | 60.135919 | 0.06799949999999999 | 0.07016055 | 0.07483569999999999 | 475349.7014061141 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 4 | ok | 59.045771 | 0.058523 | 0.06170994999999999 | 0.06500481999999999 | 542366.8142017394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 8 | ok | 59.354263 | 0.055774000000000004 | 0.0602399 | 0.06330245999999999 | 562690.5894570774 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 64 | ok | 65.612451 | 0.0927705 | 0.09807814999999999 | 0.10243398 | 343020.0038546873 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 1 | ok | 59.471214 | 0.0640815 | 0.06805564999999998 | 0.07258639 | 990444.0717898625 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 2 | ok | 59.772215 | 0.11619850000000001 | 0.12154855 | 0.12588950000000002 | 549357.4492097578 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 4 | ok | 59.889319 | 0.0763215 | 0.0882931 | 0.09022168 | 822758.3178294813 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 8 | ok | 60.610628 | 0.0706345 | 0.0771217 | 0.08025399999999999 | 907167.5592344981 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 64 | ok | 66.245536 | 0.104449 | 0.11207545 | 0.11588161 | 607944.1968021755 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 1 | ok | 59.677405 | 0.09273300000000001 | 0.098922 | 0.10593066999999999 | 1381366.9143630217 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 2 | ok | 61.224494 | 0.15813549999999998 | 0.16315355 | 0.16477547 | 813087.1457909718 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 4 | ok | 60.78083 | 0.1351215 | 0.1701404 | 0.17232232 | 866974.0721921183 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 8 | ok | 60.833377 | 0.097076 | 0.10221585 | 0.10414798 | 1311415.4824482405 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 64 | ok | 69.499173 | 0.170919 | 0.1795112 | 0.18567060999999999 | 748328.9172183351 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 1 | ok | 58.186169 | 0.0828 | 0.08891205 | 0.09135736 | 11955.027100850935 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 2 | ok | 58.461078 | 0.091258 | 0.0969608 | 0.09957711 | 10890.024258618037 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 4 | ok | 58.343334 | 0.093829 | 0.10099564999999999 | 0.10527937999999998 | 10501.115323458504 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 8 | ok | 58.611583 | 0.104189 | 0.11718214999999998 | 0.12191877999999999 | 9587.885821319394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 64 | ok | 57.859068 | 0.2156805 | 0.23299915 | 0.23487574 | 4634.167681240445 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 1 | ok | 58.26433 | 0.10864499999999999 | 0.11728019999999999 | 0.11902489 | 18242.82217918538 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 2 | ok | 58.571961 | 0.119205 | 0.1240303 | 0.12995206999999998 | 16716.705337978347 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 4 | ok | 58.512347 | 0.11758199999999999 | 0.13230414999999998 | 0.13275704 | 16763.46449850252 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 8 | ok | 59.257747 | 0.144113 | 0.16066425 | 0.16503448999999998 | 13789.527681097867 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 64 | ok | 58.662827 | 0.32142950000000003 | 0.3593699 | 0.36040536 | 6160.567427687721 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 1 | ok | 58.326896 | 0.1110265 | 0.11567279999999999 | 0.12209671999999999 | 35853.236211696865 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 2 | ok | 58.743658 | 0.1320595 | 0.13888045 | 0.14147997 | 30165.239146999436 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 4 | ok | 58.893813 | 0.12485299999999999 | 0.1339598 | 0.13752384 | 31787.264368956046 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 8 | ok | 58.645695 | 0.138897 | 0.15061634999999998 | 0.15403592 | 28750.83790723226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 64 | ok | 58.782223 | 0.3227815 | 0.36888109999999996 | 0.37246354 | 12308.436873442868 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 1 | ok | 58.833128 | 0.1177645 | 0.1293056 | 0.13430806 | 67029.31761808638 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 2 | ok | 58.827194 | 0.142318 | 0.1460714 | 0.15013528 | 56634.52987464372 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 4 | ok | 59.49707 | 0.133565 | 0.1410377 | 0.14399983 | 59928.62201474933 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 8 | ok | 58.646429 | 0.142665 | 0.1544024 | 0.15670651 | 55651.27574984529 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 64 | ok | 58.493635 | 0.308158 | 0.33396804999999996 | 0.34660988 | 26263.181901358512 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 1 | ok | 58.717346 | 0.127692 | 0.13731274999999998 | 0.14374622 | 123893.82175581704 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 2 | ok | 58.670238 | 0.16786600000000002 | 0.17487049999999998 | 0.18056997999999996 | 94668.70803957435 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 4 | ok | 59.295925 | 0.152886 | 0.1628751 | 0.16899255 | 104971.04832880184 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 8 | ok | 59.844984 | 0.160395 | 0.17665685 | 0.18039525 | 98815.15686165038 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 64 | ok | 65.628504 | 0.332592 | 0.3782515 | 0.3892563 | 47449.683317847936 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 1 | ok | 58.80409 | 0.154087 | 0.1614344 | 0.16916203 | 206574.67810178013 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 2 | ok | 59.253149 | 0.217254 | 0.2234831 | 0.2278286 | 148795.5327117214 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 4 | ok | 59.884037 | 0.18394749999999999 | 0.1991619 | 0.20756919 | 174160.18056056718 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 8 | ok | 59.358103 | 0.1717075 | 0.19002884999999997 | 0.19495889 | 185321.0443860117 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 64 | ok | 65.659755 | 0.34034 | 0.37661705 | 0.39155062999999996 | 94105.93917286172 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 1 | ok | 59.195534 | 0.20543050000000002 | 0.21444839999999998 | 0.21866191 | 309701.5261220172 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 2 | ok | 60.309529 | 0.3211715 | 0.32932324999999996 | 0.33336074000000004 | 217070.18204590818 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 4 | ok | 59.340475 | 0.213764 | 0.24910275 | 0.25390904 | 288389.38045760005 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 8 | ok | 59.90094 | 0.21811750000000002 | 0.23545824999999998 | 0.24596325 | 298048.8511381042 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 64 | ok | 66.412726 | 0.32521449999999996 | 0.4498276 | 0.45533178999999996 | 191230.38947295828 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 1 | ok | 60.904298 | 0.312432 | 0.32080225 | 0.32890544 | 409097.30123624095 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 2 | ok | 61.063701 | 0.38491450000000005 | 0.42106855 | 0.43175009 | 340159.51993286953 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 4 | ok | 60.799391 | 0.3297675 | 0.3398652 | 0.34477697 | 410735.70723169524 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 8 | ok | 60.630496 | 0.275993 | 0.30321529999999997 | 0.33250346 | 466457.03795845166 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 64 | ok | 69.984896 | 0.4168265 | 0.45339294999999996 | 2.212817879999993 | 267872.95438713743 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 1 | ok | 1368.257395 | 0.0578395 | 0.0606407 | 0.06884605999999999 | 17146.71172365262 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 2 | ok | 1366.197207 | 0.0596305 | 0.0617937 | 0.06778355999999999 | 16629.920086582017 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 4 | ok | 1375.920875 | 0.06402150000000001 | 0.06608025 | 0.07658757999999999 | 15495.654863419748 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 8 | ok | 1384.516571 | 0.0613135 | 0.0639316 | 0.07807553999999997 | 16133.627739772328 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 64 | ok | 1503.241679 | 0.060555 | 0.064375 | 0.0672572 | 16403.564428936148 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 1 | ok | 1363.318601 | 0.0586615 | 0.06030215 | 0.06103806 | 33987.60607956703 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 2 | ok | 1364.437231 | 0.061025499999999996 | 0.06398175 | 0.07252506999999998 | 32560.074966316603 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 4 | ok | 1406.953289 | 0.058143 | 0.0607184 | 0.0669469 | 34145.39038937013 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 8 | ok | 1358.519645 | 0.059286 | 0.06134195 | 0.06536231999999999 | 33543.35948820884 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 64 | ok | 1499.190711 | 0.059528 | 0.0612813 | 0.06522462 | 33419.991370958225 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 1 | ok | 1362.275529 | 0.061688 | 0.0648688 | 0.06873924999999999 | 64374.09071596863 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 2 | ok | 1380.080613 | 0.0635665 | 0.0661431 | 0.07136544999999998 | 62555.79194694268 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 4 | ok | 1325.466605 | 0.060947 | 0.06301245 | 0.06377806999999999 | 65307.252664291 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 8 | ok | 1391.413213 | 0.07028899999999999 | 0.0724733 | 0.08065475999999996 | 56805.72329023294 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 64 | ok | 1528.34204 | 0.060226 | 0.06202665 | 0.06895693999999998 | 66103.11557204314 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 1 | ok | 1353.331178 | 0.062958 | 0.06525405000000001 | 0.06848024999999999 | 126453.46401877329 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 2 | ok | 1323.881805 | 0.0671645 | 0.0693898 | 0.07261769 | 118773.37979747953 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 4 | ok | 1386.151843 | 0.09293950000000001 | 0.0951295 | 0.09973795999999999 | 85736.98764379969 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 8 | ok | 1395.789129 | 0.0659035 | 0.0682711 | 0.0720097 | 120994.00202483464 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 64 | ok | 1542.602648 | 0.063858 | 0.06600015 | 0.06925713 | 124659.71792622325 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 1 | ok | 1370.203181 | 0.07145950000000001 | 0.07579425 | 0.07883714 | 222740.34100432508 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 2 | ok | 1396.681711 | 0.107465 | 0.11294919999999999 | 0.11886652 | 147973.64885261233 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 4 | ok | 1448.484964 | 0.09201899999999999 | 0.09502334999999999 | 0.10498523 | 172633.34253325188 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 8 | ok | 1399.922643 | 0.084588 | 0.0870608 | 0.10935060999999995 | 187017.96120499415 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 64 | ok | 1567.454171 | 0.10675 | 0.1104343 | 0.11833714999999997 | 149342.9749157939 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 1 | ok | 1333.857866 | 0.0747275 | 0.0766837 | 0.07972807999999999 | 426746.55362150463 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 2 | ok | 1365.84838 | 0.11333399999999999 | 0.1193197 | 0.12839384999999998 | 280128.2777415848 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 4 | ok | 1393.680232 | 0.0998465 | 0.10223435 | 0.10739963999999999 | 319321.25074940705 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 8 | ok | 1344.541645 | 0.0910695 | 0.09325385 | 0.09627979999999998 | 350936.7598798568 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 64 | ok | 1489.99983 | 0.1008995 | 0.10376525 | 0.10644580999999999 | 317187.43557130214 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 1 | ok | 1367.027573 | 0.086831 | 0.08929795 | 0.09486724999999999 | 733937.2107541985 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 2 | ok | 1373.574631 | 0.1360245 | 0.139263 | 0.14042548 | 469549.2195431667 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 4 | ok | 1400.647157 | 0.1235495 | 0.12674725 | 0.13857461999999998 | 515090.37743569334 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 8 | ok | 1420.131848 | 0.109318 | 0.11375215 | 0.12352211999999999 | 580786.5992204029 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 64 | ok | 1541.698314 | 0.1165835 | 0.1211807 | 0.12752404999999997 | 546848.698927322 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 1 | ok | 1324.038198 | 0.10714599999999999 | 0.11434729999999999 | 0.12193646999999999 | 1182789.160994053 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 2 | ok | 1390.460659 | 0.207613 | 0.2105651 | 0.21677633000000002 | 647069.6238826928 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 4 | ok | 1325.769035 | 0.1694405 | 0.17261680000000001 | 0.17670196 | 772489.150751982 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 8 | ok | 1382.857362 | 0.1428505 | 0.14814 | 0.15420167999999998 | 894331.5172348163 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 64 | ok | 1519.33739 | 0.15457549999999998 | 0.15857084999999999 | 0.16036494 | 829898.0094247886 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 1 | ok | 1400.218777 | 0.117293 | 0.12401754999999998 | 0.12938924 | 8467.434922682158 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 2 | ok | 1316.299323 | 0.1190495 | 0.1266633 | 0.13353871 | 8340.808087514428 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 4 | ok | 1415.15151 | 0.118699 | 0.13123955 | 0.13213695 | 8306.920096400147 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 8 | ok | 1397.989507 | 0.1323105 | 0.1440162 | 0.14894412999999998 | 7565.1754999673185 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 64 | ok | 1574.831905 | 0.2315695 | 0.2479287 | 0.25087831 | 4328.647168766075 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 1 | ok | 1376.376819 | 0.151423 | 0.16166139999999998 | 0.1642804 | 13109.649501721888 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 2 | ok | 1334.395325 | 0.16139399999999998 | 0.16463085 | 0.17355088999999999 | 12362.811426946602 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 4 | ok | 1378.54009 | 0.252687 | 0.26362195 | 0.26908781 | 8403.002628123102 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 8 | ok | 1385.532292 | 0.1774825 | 0.18588339999999998 | 0.19389575 | 11317.24594367271 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 64 | ok | 1524.253908 | 0.360153 | 0.39194675 | 0.40172803999999995 | 5627.069038116134 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 1 | ok | 1311.574801 | 0.1529665 | 0.16303335 | 0.16607854 | 25945.310916930637 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 2 | ok | 1392.944158 | 0.170456 | 0.1796789 | 0.18202860999999998 | 23364.464145360613 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 4 | ok | 1385.839323 | 0.166623 | 0.17921355 | 0.18242785 | 23725.1864206523 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 8 | ok | 1328.868908 | 0.2334105 | 0.29980455 | 0.30350241 | 15886.976238167383 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 64 | ok | 1555.594589 | 0.3500065 | 0.38901454999999996 | 0.39225217 | 11347.231876825343 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 1 | ok | 1381.860745 | 0.152473 | 0.16439094999999998 | 0.16700173 | 52098.39981067441 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 2 | ok | 1375.227362 | 0.182135 | 0.18922475 | 0.19248591 | 43742.24532007185 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 4 | ok | 1388.270642 | 0.243156 | 0.28234005 | 0.28568832 | 33524.31015350782 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 8 | ok | 1316.056235 | 0.192628 | 0.20611174999999998 | 0.20814725 | 41182.76488934552 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 64 | ok | 1581.730327 | 0.3574535 | 0.39036005 | 0.40021421 | 22606.934473687394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 1 | ok | 1361.698108 | 0.16947050000000002 | 0.17806724999999998 | 0.18208495 | 93997.19794352929 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 2 | ok | 1380.694701 | 0.212813 | 0.2192372 | 0.22459832999999998 | 76015.28629399728 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 4 | ok | 1411.620247 | 0.1922375 | 0.20428834999999998 | 0.20785327999999997 | 82440.95295558545 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 8 | ok | 1336.978337 | 0.2040795 | 0.2183872 | 0.22737597999999998 | 77813.2606246226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 64 | ok | 1480.879244 | 0.356126 | 0.40142304999999995 | 0.40776267 | 44898.946502344006 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 1 | ok | 1395.189283 | 0.1941235 | 0.20180800000000002 | 0.20705627999999998 | 164188.44729037755 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 2 | ok | 1329.389722 | 0.2515675 | 0.25709485 | 0.26336645 | 130519.93267781874 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 4 | ok | 1411.938306 | 0.22491250000000002 | 0.2427767 | 0.24404445 | 142134.73039928757 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 8 | ok | 1388.966757 | 0.2160445 | 0.2283651 | 0.23297421 | 148312.92655608762 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 64 | ok | 1466.204609 | 0.3812175 | 0.4347908 | 0.44455811 | 81871.89443274189 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 1 | ok | 1331.946293 | 0.250904 | 0.2580197 | 0.26022565999999997 | 254264.552891278 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 2 | ok | 1370.604878 | 0.3070155 | 0.37021390000000004 | 0.37345971 | 203819.63083678912 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 4 | ok | 1387.690151 | 0.269034 | 0.28144205 | 0.28433005 | 242420.97341414634 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 8 | ok | 1388.410859 | 0.25006700000000004 | 0.26919035 | 0.27781282 | 257166.52794407654 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 64 | ok | 1598.795064 | 0.370212 | 0.40605615 | 0.4321883299999999 | 172522.63294769314 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 1 | ok | 1317.716051 | 0.357355 | 0.3638933 | 0.36696667 | 357703.3855339944 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 2 | ok | 1382.555747 | 0.38418300000000005 | 0.46636835 | 0.46933655 | 313539.56430836086 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 4 | ok | 1383.711493 | 0.3196605 | 0.37986525 | 0.38358271 | 402783.1560867175 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 8 | ok | 1407.062113 | 0.31973 | 0.348663 | 0.35831963 | 404364.22728681396 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 64 | ok | 1560.985262 | 0.41463300000000003 | 0.46474115 | 0.4665455 | 306603.0708022183 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0819735 | 0.08818899999999999 | 0.09304928999999998 | 12074.194476152623 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0908635 | 0.09522385 | 0.10105074 | 10920.54994142217 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.08640400000000001 | 0.0912617 | 0.09300950999999999 | 11493.277926539105 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.091422 | 0.0943518 | 0.09878004 | 10902.466966615339 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0832705 | 0.08714944999999999 | 0.09226084999999999 | 11923.20312559616 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1007045 | 0.10640329999999999 | 0.12794336999999995 | 19571.532188711844 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.102239 | 0.10862664999999999 | 0.11473377999999998 | 19384.931627407655 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1014455 | 0.10659904999999999 | 0.11038808 | 19574.612354164245 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.104517 | 0.11350534999999999 | 0.11569057 | 18988.20585569685 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.1020635 | 0.10672839999999999 | 0.11011407 | 19495.218115424555 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1104195 | 0.11979455 | 0.12875822 | 35819.4477751556 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.1076105 | 0.113624 | 0.11800163999999999 | 36845.65796028333 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.106572 | 0.1117428 | 0.1157484 | 37373.20905246395 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.109121 | 0.11388369999999999 | 0.12212724 | 36459.42521716145 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.1079165 | 0.1133767 | 0.12277953999999996 | 36801.405813702084 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.108099 | 0.11299124999999999 | 0.11894619 | 73600.04887043245 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1123865 | 0.1183627 | 0.12529654999999998 | 70564.32229054613 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.15180949999999999 | 0.1584425 | 0.16470276999999997 | 52406.60291752799 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.10767399999999999 | 0.11263764999999999 | 0.11678161 | 73785.43637813855 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.112569 | 0.11731069999999999 | 0.1206525 | 70739.80573080851 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.11396600000000001 | 0.1184325 | 0.12343472 | 139713.26994891558 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1880735 | 0.1944827 | 0.1998941 | 86135.52843985193 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.16256199999999998 | 0.16938474999999997 | 0.17363515999999998 | 98036.96900573478 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1556595 | 0.1610646 | 0.16627058 | 102931.52853191204 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.213011 | 0.21653894999999998 | 0.21923275 | 75128.07928867232 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.124385 | 0.13101444999999998 | 0.14000906999999999 | 255269.520739452 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.21241149999999998 | 0.22038575 | 0.22624961999999998 | 149925.33249924873 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1769755 | 0.18100734999999998 | 0.19012687999999997 | 183180.10508355705 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.15494249999999998 | 0.16180399999999998 | 0.16418112 | 205514.9684900597 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.2000205 | 0.2106707 | 0.21557257 | 160064.6340992493 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.1522125 | 0.16128515000000002 | 0.16573905 | 417312.84953211405 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.2274755 | 0.2328535 | 0.23811376 | 286128.6101606773 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.210382 | 0.21530775000000002 | 0.21987101 | 314763.28325646225 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1875545 | 0.1935251 | 0.19779698 | 340536.7860074286 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.21781699999999998 | 0.230772 | 0.23845043999999999 | 290666.56929336593 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.21634550000000002 | 0.22625935 | 0.22926881 | 588677.5056299185 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.284292 | 0.32667615 | 0.33045629 | 443760.59088386776 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.27522100000000005 | 0.28069965 | 0.28777387 | 485760.64766467153 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.2259875 | 0.24104105 | 0.24512741999999998 | 573006.0776247752 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.299801 | 0.32560185 | 0.33102296999999997 | 419163.93953586376 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.190472 | 0.19776629999999998 | 0.20417755999999998 | 5217.233063400651 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.2078105 | 0.21757705 | 0.22243986000000002 | 4781.675853062926 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.214104 | 0.22459205 | 0.23004371999999998 | 4643.631899670804 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.2262495 | 0.24170774999999997 | 0.24807718 | 4409.031884178612 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.38741899999999996 | 0.44051450000000003 | 0.44391477999999995 | 2527.6210849297872 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.229928 | 0.2376451 | 0.24202547 | 8656.951085888988 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.23901650000000002 | 0.24729835 | 0.24904699 | 8362.637408585919 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2470215 | 0.26149655 | 0.26590902 | 8058.002793387248 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.277257 | 0.29696495 | 0.31138339 | 7155.4271840116835 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.47720450000000003 | 0.5604954 | 0.59155381 | 4153.310830418367 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.238756 | 0.25029695 | 0.26394762 | 16610.605871849177 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.2597035 | 0.2677575 | 0.28065916999999996 | 15340.397279472585 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.25797000000000003 | 0.27197825 | 0.27812106999999997 | 15428.19647931642 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.268473 | 0.28634085 | 0.29038779 | 14871.806883638892 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.498988 | 0.604595 | 0.62038595 | 7950.979982135738 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.257505 | 0.26570385 | 0.27849247 | 30879.055693773636 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.289948 | 0.29775825 | 0.29897469000000004 | 27853.417213899273 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.27565399999999995 | 0.29208225 | 0.29752074 | 28967.06697946271 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.28709850000000003 | 0.3094151 | 0.31399632 | 27768.92064912907 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.4697755 | 0.5279556999999999 | 0.53226088 | 16927.51831324728 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.27173749999999997 | 0.28312785 | 0.29860662999999993 | 58367.835412042994 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.3138615 | 0.3317109 | 0.33615508 | 51592.861895388894 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.294769 | 0.31431794999999996 | 0.32178365 | 54596.82767862614 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.305551 | 0.32995090000000005 | 0.33521355 | 52220.790284635295 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.5233055 | 0.634342 | 0.6403607 | 30178.97300426734 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.30539700000000003 | 0.3238223 | 0.33091055999999996 | 103702.98066347741 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.36189550000000004 | 0.4158384 | 0.42261361999999997 | 85877.71560767903 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.343322 | 0.36798895 | 0.37293223 | 94696.4204397962 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.321915 | 0.36060245 | 0.37984614999999994 | 98175.06061389603 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.50503 | 0.59957645 | 0.61259749 | 62546.63878076898 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.3889675 | 0.39736974999999997 | 0.39892023 | 164384.7917149654 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.4695945 | 0.58665815 | 0.5895003799999999 | 136540.9301065753 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.39153400000000005 | 0.44814545 | 0.45874086999999997 | 164410.99913694503 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.363716 | 0.42661345 | 0.43674858 | 171531.1562491775 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.49405299999999996 | 0.63879825 | 0.65248747 | 126525.1369664262 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.5598574999999999 | 0.56945925 | 0.5768560699999999 | 228328.51149896276 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.4899905 | 0.73767575 | 0.7425741499999999 | 234162.99844841403 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.495168 | 0.64779015 | 0.66328417 | 259852.04674088687 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.43194750000000004 | 0.5176923999999999 | 0.5293566199999999 | 292076.06756084535 | - |
| `full_mlp_capacity_search_hd128_depth4` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.5678255000000001 | 0.69368145 | 0.70547939 | 220812.42688176 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 1 | ok | 62.779361 | 0.0451565 | 0.05013445 | 0.052569149999999995 | 21730.825480175183 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 2 | ok | 62.200737 | 0.048344 | 0.0538951 | 0.055463559999999995 | 20378.651650589272 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 4 | ok | 62.489072 | 0.048891 | 0.05361345 | 0.055955929999999994 | 20149.517479504917 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 8 | ok | 62.81535 | 0.051957 | 0.058252849999999995 | 0.06123595 | 18888.794864665564 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 1 | 64 | ok | 62.744427 | 0.049936 | 0.05578544999999999 | 0.060498819999999995 | 19689.946352772167 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 1 | ok | 62.857879 | 0.047253500000000004 | 0.050649599999999996 | 0.05447862999999999 | 41888.009892272414 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 2 | ok | 62.902125 | 0.0475445 | 0.0500821 | 0.05390455 | 41834.59771432492 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 4 | ok | 62.979683 | 0.047292 | 0.0512151 | 0.055228659999999985 | 41912.29015040225 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 8 | ok | 63.177989 | 0.0480615 | 0.051498999999999996 | 0.05588762 | 41226.25007266126 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 2 | 64 | ok | 62.54929 | 0.0529005 | 0.0547907 | 0.05751202 | 37586.59481613202 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 1 | ok | 63.155276 | 0.047717499999999996 | 0.05078819999999999 | 0.052488309999999996 | 83213.88673341808 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 2 | ok | 62.964733 | 0.0493315 | 0.05187915 | 0.05745811999999999 | 80240.97971026586 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 4 | ok | 62.885067 | 0.0493255 | 0.05235144999999999 | 0.055279739999999994 | 80473.47373003807 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 8 | ok | 63.525868 | 0.0496165 | 0.052616699999999995 | 0.05455918 | 80059.05155642802 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 4 | 64 | ok | 63.198924 | 0.0510635 | 0.05531865 | 0.06041751 | 77616.85217094336 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 1 | ok | 63.397758 | 0.0527055 | 0.05724535 | 0.05973889999999999 | 150129.1298177845 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 2 | ok | 62.664862 | 0.0513975 | 0.05492874999999999 | 0.05802867999999999 | 154379.11806297433 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 4 | ok | 63.402085 | 0.057845499999999994 | 0.06107175 | 0.06513035999999998 | 136998.9221609799 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 8 | ok | 63.123349 | 0.048561 | 0.0501893 | 0.0528832 | 163847.60043141074 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 8 | 64 | ok | 63.099614 | 0.0554915 | 0.05896825 | 0.06278502999999999 | 143034.40334986572 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 1 | ok | 63.678779 | 0.055077 | 0.0598142 | 0.06434214999999999 | 288154.4392532622 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 2 | ok | 63.815653 | 0.064647 | 0.0677969 | 0.07129574 | 246228.32075055316 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 4 | ok | 63.538762 | 0.062158 | 0.06494875 | 0.06759617999999999 | 256042.85133159885 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 8 | ok | 64.276588 | 0.065609 | 0.07021034999999999 | 0.07427669 | 241526.34992096052 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 16 | 64 | ok | 66.401906 | 0.092175 | 0.09659085 | 0.10379895999999998 | 172487.47201929791 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 1 | ok | 63.326209 | 0.0620555 | 0.06511404999999999 | 0.06984826 | 511567.8274993285 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 2 | ok | 64.065018 | 0.0902255 | 0.09445885 | 0.09884717999999999 | 353116.22864805476 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 4 | ok | 64.017041 | 0.072024 | 0.08024975 | 0.08235603999999999 | 435822.8990067596 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 8 | ok | 63.705295 | 0.0726175 | 0.0764128 | 0.07780661 | 439628.1954444078 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 32 | 64 | ok | 70.817376 | 0.123558 | 0.13065965000000002 | 0.13605599000000002 | 258276.761306267 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 1 | ok | 63.87199 | 0.08473900000000001 | 0.08907495 | 0.09191339999999999 | 754867.59858217 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 2 | ok | 64.343489 | 0.14190599999999998 | 0.14666395 | 0.14754725999999999 | 449825.8611634971 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 4 | ok | 64.225365 | 0.100952 | 0.1042443 | 0.10576672 | 633969.3036025503 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 8 | ok | 64.018718 | 0.0903335 | 0.10139770000000001 | 0.10425626999999998 | 691931.0888530334 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 64 | 64 | ok | 71.908091 | 0.14388099999999998 | 0.15234265 | 0.15526091 | 443363.622458141 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 1 | ok | 64.171276 | 0.1191155 | 0.124322 | 0.12867347999999998 | 1069971.451824201 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 2 | ok | 65.008762 | 0.1773305 | 0.21577035 | 0.21932257 | 668831.1016494733 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 4 | ok | 65.32361 | 0.17032150000000001 | 0.1748179 | 0.17711849 | 748811.5249581806 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 8 | ok | 64.83885 | 0.1266065 | 0.13395815 | 0.13847171 | 1024203.8575358039 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `fp32` | 128 | 64 | ok | 74.6838 | 0.21954200000000001 | 0.23059765 | 0.23131567 | 582221.7673614665 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 1 | ok | 62.599534 | 0.1145775 | 0.12068279999999999 | 0.12513501 | 8671.832535120055 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 2 | ok | 63.299137 | 0.12482850000000001 | 0.1307464 | 0.13213927 | 7974.671168421706 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 4 | ok | 63.114168 | 0.1276235 | 0.14008225 | 0.14288811 | 7749.783664788997 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 8 | ok | 63.014729 | 0.1476875 | 0.1610934 | 0.16867419 | 6736.012568860572 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 1 | 64 | ok | 62.705217 | 0.302782 | 0.34699585 | 0.35382709999999995 | 3270.3860330969605 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 1 | ok | 62.690387 | 0.138679 | 0.14666285 | 0.14895713 | 14284.789855742192 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 2 | ok | 62.252847 | 0.148563 | 0.15631314999999998 | 0.15915198 | 13404.37832571004 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 4 | ok | 62.785357 | 0.158213 | 0.1679396 | 0.17002852999999998 | 12538.417711869166 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 8 | ok | 62.643257 | 0.1890565 | 0.20610935 | 0.20990984999999998 | 10450.069879617286 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 2 | 64 | ok | 63.348931 | 0.392586 | 0.46753724999999996 | 0.47840116 | 5000.182506661494 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 1 | ok | 63.140539 | 0.144347 | 0.1515065 | 0.15725667999999998 | 27511.852793979964 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 2 | ok | 63.007167 | 0.17246899999999998 | 0.1794704 | 0.18176376 | 23479.49160326421 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 4 | ok | 62.956628 | 0.1621155 | 0.17285425000000001 | 0.17520646 | 24459.34752489136 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 8 | ok | 62.51302 | 0.188813 | 0.20164965 | 0.20435089 | 21153.616718464822 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 4 | 64 | ok | 63.661529 | 0.41198049999999997 | 0.48137519999999995 | 0.49334963 | 9803.891291904343 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 1 | ok | 63.438904 | 0.161427 | 0.1688319 | 0.17035016 | 49360.68046659664 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 2 | ok | 63.086637 | 0.1927395 | 0.20064525 | 0.2287674099999999 | 41744.023560326896 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 4 | ok | 63.93655 | 0.18487199999999998 | 0.1983062 | 0.20445613 | 42825.26862149743 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 8 | ok | 63.658873 | 0.19393500000000002 | 0.21430749999999998 | 0.21959589999999998 | 41087.24661109822 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 8 | 64 | ok | 62.967014 | 0.403083 | 0.4647235 | 0.47023354 | 19790.048366383457 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 1 | ok | 63.524487 | 0.1819645 | 0.19411574999999998 | 0.21191572 | 86846.74556476384 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 2 | ok | 63.409245 | 0.234477 | 0.2449433 | 0.27072057999999993 | 69612.23894540244 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 4 | ok | 63.672915 | 0.19852399999999998 | 0.21509974999999998 | 0.22365443 | 79140.72952913739 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 8 | ok | 63.491014 | 0.2077265 | 0.22949355 | 0.24205937 | 76478.09208574111 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 16 | 64 | ok | 70.955288 | 0.378715 | 0.5304534 | 0.53402956 | 39658.94890301364 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 1 | ok | 63.158382 | 0.2200585 | 0.23334349999999998 | 0.24406309 | 144044.9866897931 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 2 | ok | 64.663028 | 0.2761255 | 0.3267457 | 0.33282825 | 113779.63870697975 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 4 | ok | 63.66315 | 0.2310305 | 0.26720135 | 0.27706318999999996 | 132852.9260815453 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 8 | ok | 63.31241 | 0.2311865 | 0.26120755 | 0.26377673 | 136747.262939432 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 32 | 64 | ok | 71.335365 | 0.4070555 | 0.52615485 | 0.53483755 | 74910.14761887328 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 1 | ok | 63.799493 | 0.3006345 | 0.31071165 | 0.31559917 | 211868.50275442295 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 2 | ok | 64.828148 | 0.38406549999999995 | 0.4634384 | 0.46891186 | 164897.95573427746 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 4 | ok | 64.541633 | 0.31170549999999997 | 0.38202624999999996 | 0.38987239 | 192490.44254876114 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 8 | ok | 63.775598 | 0.2808125 | 0.32349989999999995 | 0.33730462999999994 | 222670.12937273688 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 64 | 64 | ok | 67.410606 | 0.40033799999999997 | 0.5416136500000001 | 0.5621236199999999 | 151939.51022203916 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 1 | ok | 64.605593 | 0.4722905 | 0.4815048 | 0.48527738 | 270088.1546636185 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 2 | ok | 65.876676 | 0.4282395 | 0.54651575 | 0.69736678 | 272758.2679849983 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 4 | ok | 65.901427 | 0.391528 | 0.47814555 | 0.55587733 | 305927.1960195048 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 8 | ok | 65.344655 | 0.3719705 | 0.47164055 | 0.47606289999999996 | 341435.45440150844 | - |
| `full_mlp_capacity_search_hd128_depth4` | `jit` | `bf16` | 128 | 64 | ok | 74.760051 | 0.5362610000000001 | 0.62665355 | 0.63714466 | 237132.79477264464 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 1 | ok | 1384.01677 | 0.07311799999999999 | 0.0773777 | 0.08317948999999998 | 13605.564349285314 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 2 | ok | 1383.945969 | 0.07621149999999999 | 0.08412815 | 0.08596125 | 12992.014068792194 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 4 | ok | 1400.401929 | 0.0775265 | 0.0805854 | 0.09029954 | 12791.70851687303 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 8 | ok | 1317.337129 | 0.074532 | 0.0773716 | 0.08480221 | 13319.935254458715 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 1 | 64 | ok | 1498.136521 | 0.072755 | 0.07531425 | 0.07859142999999999 | 13719.954329016029 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 1 | ok | 1398.153726 | 0.07632900000000001 | 0.07799475 | 0.08129592999999999 | 26142.52643989768 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 2 | ok | 1399.275311 | 0.0747865 | 0.07790504999999999 | 0.08549329 | 26595.8720015149 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 4 | ok | 1402.580238 | 0.07162099999999999 | 0.07405865 | 0.07733031 | 27806.774063843237 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 8 | ok | 1389.724203 | 0.071159 | 0.0746317 | 0.08146955 | 27918.619457770154 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 2 | 64 | ok | 1503.702406 | 0.07178899999999999 | 0.07567745 | 0.07921024 | 27685.546455931668 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 1 | ok | 1353.290228 | 0.078158 | 0.08111884999999999 | 0.08669167999999998 | 50849.93117461816 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 2 | ok | 1379.920889 | 0.07425999999999999 | 0.0766748 | 0.08077313999999998 | 53585.08319887943 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 4 | ok | 1420.227917 | 0.074936 | 0.0771625 | 0.08337067999999999 | 53150.84875261601 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 8 | ok | 1426.487453 | 0.079776 | 0.08193679999999999 | 0.09090358999999996 | 49841.342546339365 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 4 | 64 | ok | 1482.109698 | 0.07732549999999999 | 0.0794247 | 0.08382936999999999 | 51479.322686551415 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 1 | ok | 1378.02309 | 0.07682749999999999 | 0.07934145000000001 | 0.08343239999999999 | 103589.07663545787 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 2 | ok | 1396.442608 | 0.080259 | 0.08262934999999999 | 0.09200224999999998 | 99130.67355823108 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 4 | ok | 1394.289802 | 0.12247549999999999 | 0.1251869 | 0.12862477999999997 | 65232.9975899669 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 8 | ok | 1413.106435 | 0.0764175 | 0.08306575 | 0.08640046 | 103912.99987998049 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 8 | 64 | ok | 1472.342711 | 0.07761599999999999 | 0.08061629999999999 | 0.08798296 | 102540.12395050183 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 1 | ok | 1397.192075 | 0.08139099999999999 | 0.0832907 | 0.08649183999999999 | 196269.74626983213 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 2 | ok | 1365.630481 | 0.15735749999999998 | 0.15986324999999998 | 0.16792171999999997 | 101439.18106120346 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 4 | ok | 1407.861505 | 0.131672 | 0.13373995 | 0.13709573 | 121363.89964055049 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 8 | ok | 1431.867625 | 0.1175075 | 0.12380859999999999 | 0.12939515000000001 | 135230.07029935205 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 16 | 64 | ok | 1517.35202 | 0.146485 | 0.15074869999999999 | 0.15328515 | 108821.71384949467 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 1 | ok | 1415.532318 | 0.095801 | 0.0988643 | 0.10659893999999999 | 332208.1167578647 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 2 | ok | 1365.494703 | 0.176033 | 0.17932765 | 0.18311897 | 181379.26460910783 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 4 | ok | 1402.532351 | 0.14048 | 0.1428207 | 0.14521444 | 227442.85711442682 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 8 | ok | 1415.267887 | 0.122164 | 0.1266575 | 0.13936806999999998 | 260292.74800001623 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 32 | 64 | ok | 1554.061439 | 0.1482175 | 0.1547729 | 0.15584032 | 215266.8023475383 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 1 | ok | 1377.492898 | 0.1098465 | 0.1116123 | 0.11584892 | 581414.3631150727 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 2 | ok | 1371.621489 | 0.1917015 | 0.1952249 | 0.19854912 | 348452.01282216294 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 4 | ok | 1396.344829 | 0.17995450000000002 | 0.18256425 | 0.18765370999999997 | 355312.52179065073 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 8 | ok | 1386.731793 | 0.14958749999999998 | 0.15401545 | 0.15997274 | 426818.0447998891 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 64 | 64 | ok | 1511.091529 | 0.176809 | 0.18811594999999998 | 0.19153071 | 361934.0397833372 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 1 | ok | 1356.848061 | 0.1636685 | 0.1710371 | 0.17476455 | 778022.7063495527 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 2 | ok | 1370.477634 | 0.2343005 | 0.28759999999999997 | 0.29067308999999997 | 505090.164513391 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 4 | ok | 1426.643485 | 0.2347245 | 0.23912055000000002 | 0.24411630999999998 | 543555.8985242287 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 8 | ok | 1337.072884 | 0.202349 | 0.20737775 | 0.21079462999999998 | 631477.8901337007 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `fp32` | 128 | 64 | ok | 1545.028374 | 0.21739199999999997 | 0.2301155 | 0.23159677 | 592932.4856498757 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 1 | ok | 1367.6295 | 0.157312 | 0.1616637 | 0.16823164 | 6336.25705130366 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 2 | ok | 1349.67128 | 0.17954900000000001 | 0.1882059 | 0.19044334999999998 | 5544.223778177824 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 4 | ok | 1397.455754 | 0.174636 | 0.18679874999999999 | 0.19271554 | 5683.631130975528 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 8 | ok | 1390.320534 | 0.1990075 | 0.2110957 | 0.22316087999999995 | 5010.379000098705 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 1 | 64 | ok | 1495.971811 | 0.3630965 | 0.41062339999999997 | 0.41922908 | 2690.1352632291973 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 1 | ok | 1406.062255 | 0.194926 | 0.20129965 | 0.20538351999999999 | 10197.84741797132 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 2 | ok | 1392.123625 | 0.219326 | 0.22565035 | 0.22855204 | 9156.99093219816 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 4 | ok | 1341.55158 | 0.2612865 | 0.36703979999999997 | 0.37375991999999997 | 6782.098894124518 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 8 | ok | 1387.139195 | 0.2509 | 0.26773245 | 0.27170863 | 7984.200863331639 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 2 | 64 | ok | 1536.274332 | 0.46050800000000003 | 0.5476752500000001 | 0.55490786 | 4288.067209450214 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 1 | ok | 1364.691203 | 0.202426 | 0.20657745 | 0.21048812 | 19737.464095085626 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 2 | ok | 1391.390259 | 0.24517 | 0.25417685 | 0.25721712 | 16298.537043314993 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 4 | ok | 1348.819957 | 0.23128949999999998 | 0.25138865 | 0.25921391 | 17097.74060197041 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 8 | ok | 1357.4008 | 0.25260950000000004 | 0.27074785 | 0.27415292 | 15788.142079069385 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 4 | 64 | ok | 1533.321036 | 0.4602355 | 0.55417745 | 1.7788571899999952 | 7843.155094235118 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 1 | ok | 1382.229321 | 0.224793 | 0.23384739999999998 | 0.2375278 | 35439.565794439884 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 2 | ok | 1362.894185 | 0.25989300000000004 | 0.27237185 | 0.27682526999999996 | 31134.58312739104 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 4 | ok | 1393.58127 | 0.282002 | 0.38256514999999996 | 0.38920936 | 25959.01357305963 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 8 | ok | 1345.453757 | 0.250168 | 0.2662569 | 0.27178907 | 31947.67737190739 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 8 | 64 | ok | 1526.260978 | 0.4546395 | 0.52861655 | 0.53908012 | 17419.024073918674 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 1 | ok | 1388.925825 | 0.234831 | 0.2432277 | 0.24726418 | 67886.64051164125 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 2 | ok | 1365.499349 | 0.298913 | 0.30433255 | 0.30789711 | 55212.1817949425 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 4 | ok | 1366.00474 | 0.2802285 | 0.29719219999999996 | 0.30557660999999997 | 57741.30035284987 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 8 | ok | 1337.996652 | 0.281002 | 0.30234295 | 0.34381746999999985 | 56455.20452415072 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 16 | 64 | ok | 1573.378932 | 0.46686249999999996 | 0.5390142 | 0.55886733 | 34116.23700464557 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 1 | ok | 1390.424785 | 0.2872935 | 0.29769445 | 0.30002112999999997 | 110888.84059063831 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 2 | ok | 1391.438391 | 0.336908 | 0.3896042 | 0.395016 | 94416.24119856225 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 4 | ok | 1398.733908 | 0.31763050000000004 | 0.33262365 | 0.33531071 | 103187.83332894785 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 8 | ok | 1416.052915 | 0.2998245 | 0.33038375 | 0.33643692 | 105434.3638795576 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 32 | 64 | ok | 1547.78899 | 0.48397900000000005 | 0.5949339499999999 | 0.6084906699999999 | 65151.62534166937 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 1 | ok | 1344.83979 | 0.371885 | 0.38180705 | 0.38960117 | 171388.99587311366 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 2 | ok | 1371.817598 | 0.42847599999999997 | 0.56978245 | 0.57465008 | 147500.5932289484 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 4 | ok | 1394.989265 | 0.370933 | 0.41689895 | 0.42252139 | 167408.154001272 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 8 | ok | 1413.425609 | 0.3428635 | 0.38796495 | 0.40001861 | 182244.31251095244 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 64 | 64 | ok | 1520.673658 | 0.499167 | 0.57932745 | 0.5984105799999999 | 127439.76459963883 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 1 | ok | 1336.578852 | 0.541296 | 0.55209405 | 0.55714293 | 236093.88568481614 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 2 | ok | 1364.133342 | 0.5619325 | 0.7302555000000001 | 0.7362695300000001 | 239492.56315718254 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 4 | ok | 1399.52292 | 0.499916 | 0.5988764 | 0.6022050800000001 | 252735.40664297642 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 8 | ok | 1416.305261 | 0.435435 | 0.5028302 | 0.50967632 | 301051.4173712407 | - |
| `full_mlp_capacity_search_hd128_depth4` | `compile` | `bf16` | 128 | 64 | ok | 1486.262589 | 0.5465530000000001 | 0.64135905 | 0.65016427 | 233149.53621092805 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.095893 | 0.10050144999999999 | 0.10748627999999999 | 10344.132745841762 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.110817 | 0.1157584 | 0.12307017999999997 | 8958.727144047374 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.1036195 | 0.10973664999999999 | 0.11606602999999999 | 9588.790473805151 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.10296050000000001 | 0.10593404999999999 | 0.10911862 | 9679.953629150135 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.09979650000000001 | 0.10497059999999998 | 0.10855644 | 9968.634687818248 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1170505 | 0.12214914999999998 | 0.12768649 | 16961.833669205913 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.1165365 | 0.12247744999999999 | 0.12610125 | 17053.68585575822 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.116812 | 0.12375245 | 0.12976221999999998 | 16978.53115670828 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.1193395 | 0.12444594999999999 | 0.12721348 | 16658.268123154576 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.117591 | 0.1264411 | 0.12937644999999998 | 16856.1636585157 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.12273 | 0.1299997 | 0.13449974999999997 | 32348.38956777376 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.122451 | 0.12833930000000002 | 0.1296979 | 32479.311490563297 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.128743 | 0.1351263 | 0.13846735999999998 | 30902.759214662 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1215925 | 0.1298979 | 0.13421571000000002 | 32555.156574025546 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.127134 | 0.13226885 | 0.13470418 | 31327.408067371467 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.1268055 | 0.13336245 | 0.13823934 | 62746.10197684731 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1257015 | 0.13089604999999999 | 0.13559245 | 63269.08515908141 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.1888745 | 0.19497825 | 0.19934868 | 42349.61598426966 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.1322285 | 0.1397723 | 0.1464529 | 60052.756346450355 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.1284925 | 0.13499659999999997 | 0.13857681 | 61979.440489897905 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.135732 | 0.15927434999999998 | 0.17091564999999997 | 110573.05036215438 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.2482425 | 0.25748365 | 0.26274684 | 64187.30368714339 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.196154 | 0.20439715 | 0.20561694 | 81203.204116028 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.19222499999999998 | 0.1985375 | 0.20513481999999997 | 82861.74641902607 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.2705835 | 0.2792181 | 0.28261147999999997 | 59653.17346680532 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.152287 | 0.15845225 | 0.16478411999999998 | 208852.20051899774 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.273864 | 0.28069085 | 0.28584471 | 121108.42670919187 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.21556599999999998 | 0.22321280000000002 | 0.23226076999999998 | 147551.84741368357 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1876325 | 0.1942543 | 0.20137450999999998 | 172447.46495759732 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.252317 | 0.2720376 | 0.2757688 | 124735.86206708373 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.1833765 | 0.19089705 | 0.19477493999999998 | 347354.4023317033 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.268926 | 0.28980629999999996 | 0.29394741999999996 | 234543.50943348653 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.2544425 | 0.26100725 | 0.26353354 | 251089.29598484677 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.222829 | 0.22771070000000002 | 0.23273618999999998 | 297289.15950873337 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.2776695 | 0.29784635 | 0.30574688 | 228787.4856677152 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.279597 | 0.28860979999999997 | 0.29402427999999997 | 455899.65135786514 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.33207200000000003 | 0.41868754999999996 | 0.43051069 | 352860.5633992293 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.344619 | 0.35912285 | 0.36474957 | 395018.25200349867 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.3002235 | 0.3105333 | 0.31799731 | 444055.89553585055 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.374537 | 0.3933911 | 0.40043159 | 343358.6733822616 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.2480095 | 0.25769665 | 0.26224797 | 4017.1099962648905 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.254083 | 0.26247075 | 0.26648999 | 3930.2537175287343 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.2599715 | 0.26928805 | 0.28204332 | 3838.2370393866063 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.29695950000000004 | 0.3185068 | 0.32038379 | 3369.556060988965 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.49679850000000003 | 0.57408765 | 0.59844805 | 1999.5087606876743 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.290844 | 0.30331494999999997 | 0.30995759 | 6855.50239418138 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.29498599999999997 | 0.3038126 | 0.30930259 | 6773.436802547679 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.307897 | 0.32026774999999996 | 0.32412619 | 6472.101488764561 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.3269265 | 0.35114165 | 0.36253698999999995 | 6079.690023356345 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.5708445 | 0.6604751 | 0.6728124499999999 | 3520.49666603685 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.287097 | 0.29628815 | 0.30754429 | 13876.644677272956 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.3158805 | 0.33349175 | 0.3415974 | 12586.58785784456 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.311078 | 0.3260578 | 0.32709924 | 12852.03738529155 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.353885 | 0.37414935 | 0.37665454 | 11319.78762267254 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.5406895 | 0.64824225 | 0.66875792 | 7168.291778148538 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.3034055 | 0.3135309 | 0.31682196 | 26260.549026672445 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.3491285 | 0.3677944 | 0.37252759 | 23022.67192151479 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.33637300000000003 | 0.35663065 | 0.3649595 | 23686.985955393855 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.3492225 | 0.38322219999999996 | 0.39445186 | 22667.624676044226 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.5432185 | 0.6467902 | 0.6521782899999999 | 14580.36045858733 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.340813 | 0.34988095 | 0.35132653 | 46788.214259375025 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.379781 | 0.41959825 | 0.42745725999999995 | 42064.005010664274 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.361616 | 0.38821325 | 0.39117611 | 44009.88836176657 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.37963199999999997 | 0.4164905 | 0.42350466 | 41988.05901589647 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.546985 | 0.6986958 | 0.73293215 | 28032.91625026106 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.39253499999999997 | 0.4030485 | 0.40813373 | 81210.3839251354 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.446983 | 0.52314775 | 0.5296169000000001 | 70342.42781789789 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.39442 | 0.43147254999999995 | 0.43971063 | 79948.52514208727 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.40102150000000003 | 0.4439876 | 0.46436585 | 78626.04137734741 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.5050105 | 0.6911973999999999 | 0.7384040799999999 | 59678.77006829191 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.512453 | 0.5207227 | 0.5246810000000001 | 124566.23215759029 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.514308 | 0.6404666 | 0.64708807 | 120885.55164175991 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.4917265 | 0.5862438 | 0.60091118 | 132576.8003939851 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.44712549999999995 | 0.5284573499999999 | 0.54427945 | 141566.0415980311 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.5877220000000001 | 0.6989271499999999 | 0.7122773299999999 | 108706.38619105495 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.7256739999999999 | 0.7354240999999999 | 0.74293078 | 176131.39930782563 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.6085275 | 0.7591376 | 0.9738683899999999 | 200256.43462254776 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.536995 | 0.6996858499999998 | 0.78496245 | 226447.39246712133 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.493297 | 0.6045953499999999 | 0.69303539 | 249637.4990472819 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.5841825 | 0.7736942 | 3.4069781399999894 | 176542.61697396575 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 1 | ok | 67.248136 | 0.048096 | 0.054099799999999997 | 0.05509645 | 20476.07698022093 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 2 | ok | 66.50889 | 0.054793999999999995 | 0.061339649999999996 | 0.06455016 | 17923.36970821471 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 4 | ok | 67.324167 | 0.059177999999999994 | 0.0646549 | 0.06596475 | 16743.749307227372 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 8 | ok | 67.138098 | 0.06291949999999999 | 0.06724955 | 0.0701796 | 15946.740438414168 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 64 | ok | 65.591793 | 0.047358 | 0.05265195 | 0.05353574 | 20873.706560772964 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 1 | ok | 66.959356 | 0.052803 | 0.0552982 | 0.061342839999999996 | 37501.92197350114 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 2 | ok | 67.271943 | 0.0512965 | 0.05255715 | 0.054463109999999995 | 38872.827932148255 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 4 | ok | 67.740711 | 0.0517705 | 0.0542492 | 0.05920282999999999 | 38357.868944203496 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 8 | ok | 67.490192 | 0.053612999999999994 | 0.05696515 | 0.06104237 | 36789.511751673745 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 64 | ok | 67.249725 | 0.051976 | 0.057219799999999994 | 0.0606988 | 37794.88038109334 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 1 | ok | 67.704756 | 0.0536165 | 0.0570782 | 0.058395449999999995 | 74192.58986670188 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 2 | ok | 67.513246 | 0.056555 | 0.0588214 | 0.06269757 | 70426.30450643838 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 4 | ok | 68.025618 | 0.053826 | 0.057239599999999995 | 0.06111478 | 73730.62589559669 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 8 | ok | 67.760702 | 0.0531535 | 0.05686975 | 0.05909878 | 74269.29228420036 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 64 | ok | 68.247893 | 0.0540735 | 0.05688015 | 0.061692469999999985 | 73333.19155582966 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 1 | ok | 67.711577 | 0.055654499999999996 | 0.06015329999999999 | 0.06448878000000001 | 142010.4058124859 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 2 | ok | 67.815689 | 0.057485 | 0.05983085 | 0.06350713 | 138325.40498220443 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 4 | ok | 68.260847 | 0.0686235 | 0.0712515 | 0.07465591999999999 | 116000.4326816139 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 8 | ok | 67.854549 | 0.0556515 | 0.058527949999999995 | 0.06406745999999998 | 142533.08371039273 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 64 | ok | 66.177033 | 0.055343500000000004 | 0.057182199999999996 | 0.06225026999999999 | 143398.93476101314 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 1 | ok | 68.172039 | 0.063549 | 0.06610035 | 0.07119878999999998 | 250042.5853778222 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 2 | ok | 68.671741 | 0.08430950000000001 | 0.08744815 | 0.09276114999999999 | 191000.57801549922 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 4 | ok | 68.365123 | 0.07300699999999999 | 0.08102575 | 0.08428316999999999 | 213799.64461154075 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 8 | ok | 68.378508 | 0.0762555 | 0.07887365 | 0.08252034999999999 | 208870.9584774978 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 64 | ok | 74.16994 | 0.119326 | 0.12408620000000001 | 0.12902038999999998 | 133980.1448124395 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 1 | ok | 67.884476 | 0.0748605 | 0.07847114999999999 | 0.08109637 | 427980.8532065797 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 2 | ok | 68.733182 | 0.1014075 | 0.10507799999999999 | 0.10796193 | 315097.70983045775 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 4 | ok | 68.270647 | 0.08862 | 0.09127225 | 0.09476256999999999 | 359723.7950770449 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 8 | ok | 68.184114 | 0.093267 | 0.0966661 | 0.10141906999999997 | 352380.7615080399 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 64 | ok | 71.305247 | 0.1261065 | 0.1309258 | 0.13363428 | 255570.02898323862 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 1 | ok | 69.206607 | 0.099551 | 0.10489804999999999 | 0.10753621 | 637631.9624179722 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 2 | ok | 69.382315 | 0.17107050000000001 | 0.17550265 | 0.17745328999999999 | 372878.9306391587 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 4 | ok | 68.902183 | 0.11674000000000001 | 0.14091109999999998 | 0.1433061 | 520410.58443572285 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 8 | ok | 69.413183 | 0.1202255 | 0.12608475 | 0.12685791999999999 | 530352.3113831008 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 64 | ok | 75.292583 | 0.14618550000000002 | 0.15378285 | 0.15559075 | 435161.54216354975 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 1 | ok | 69.396692 | 0.1503785 | 0.15618685 | 0.15937999 | 848093.5850068715 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 2 | ok | 71.307093 | 0.23677399999999998 | 0.2660757 | 0.26866967999999997 | 540851.9872338648 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 4 | ok | 70.606956 | 0.194596 | 0.23503754999999998 | 0.2412456 | 635771.8628036109 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 8 | ok | 70.177109 | 0.141478 | 0.16238645 | 0.164257 | 863968.4392329148 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 64 | ok | 78.49652 | 0.185222 | 0.1907305 | 0.19380863 | 690220.2158383323 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 1 | ok | 67.172342 | 0.1431655 | 0.1484527 | 0.15146228 | 6952.661138003232 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 2 | ok | 67.980587 | 0.1548475 | 0.16067504999999999 | 0.16391305 | 6466.131245135045 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 4 | ok | 66.933383 | 0.158599 | 0.17411545 | 0.17575199 | 6219.14835977425 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 8 | ok | 67.998973 | 0.1944425 | 0.20747629999999997 | 0.21468317999999997 | 5177.972614531318 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 64 | ok | 65.691984 | 0.36897599999999997 | 0.4214169 | 0.47045530999999996 | 2650.4580203495802 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 1 | ok | 67.994022 | 0.174711 | 0.1846551 | 0.19339067999999998 | 11355.065869601603 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 2 | ok | 67.863727 | 0.184679 | 0.19174929999999998 | 0.1959061 | 10793.72638400128 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 4 | ok | 67.42784 | 0.1891335 | 0.20333565 | 0.20602019 | 10470.219658926313 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 8 | ok | 67.779101 | 0.2412065 | 0.26268674999999997 | 0.26833247 | 8299.582389912886 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 64 | ok | 65.934615 | 0.5051445000000001 | 0.6449328499999999 | 3.25098933999999 | 3319.541303127101 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 1 | ok | 67.31133 | 0.1796035 | 0.1877456 | 0.18851036999999998 | 22174.02082850125 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 2 | ok | 67.278077 | 0.211234 | 0.22078675 | 0.22222122 | 19083.792831306808 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 4 | ok | 67.51487 | 0.1984185 | 0.20763765 | 0.21425385 | 20157.175576558213 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 8 | ok | 68.019616 | 0.234561 | 0.24895864999999998 | 0.25790407 | 17112.54544664257 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 64 | ok | 65.492985 | 0.42202 | 0.52566005 | 0.53802133 | 8981.925267597377 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 1 | ok | 67.830487 | 0.1982795 | 0.2055565 | 0.21048171000000002 | 40183.82088865516 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 2 | ok | 68.177729 | 0.24897 | 0.2549383 | 0.25776319 | 33174.818521302135 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 4 | ok | 68.328832 | 0.2269115 | 0.24524625 | 0.25099745 | 35304.66647372667 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 8 | ok | 67.644629 | 0.234794 | 0.2591115 | 0.26602599 | 33632.86199679983 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 64 | ok | 65.158849 | 0.421855 | 0.5237898 | 0.5283298599999999 | 18319.82223543692 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 1 | ok | 68.11887 | 0.2280605 | 0.23560285 | 0.24017924 | 69832.53024468708 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 2 | ok | 68.259739 | 0.27881849999999997 | 0.28932685 | 0.30633620999999994 | 58809.03903751363 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 4 | ok | 68.819802 | 0.2617475 | 0.27254944999999997 | 0.27903726 | 62283.42883978118 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 8 | ok | 68.904863 | 0.2626805 | 0.2971826 | 0.30277572 | 59953.64234489787 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 64 | ok | 74.471284 | 0.42645 | 0.52421805 | 0.5903336499999998 | 34769.00796675318 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 1 | ok | 68.094756 | 0.27932900000000005 | 0.29231399999999996 | 0.29951601 | 113987.71782340453 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 2 | ok | 68.989679 | 0.355255 | 0.3645062 | 0.37055224 | 95885.30966340541 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 4 | ok | 68.835753 | 0.299134 | 0.33264894999999994 | 0.3472018 | 108789.94911418123 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 8 | ok | 68.635762 | 0.2934955 | 0.331583 | 0.34215629999999997 | 106322.11214191875 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 64 | ok | 74.776685 | 0.420505 | 0.6390613999999999 | 0.6431538 | 74810.71369841679 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 1 | ok | 68.367765 | 0.39683650000000004 | 0.4049357 | 0.40726376999999997 | 160891.1276889307 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 2 | ok | 69.538145 | 0.413293 | 0.5134801 | 0.6522214599999999 | 142747.3230081518 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 4 | ok | 69.270349 | 0.3424855 | 0.4065929 | 0.41218399 | 176791.16521510071 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 8 | ok | 68.715623 | 0.346263 | 0.3764558 | 0.40518283999999993 | 187997.3889512642 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 64 | ok | 75.119949 | 0.415106 | 0.5269518 | 0.53154762 | 146568.45236813853 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 1 | ok | 69.650373 | 0.6282544999999999 | 0.6359601500000001 | 0.63845504 | 203709.71950635788 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 2 | ok | 70.614206 | 0.536394 | 0.7122999 | 0.71620413 | 235837.60193251228 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 4 | ok | 69.634976 | 0.4772145 | 0.6178752000000001 | 0.62203742 | 267470.45997730846 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 8 | ok | 74.89795 | 0.41264100000000004 | 0.49212135 | 0.49555565 | 295679.38043342624 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 64 | ok | 77.212852 | 0.5723595 | 0.7140366499999999 | 0.71692124 | 229904.3730721485 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 1 | ok | 1368.826152 | 0.0798365 | 0.08501249999999999 | 0.08896295 | 12467.413298491067 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 2 | ok | 1363.398231 | 0.0884035 | 0.09565269999999998 | 0.09846455 | 11233.213647186327 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 4 | ok | 1329.540895 | 0.08898800000000001 | 0.09101125 | 0.09484119999999999 | 11208.236708992705 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 8 | ok | 1387.814574 | 0.09156500000000001 | 0.09470619999999999 | 0.09897784999999999 | 10885.834807456797 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 64 | ok | 1607.957452 | 0.078071 | 0.0817319 | 0.08657503999999999 | 12702.80343250073 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 1 | ok | 1299.076754 | 0.08129 | 0.08425415 | 0.08923936999999998 | 24432.43454550785 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 2 | ok | 1317.14941 | 0.085743 | 0.08875229999999999 | 0.09149803 | 23250.56365178933 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 4 | ok | 1387.938587 | 0.082026 | 0.0843294 | 0.08752001999999999 | 24279.953693272317 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 8 | ok | 1408.925975 | 0.0853005 | 0.0876216 | 0.09464118999999999 | 23321.48291048375 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 64 | ok | 1581.923385 | 0.087344 | 0.09104785 | 0.09732490999999999 | 22743.917139361078 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 1 | ok | 1382.648187 | 0.08966299999999999 | 0.0924971 | 0.09706414 | 44476.25979561825 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 2 | ok | 1378.388065 | 0.0849985 | 0.08840554999999999 | 0.09204867 | 46899.948574206384 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 4 | ok | 1333.825644 | 0.087519 | 0.08943795 | 0.09416107999999998 | 45532.861977146145 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 8 | ok | 1431.558713 | 0.0866635 | 0.08928734999999999 | 0.09418104 | 45955.7239577529 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 64 | ok | 1545.25915 | 0.0892995 | 0.0916671 | 0.09726632999999998 | 44595.7131032908 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 1 | ok | 1333.100888 | 0.0884515 | 0.09117109999999999 | 0.09453233999999999 | 90034.6093038164 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 2 | ok | 1382.699876 | 0.0948925 | 0.0965606 | 0.10118329999999999 | 84220.04761380392 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 4 | ok | 1419.939256 | 0.1536525 | 0.1558207 | 0.16290631 | 51946.27093304853 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 8 | ok | 1383.322357 | 0.09374450000000001 | 0.0954751 | 0.09916844 | 85112.08729946794 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 64 | ok | 1569.544923 | 0.08939749999999999 | 0.0939465 | 0.09913111 | 88930.84448507596 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 1 | ok | 1364.007211 | 0.09773899999999999 | 0.10045275 | 0.10404594999999998 | 163287.49989284258 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 2 | ok | 1365.674797 | 0.2092235 | 0.21334535 | 0.2163958 | 76322.11366461583 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 4 | ok | 1372.762385 | 0.161985 | 0.16997894999999996 | 0.17568150999999999 | 98093.4676249254 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 8 | ok | 1397.31608 | 0.1542535 | 0.15575015 | 0.16111107 | 103641.0124068656 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 64 | ok | 1603.279239 | 0.208968 | 0.23142490000000002 | 0.23397711 | 75277.84346292383 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 1 | ok | 1319.827993 | 0.111711 | 0.11387359999999999 | 0.11825547999999998 | 286038.2751391442 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 2 | ok | 1332.63135 | 0.23698550000000002 | 0.24047359999999998 | 0.24453453 | 144870.55906092006 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 4 | ok | 1390.433602 | 0.1830765 | 0.18737009999999998 | 0.19237173 | 177599.11307002933 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 8 | ok | 1489.036492 | 0.1525605 | 0.17282154999999996 | 0.18860619999999997 | 207036.25657744484 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 64 | ok | 1491.205817 | 0.1986525 | 0.2059297 | 0.21845088999999998 | 163149.65299598334 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 1 | ok | 1388.991308 | 0.1323375 | 0.13523415 | 0.13839153999999998 | 482735.12896721525 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 2 | ok | 1378.071174 | 0.208461 | 0.25126434999999997 | 0.25689483 | 282602.97831746313 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 4 | ok | 1358.244734 | 0.215145 | 0.23557654999999997 | 0.24270477999999998 | 291434.1227281913 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 8 | ok | 1369.071322 | 0.18915500000000002 | 0.19402329999999998 | 0.19737514 | 339731.5759585128 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 64 | ok | 1560.622946 | 0.232172 | 0.2409403 | 0.24351301 | 281775.56646574894 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 1 | ok | 1426.007825 | 0.215289 | 0.221792 | 0.22559113 | 593344.6569452707 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 2 | ok | 1367.003783 | 0.2870565 | 0.3590735 | 0.36407772 | 427984.00142304675 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 4 | ok | 1322.501461 | 0.2520125 | 0.29089195 | 0.29439684 | 490673.5982490313 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 8 | ok | 1358.872647 | 0.23340349999999999 | 0.2668605 | 0.26961625 | 528044.2659508147 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 64 | ok | 1589.936632 | 0.28255549999999996 | 0.29467715 | 0.29651805 | 458769.62145126465 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 1 | ok | 1363.414129 | 0.1989105 | 0.2032138 | 0.21030509 | 5014.849973742245 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 2 | ok | 1330.688867 | 0.2218275 | 0.2262555 | 0.23078678 | 4514.775415205836 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 4 | ok | 1368.177585 | 0.2309765 | 0.2452284 | 0.25321871999999995 | 4320.429305506543 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 8 | ok | 1447.490223 | 0.31352349999999996 | 0.42619915 | 0.43033963999999997 | 2908.619945349358 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 64 | ok | 1584.603047 | 0.47419049999999996 | 0.56926485 | 0.6060209999999999 | 2096.7852591634964 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 1 | ok | 1306.223265 | 0.2385945 | 0.24692545 | 0.25538851999999995 | 8335.080921966639 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 2 | ok | 1368.840841 | 0.2597025 | 0.27899240000000003 | 0.28164144999999996 | 7618.50024794409 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 4 | ok | 1384.565082 | 0.2766235 | 0.299325 | 0.30509399 | 7181.266660538652 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 8 | ok | 1337.957996 | 0.299264 | 0.32190635 | 0.32582629 | 6661.37398299138 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 64 | ok | 1551.055613 | 0.5014365000000001 | 0.59872615 | 0.6395832099999998 | 3905.970784119729 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 1 | ok | 1313.325755 | 0.26046199999999997 | 0.2704595 | 0.27299837 | 15328.592914879098 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 2 | ok | 1392.909917 | 0.283485 | 0.30519815 | 0.31047441 | 13853.237006235135 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 4 | ok | 1383.769765 | 0.289948 | 0.30850645 | 0.31870598 | 13759.166700835263 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 8 | ok | 1383.692544 | 0.296269 | 0.31845144999999997 | 0.32767234 | 13438.761311741366 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 64 | ok | 1563.368981 | 0.5136065000000001 | 0.6179802999999999 | 2.1845654999999935 | 6788.9690387030305 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 1 | ok | 1326.38565 | 0.2656085 | 0.2748693 | 0.27965433 | 29988.4559438844 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 2 | ok | 1400.668477 | 0.30703349999999996 | 0.33235915 | 0.33779641 | 25553.122898255642 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 4 | ok | 1418.581477 | 0.302189 | 0.32483025 | 0.33264237999999996 | 26387.38260913162 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 8 | ok | 1348.834818 | 0.31550599999999995 | 0.34176 | 0.34445439 | 25202.948316502912 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 64 | ok | 1551.204641 | 0.49472499999999997 | 0.6043232 | 3.3917625099999893 | 12927.037827130407 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 1 | ok | 1341.456528 | 0.28837 | 0.2986642 | 0.30137332 | 55242.14080420679 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 2 | ok | 1375.81189 | 0.345645 | 0.39064695 | 0.39890746 | 44952.651652968 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 4 | ok | 1376.073452 | 0.3768405 | 0.38929035 | 0.40381738999999994 | 46869.41869441836 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 8 | ok | 1330.565006 | 0.3299175 | 0.3431884 | 0.35266363 | 48785.83342333303 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 64 | ok | 1586.921414 | 0.514447 | 0.6158855999999999 | 0.62568728 | 31232.095225409063 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 1 | ok | 1400.157432 | 0.36099000000000003 | 0.37266815 | 0.37372405000000003 | 88466.80882914235 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 2 | ok | 1372.11827 | 0.3997175 | 0.46304034999999993 | 0.47502074 | 76622.17280943434 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 4 | ok | 1308.297484 | 0.3832405 | 0.3990315 | 0.4232116799999999 | 86129.29201775792 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 8 | ok | 1450.375198 | 0.3693735 | 0.38774795 | 0.39519185 | 88331.3512146168 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 64 | ok | 1556.34771 | 0.5153235 | 0.63733275 | 0.72131585 | 58422.76428404637 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 1 | ok | 1384.507066 | 0.483211 | 0.49366385 | 0.49668781 | 132112.0837266936 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 2 | ok | 1375.342418 | 0.48839449999999995 | 0.6348735999999999 | 0.6383832899999999 | 124791.4909673969 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 4 | ok | 1354.581253 | 0.41884449999999995 | 0.47773235000000003 | 1.8523310799999952 | 133651.59118484225 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 8 | ok | 1432.362928 | 0.4315945 | 0.4907902 | 0.49615417 | 150584.40159241122 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 64 | ok | 1504.196273 | 0.5764205 | 0.65927175 | 1.0107836899999987 | 109208.21111967563 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 1 | ok | 1371.076918 | 0.7244665 | 0.73489075 | 0.7382894999999999 | 176375.07659225492 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 2 | ok | 1375.760556 | 0.5942945 | 0.7902659 | 0.79899995 | 213818.2160290964 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 4 | ok | 1386.66121 | 0.493422 | 0.62219915 | 0.7722382999999999 | 243263.71417243377 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 8 | ok | 1403.588036 | 0.5017875 | 0.60708615 | 0.6345226399999999 | 256119.7206021983 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 64 | ok | 1527.609688 | 0.5954999999999999 | 0.80354105 | 0.81808184 | 204583.78276501183 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.05846 | 0.0623695 | 0.06767441999999999 | 16927.731788892226 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.061796000000000004 | 0.06683364999999998 | 0.068899 | 16033.827527965399 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.055342 | 0.0594257 | 0.06533451999999998 | 17801.139201704354 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0564615 | 0.0606339 | 0.07004745 | 17450.045754019968 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.060325500000000004 | 0.06501499999999999 | 0.06630375999999999 | 16397.517809344095 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.06796150000000001 | 0.07155585 | 0.07920098999999999 | 29116.921619285185 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0681925 | 0.07227584999999999 | 0.07974574999999999 | 29035.878182876666 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0674255 | 0.0716595 | 0.07963920999999999 | 29338.70270711144 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.067917 | 0.07208935 | 0.07698322999999999 | 29168.3888169564 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.067854 | 0.07121505 | 0.0740683 | 29848.064397795897 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0726125 | 0.07464114999999999 | 0.08397379999999997 | 54748.8015487341 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0701515 | 0.0741067 | 0.08015636999999998 | 56484.58578017499 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.067857 | 0.07169805 | 0.07723006999999998 | 58400.9811364831 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.068849 | 0.07179785 | 0.08038270999999997 | 57578.330280242364 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.067637 | 0.07210855 | 0.07711085 | 58593.91021772325 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.07006499999999999 | 0.07307269999999999 | 0.07503669 | 113588.99186362051 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0712285 | 0.0756982 | 0.08234722999999998 | 111169.72225912439 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.068272 | 0.07141934999999999 | 0.07853072999999998 | 115953.12015352194 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.07321849999999999 | 0.0767847 | 0.08061518 | 108619.5004589174 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.059873499999999996 | 0.0621411 | 0.06584565999999999 | 132607.4841674952 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0710685 | 0.0754016 | 0.08194945 | 222958.91563931378 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.073853 | 0.07670285 | 0.08345751999999998 | 215058.50397777586 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0692165 | 0.07360585 | 0.07810033 | 229128.5382458492 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.07172100000000001 | 0.07581975 | 0.08084071 | 221579.33440337685 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.0644905 | 0.06799795 | 0.06975132999999999 | 246778.91817057854 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0728325 | 0.07676904999999999 | 0.08263584999999998 | 435456.61981153436 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0735105 | 0.07718509999999999 | 0.08443390999999997 | 431615.8618829242 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0708945 | 0.07484369999999999 | 0.07957320999999999 | 447606.5359506379 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0706505 | 0.0752282 | 0.08259768999999997 | 448161.9478014576 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.06460250000000001 | 0.06716985 | 0.07478997999999998 | 490723.7869154637 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0799835 | 0.08359615000000001 | 0.09193188999999999 | 793972.1633359534 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0915585 | 0.09738495 | 0.10970696999999999 | 689632.7296245316 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.08751400000000001 | 0.09407305 | 0.10170553999999998 | 723472.1962852415 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.07571549999999999 | 0.08161009999999999 | 0.08597169999999998 | 838201.6697501139 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.0785265 | 0.08167714999999999 | 0.08747180999999998 | 809139.4321712322 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.08806149999999999 | 0.09337109999999998 | 0.10237229 | 1438477.6590933816 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1177405 | 0.12308024999999999 | 0.13557420999999997 | 1077149.848037719 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.11425199999999999 | 0.12115390000000001 | 0.13116426 | 1110782.504620161 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.10832 | 0.11187395 | 0.11729904 | 1175372.7814150054 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.12460750000000001 | 0.1299683 | 0.13166455999999999 | 1022616.4408601739 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.07370499999999999 | 0.07679309999999999 | 0.08062122999999999 | 13497.535350045082 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.071106 | 0.07436849999999999 | 0.07700392 | 13977.86967761162 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0733885 | 0.07636119999999999 | 0.07871042999999998 | 13536.705182202697 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.07086600000000001 | 0.0740318 | 0.07716322999999999 | 14031.267275997834 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.071591 | 0.07528375 | 0.07746662 | 13867.798279006232 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1104845 | 0.115107 | 0.12311260999999998 | 17970.553092291906 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.11352 | 0.11933964999999999 | 0.13064077999999996 | 17471.326059670868 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1271625 | 0.13990765 | 0.14444013 | 15615.435545728242 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.130446 | 0.1413984 | 0.14423011 | 15180.781858844231 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.2428455 | 0.261424 | 0.26243204 | 8173.467127541233 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1184865 | 0.1253704 | 0.13047671 | 33504.204777699604 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.1249085 | 0.14259070000000001 | 0.14648549 | 31275.93459138695 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.126761 | 0.1359419 | 0.13848765999999998 | 31165.07517016131 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.13079600000000002 | 0.1426711 | 0.14391742999999999 | 30392.060621219796 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.22331299999999998 | 0.24329635 | 0.25455512 | 17600.031539256517 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.119982 | 0.12690885 | 0.13108522 | 66107.45270631516 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.126881 | 0.14368620000000001 | 0.14513317 | 61606.759740105714 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.130185 | 0.15958624999999999 | 0.17054076999999995 | 59194.66253566849 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.13508599999999998 | 0.14618535 | 0.15303046 | 58839.294881657996 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.2229225 | 0.23815845 | 0.24234498999999998 | 36369.915133530776 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.122573 | 0.12982115 | 0.13285627 | 129536.88781474021 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.128169 | 0.14536079999999998 | 0.15678917999999997 | 121981.41735088074 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.13664749999999998 | 0.16459269999999998 | 0.17524805999999996 | 113056.70712448037 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.152081 | 0.21011034999999995 | 0.23511626999999996 | 99300.44080706929 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.2453345 | 0.25597395 | 0.26894364 | 65336.23045068524 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1283635 | 0.13886904999999997 | 0.14517803999999998 | 246825.78187851392 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.1451445 | 0.15191875 | 0.16010564 | 219313.06487677965 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.14119900000000002 | 0.17265325 | 0.17618284 | 217727.74696552154 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.14834999999999998 | 0.19573225 | 0.23822942999999994 | 204527.94184043448 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.2410775 | 0.25583685 | 0.25753262 | 133042.98918327238 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.140009 | 0.1466277 | 0.15043734 | 454243.87274698593 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1651725 | 0.1834249 | 0.18858433 | 386425.7801061005 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1624275 | 0.17230104999999998 | 0.17528544 | 394724.60434409225 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1724275 | 0.23680664999999998 | 0.26405973999999993 | 351529.1021606187 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.2500845 | 0.26291529999999996 | 0.26729902 | 261511.61887950444 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.16678500000000002 | 0.18659544999999997 | 0.2016872 | 755798.8073022447 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.20616099999999998 | 0.23675184999999999 | 0.24642222999999996 | 606182.4739954822 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2081285 | 0.2442182 | 0.2464066 | 610864.803210858 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.21183449999999998 | 0.23479445 | 0.24139173999999997 | 599559.979184027 | - |
| `full_mlp_capacity_search_hd256_depth2` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.280327 | 1.01271995 | 1.05832022 | 333401.0380545507 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 1 | ok | 53.578785 | 0.0359525 | 0.0443731 | 0.04681157999999999 | 27094.029830526848 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 2 | ok | 52.847505 | 0.0380465 | 0.044974799999999995 | 0.048324559999999996 | 25607.10607436406 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 4 | ok | 53.164759 | 0.0379915 | 0.04520299999999999 | 0.050091819999999995 | 25668.22079187488 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 8 | ok | 55.04288 | 0.0341975 | 0.0379326 | 0.044306439999999996 | 28666.716737865238 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 1 | 64 | ok | 52.067539 | 0.0341625 | 0.040503199999999996 | 0.04841839999999997 | 28471.696286721373 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 1 | ok | 53.283529 | 0.03959 | 0.0437695 | 0.0456335 | 50543.80075229392 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 2 | ok | 53.664632 | 0.039482 | 0.042305 | 0.04286054 | 51071.06231895308 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 4 | ok | 53.732002 | 0.037781999999999996 | 0.043825249999999996 | 0.04537418 | 50439.70815584861 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 8 | ok | 53.48604 | 0.0400365 | 0.04307245 | 0.04494636999999999 | 50523.29502825515 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 2 | 64 | ok | 54.374445 | 0.0368675 | 0.040353850000000004 | 0.043664079999999994 | 53567.45884010382 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 1 | ok | 53.605023 | 0.04157 | 0.04322215 | 0.04734165999999999 | 95549.58689136109 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 2 | ok | 53.672244 | 0.040091 | 0.0418769 | 0.04306671 | 99135.09585620256 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 4 | ok | 53.584628 | 0.041883000000000004 | 0.0433925 | 0.04504411 | 94999.42050353493 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 8 | ok | 53.664551 | 0.042214 | 0.04645285 | 0.05329083999999998 | 93390.82472503406 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 4 | 64 | ok | 52.749697 | 0.03651 | 0.037960549999999996 | 0.040894889999999996 | 108744.88287360528 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 1 | ok | 53.512591 | 0.041091 | 0.043127399999999996 | 0.046909179999999995 | 192991.51319820713 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 2 | ok | 54.238053 | 0.042959 | 0.045868099999999995 | 0.04987066999999999 | 183655.82531025208 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 4 | ok | 53.698484 | 0.042244500000000004 | 0.0442344 | 0.04835196999999999 | 187890.19509108682 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 8 | ok | 54.082853 | 0.042883500000000005 | 0.0448282 | 0.04687492 | 185140.10014228016 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 8 | 64 | ok | 54.320774 | 0.0373275 | 0.03928765 | 0.04342984 | 212490.3980901363 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 1 | ok | 53.88148 | 0.043833 | 0.0451975 | 0.04744046 | 363125.1824136658 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 2 | ok | 53.650209 | 0.039222 | 0.04088415 | 0.045877039999999994 | 403972.259224959 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 4 | ok | 54.395612 | 0.042495000000000005 | 0.0444876 | 0.04594578 | 373615.63730888395 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 8 | ok | 53.876924 | 0.043493500000000004 | 0.0450969 | 0.05086406999999998 | 364902.07624720107 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 16 | 64 | ok | 51.511599 | 0.0378965 | 0.040286949999999995 | 0.04277555999999999 | 418083.5780428776 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 1 | ok | 54.040618 | 0.0452525 | 0.0467811 | 0.0480007 | 703857.9333147398 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 2 | ok | 54.236165 | 0.045479 | 0.04815245 | 0.04918192 | 697865.6216128983 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 4 | ok | 53.872447 | 0.045005 | 0.0462203 | 0.04787415999999999 | 708493.4639263911 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 8 | ok | 54.534362 | 0.046533 | 0.0487017 | 0.05523546999999998 | 681283.6064428991 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 32 | 64 | ok | 52.146768 | 0.039264499999999994 | 0.0410435 | 0.04539978999999999 | 805858.1860837358 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 1 | ok | 53.954156 | 0.048748 | 0.0503297 | 0.057872359999999984 | 1301877.2663343896 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 2 | ok | 54.184863 | 0.066648 | 0.06880505 | 0.07356634 | 955086.7472461415 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 4 | ok | 54.741756 | 0.0552455 | 0.0604501 | 0.06384772999999999 | 1143501.179056919 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 8 | ok | 54.545695 | 0.049046 | 0.05110595 | 0.05256892 | 1297531.2035980541 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 64 | 64 | ok | 52.359784 | 0.042662 | 0.043784300000000005 | 0.045949239999999995 | 1493714.0312494312 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 1 | ok | 54.698186 | 0.0571655 | 0.05831385 | 0.06221835999999999 | 2232192.6828723857 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 2 | ok | 55.037745 | 0.0874595 | 0.09053595 | 0.09278407 | 1458393.292666948 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 4 | ok | 55.461612 | 0.0845005 | 0.0905642 | 0.09382079 | 1499731.6886275816 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 8 | ok | 55.701838 | 0.06999050000000001 | 0.0720861 | 0.07337805 | 1824519.4315595678 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `fp32` | 128 | 64 | ok | 58.660069 | 0.090973 | 0.09725104999999999 | 0.09917741999999999 | 1394606.2296624463 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 1 | ok | 54.069994 | 0.0440095 | 0.04884205 | 0.04930022 | 22499.64788051067 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 2 | ok | 53.603553 | 0.0434345 | 0.0493114 | 0.050717939999999996 | 22734.061944862624 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 4 | ok | 53.758939 | 0.04511 | 0.0498104 | 0.059486389999999986 | 21746.38238055908 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 8 | ok | 53.496912 | 0.044519 | 0.052180749999999984 | 0.05957190999999999 | 21767.67448097157 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 1 | 64 | ok | 51.667637 | 0.043826000000000004 | 0.04687375 | 0.05224986999999999 | 22475.5263993038 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 1 | ok | 53.428648 | 0.0759025 | 0.0823782 | 0.08653652999999999 | 26021.97399572095 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 2 | ok | 53.462017 | 0.07636 | 0.079567 | 0.08431377 | 26159.91766950711 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 4 | ok | 53.569231 | 0.0811795 | 0.0914367 | 0.09501341999999999 | 24147.24604280969 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 8 | ok | 53.843935 | 0.0939985 | 0.10422605 | 0.10576819999999999 | 21102.883096991594 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 2 | 64 | ok | 51.676806 | 0.18350149999999998 | 0.1979095 | 0.20647585999999998 | 10813.39082741147 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 1 | ok | 54.139704 | 0.080433 | 0.0860687 | 0.09089871 | 49095.826707442386 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 2 | ok | 54.090472 | 0.0847455 | 0.1013784 | 0.10297390000000001 | 45704.45533886312 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 4 | ok | 53.88049 | 0.085605 | 0.09506629999999999 | 0.10009886 | 45978.73437556396 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 8 | ok | 54.280731 | 0.0941385 | 0.10817915 | 0.10988361999999999 | 42174.2869277002 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 4 | 64 | ok | 51.690399 | 0.19238850000000002 | 0.20511245 | 0.21133573999999997 | 20725.854664296592 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 1 | ok | 53.63828 | 0.076539 | 0.0805963 | 0.0834481 | 103809.30852879111 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 2 | ok | 53.106089 | 0.08295849999999999 | 0.1008958 | 0.10859974999999998 | 93078.51847852924 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 4 | ok | 53.660607 | 0.086385 | 0.11353809999999999 | 0.11840732 | 87416.46537204339 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 8 | ok | 53.658868 | 0.09 | 0.10183624999999998 | 0.11274460999999997 | 88372.26462986271 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 8 | 64 | ok | 52.139526 | 0.1923485 | 0.20911375000000001 | 0.21171292 | 41095.152949995005 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 1 | ok | 54.133587 | 0.07968700000000001 | 0.0838675 | 0.08816521 | 199214.7452778012 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 2 | ok | 53.640426 | 0.0918755 | 0.10614214999999999 | 0.11130013999999999 | 169592.9578220194 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 4 | ok | 53.678673 | 0.0885585 | 0.12222095 | 0.12833155999999998 | 169526.61599657385 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 8 | ok | 54.145155 | 0.10157150000000001 | 0.15729079999999998 | 0.1737062 | 145259.13230546762 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 16 | 64 | ok | 52.505735 | 0.1937595 | 0.20559135 | 0.21017394 | 82309.4805603012 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 1 | ok | 54.475243 | 0.08651149999999999 | 0.09198635 | 0.09656214 | 366742.08564848825 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 2 | ok | 53.943644 | 0.100222 | 0.10634645 | 0.11041698999999998 | 316743.55946513894 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 4 | ok | 54.602933 | 0.102876 | 0.13112755 | 0.14119552999999999 | 297255.6061942866 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 8 | ok | 54.07835 | 0.102418 | 0.15935589999999994 | 0.17715634999999996 | 287234.62886323844 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 32 | 64 | ok | 52.050557 | 0.19854650000000001 | 0.20989465 | 0.21259237 | 164782.83836173316 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 1 | ok | 54.593807 | 0.09618399999999999 | 0.1002871 | 0.10462463 | 661005.1574927412 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 2 | ok | 55.132645 | 0.129442 | 0.15094245 | 0.16386899999999996 | 482922.72074435296 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 4 | ok | 54.116438 | 0.116688 | 0.12719345 | 0.12929858 | 545228.3995842633 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 8 | ok | 54.574737 | 0.122173 | 0.17831484999999997 | 0.20198652 | 492503.1023847769 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 64 | 64 | ok | 54.45521 | 0.187849 | 0.2065198 | 0.21344056 | 339304.45555525157 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 1 | ok | 54.341938 | 0.1196575 | 0.12667005 | 0.13921017999999996 | 1059345.7050566378 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 2 | ok | 55.073901 | 0.1674115 | 0.18875085 | 0.19168051 | 745496.7337924351 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 4 | ok | 55.492477 | 0.16512349999999998 | 0.19964525 | 0.20537973999999998 | 745105.5587552406 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 8 | ok | 55.208154 | 0.1552195 | 0.17454625 | 0.19463186999999993 | 815423.8439360305 | - |
| `full_mlp_capacity_search_hd256_depth2` | `jit` | `bf16` | 128 | 64 | ok | 61.864517 | 0.236522 | 0.8043422999999998 | 1.0277825399999998 | 381817.42469158396 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 1 | ok | 1405.488837 | 0.044411 | 0.04791714999999999 | 0.050263129999999996 | 22283.685578979177 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 2 | ok | 1377.853449 | 0.047538 | 0.049108349999999995 | 0.050307769999999995 | 20978.33734928638 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 4 | ok | 1314.367715 | 0.045759999999999995 | 0.04762625 | 0.04948983999999999 | 21756.75340504069 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 8 | ok | 1345.577886 | 0.044057 | 0.0470287 | 0.052632289999999984 | 22413.262913849692 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 1 | 64 | ok | 1568.223266 | 0.0453035 | 0.0498862 | 0.05049044 | 21846.76851338857 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 1 | ok | 1363.582558 | 0.046782000000000004 | 0.0488181 | 0.05528941999999998 | 42381.75279062651 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 2 | ok | 1363.981351 | 0.05105 | 0.0522633 | 0.05290496 | 39174.63739955623 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 4 | ok | 1379.882701 | 0.044955 | 0.04663235 | 0.04842058 | 44274.136598993646 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 8 | ok | 1430.414039 | 0.049029500000000004 | 0.050643499999999994 | 0.05163339 | 40672.37950909251 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 2 | 64 | ok | 1507.637462 | 0.046218 | 0.048929749999999994 | 0.05044514 | 43056.75396854101 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 1 | ok | 1408.90683 | 0.0467765 | 0.0482363 | 0.04980371 | 85372.81883119489 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 2 | ok | 1380.416356 | 0.0492885 | 0.051205 | 0.05509702999999999 | 80594.36734026096 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 4 | ok | 1421.960192 | 0.0461735 | 0.048203949999999995 | 0.05173237999999999 | 86306.33484182422 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 8 | ok | 1392.077818 | 0.046638 | 0.0485235 | 0.04930405 | 85483.74825720009 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 4 | 64 | ok | 1531.323373 | 0.0455795 | 0.04748205 | 0.04805586 | 87434.94293777038 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 1 | ok | 1365.400079 | 0.047705 | 0.049326949999999994 | 0.05043904 | 167038.95473703684 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 2 | ok | 1375.680521 | 0.051782499999999995 | 0.05341325 | 0.05417354 | 154050.03314001337 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 4 | ok | 1408.76688 | 0.0516085 | 0.0540112 | 0.05755103999999999 | 153743.14593837512 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 8 | ok | 1408.256634 | 0.048253 | 0.05158084999999999 | 0.0562137 | 164055.85445620815 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 8 | 64 | ok | 1615.684225 | 0.048904500000000004 | 0.0517228 | 0.05648845999999998 | 161955.3845306695 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 1 | ok | 1412.156535 | 0.0475575 | 0.0492046 | 0.04955911 | 335686.49566796573 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 2 | ok | 1416.34081 | 0.046766 | 0.04823575 | 0.049591539999999996 | 340766.44336525607 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 4 | ok | 1417.918386 | 0.051643499999999995 | 0.0537753 | 0.05670300999999999 | 308170.08985854604 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 8 | ok | 1388.885143 | 0.046383999999999995 | 0.0485573 | 0.05486098999999998 | 342506.9539614997 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 16 | 64 | ok | 1541.056643 | 0.051449499999999995 | 0.0540798 | 0.05972934 | 308086.065382795 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 1 | ok | 1383.852869 | 0.0488285 | 0.05064085 | 0.05190991 | 653561.3990298698 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 2 | ok | 1377.116586 | 0.053780999999999995 | 0.05601199999999999 | 0.06129654999999999 | 591413.2707223818 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 4 | ok | 1410.058046 | 0.04919900000000001 | 0.050486449999999995 | 0.054699039999999984 | 647583.7234687983 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 8 | ok | 1426.736195 | 0.052042000000000005 | 0.05389335 | 0.06175836999999998 | 609806.5274452956 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 32 | 64 | ok | 1567.784989 | 0.049415 | 0.05093645 | 0.05165403 | 645611.6565184584 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 1 | ok | 1389.817161 | 0.051715 | 0.0540624 | 0.05489654 | 1231155.6242266803 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 2 | ok | 1373.88189 | 0.07446449999999999 | 0.0758572 | 0.08484510999999997 | 854989.5050038261 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 4 | ok | 1376.960365 | 0.071756 | 0.07383050000000001 | 0.07881247 | 888971.8595957846 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 8 | ok | 1341.370574 | 0.052525 | 0.0541889 | 0.05499665 | 1217755.6383037425 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 64 | 64 | ok | 1539.147161 | 0.0540435 | 0.05550905 | 0.056189619999999996 | 1185294.9347161774 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 1 | ok | 1372.651829 | 0.0600715 | 0.06263869999999999 | 0.06594296999999999 | 2118346.736289663 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 2 | ok | 1363.155599 | 0.09489149999999999 | 0.09744415 | 0.10124383999999999 | 1345363.9703540641 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 4 | ok | 1391.385737 | 0.11057 | 0.11404044999999999 | 0.11611716 | 1156958.8180508714 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 8 | ok | 1341.905837 | 0.07476250000000001 | 0.07713765 | 0.0793154 | 1710003.224959207 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `fp32` | 128 | 64 | ok | 1507.245189 | 0.099714 | 0.10370415000000001 | 0.11108444999999997 | 1275536.387963959 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 1 | ok | 1395.747149 | 0.053717 | 0.055703749999999996 | 0.05822776999999999 | 18541.578748680768 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 2 | ok | 1373.4094 | 0.0521225 | 0.0545417 | 0.05799290999999999 | 19022.726451291357 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 4 | ok | 1364.865208 | 0.056833499999999995 | 0.058202000000000004 | 0.05856533 | 17557.147637703343 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 8 | ok | 1380.686877 | 0.0568395 | 0.0590471 | 0.06378568999999998 | 17462.197834128678 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 1 | 64 | ok | 1539.084967 | 0.052974999999999994 | 0.0552036 | 0.05806583999999999 | 18737.38505551138 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 1 | ok | 1375.404732 | 0.090528 | 0.09809339999999998 | 0.10976899999999998 | 21805.067061483747 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 2 | ok | 1407.679314 | 0.095826 | 0.09966245 | 0.10348995 | 20761.23561739451 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 4 | ok | 1385.842118 | 0.0964315 | 0.10222264999999998 | 0.11032141999999999 | 20565.527322742633 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 8 | ok | 1387.752037 | 0.10529250000000001 | 0.1159606 | 0.11705937999999999 | 18873.789647462145 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 2 | 64 | ok | 1527.392455 | 0.368271 | 0.6262831999999999 | 3.5589827899999893 | 4185.806966656196 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 1 | ok | 1401.949047 | 0.09017349999999999 | 0.09628229999999999 | 0.10374926999999999 | 43967.13720296902 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 2 | ok | 1385.929472 | 0.0965835 | 0.11314339999999999 | 0.11845626 | 40079.63020929984 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 4 | ok | 1410.91688 | 0.0981825 | 0.10675734999999997 | 0.11463680999999998 | 40436.82282221931 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 8 | ok | 1405.249489 | 0.1148305 | 0.12281935 | 0.12423551999999999 | 35004.40267874692 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 4 | 64 | ok | 1527.154341 | 0.2153465 | 0.23348555 | 0.23985726999999998 | 18519.943525284216 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 1 | ok | 1411.268523 | 0.096168 | 0.0989958 | 0.10332449999999999 | 82827.20732954523 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 2 | ok | 1378.18236 | 0.107048 | 0.12833465 | 0.13189018 | 72268.52526859952 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 4 | ok | 1332.68143 | 0.100076 | 0.12951954999999998 | 0.13751399 | 76335.60107320221 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 8 | ok | 1411.242896 | 0.108575 | 0.11639845 | 0.11790265 | 73857.09833186357 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 8 | 64 | ok | 1577.333153 | 0.214216 | 0.2281985 | 0.25997662999999993 | 37729.23220352204 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 1 | ok | 1365.836199 | 0.0969835 | 0.10252574999999998 | 0.10599046 | 163888.08082960147 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 2 | ok | 1361.869775 | 0.111492 | 0.1271766 | 0.13505860999999997 | 139743.70306140024 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 4 | ok | 1351.31683 | 0.104101 | 0.1269191 | 0.14354391999999994 | 146681.3348001467 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 8 | ok | 1352.132824 | 0.112056 | 0.17652279999999998 | 0.18458553 | 131486.66881318652 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 16 | 64 | ok | 1598.655157 | 0.210542 | 0.2247518 | 0.22895697 | 76039.86656144017 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 1 | ok | 1364.829243 | 0.1063075 | 0.10930145 | 0.11682215 | 299763.89845947584 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 2 | ok | 1345.46739 | 0.120102 | 0.12546474999999999 | 0.13015287999999997 | 264125.7766948621 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 4 | ok | 1327.587552 | 0.1192585 | 0.147116 | 0.15184957 | 256848.380474643 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 8 | ok | 1388.896463 | 0.13005 | 0.20356894999999994 | 0.21978634 | 228301.5109564749 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 32 | 64 | ok | 1555.828462 | 0.20369500000000001 | 0.23460069999999997 | 2.285817259999992 | 111106.15762825019 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 1 | ok | 1358.29813 | 0.1136125 | 0.11879274999999999 | 0.12227019 | 560956.6133366383 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 2 | ok | 1315.777052 | 0.1498605 | 0.16936749999999998 | 0.17271969 | 419602.2983191388 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 4 | ok | 1385.266419 | 0.133684 | 0.1368451 | 0.14010941 | 477933.93937348237 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 8 | ok | 1382.977345 | 0.137012 | 0.1977862 | 0.2185732 | 438634.5471036618 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 64 | 64 | ok | 1589.584276 | 0.227027 | 0.24810015 | 0.25754541 | 283940.3430691449 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 1 | ok | 1359.249502 | 0.14555099999999999 | 0.15082355 | 0.16371958999999997 | 874162.4089949126 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 2 | ok | 1404.534602 | 0.187874 | 0.2115883 | 0.21651737999999998 | 674862.2358375678 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 4 | ok | 1432.053549 | 0.15491549999999998 | 0.1905645 | 0.20000436999999996 | 793359.8753532466 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 8 | ok | 1387.999703 | 0.182252 | 0.1906487 | 0.19431585 | 709826.7978249133 | - |
| `full_mlp_capacity_search_hd256_depth2` | `compile` | `bf16` | 128 | 64 | ok | 1552.574102 | 0.23707650000000002 | 0.8147133999999998 | 0.9564169299999998 | 386750.3908747212 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0782435 | 0.0823333 | 0.09041019999999997 | 12667.733458727003 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.07943449999999999 | 0.10800255 | 0.10986019 | 11327.827556028002 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.080593 | 0.08494955 | 0.09058905999999999 | 12310.619583635149 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.07597899999999999 | 0.08000955 | 0.08434594 | 13039.761361935269 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.090626 | 0.10521589999999999 | 0.1108236 | 10747.672591500312 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0885465 | 0.09809885 | 0.10352259999999999 | 22287.181928483107 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0893535 | 0.0934117 | 0.09877801 | 22265.818695227474 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.08834800000000001 | 0.0932564 | 0.09937733999999998 | 22454.340221875827 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.092664 | 0.10075814999999999 | 0.11017173 | 21303.374539740595 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.091052 | 0.1129568 | 0.11556247 | 20275.86531310397 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.092026 | 0.09504765 | 0.10637911999999997 | 43129.34447278042 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0950515 | 0.09914864999999999 | 0.10476050999999999 | 41804.28987261815 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0966545 | 0.10135154999999998 | 0.11087310999999998 | 41079.13237586092 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0828845 | 0.11973805 | 0.12370232999999999 | 41490.42743230888 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.09178549999999999 | 0.1144669 | 0.12233821999999998 | 40267.92667693767 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.098303 | 0.10598415 | 0.11400578999999998 | 80643.14521169229 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1055185 | 0.10865335 | 0.11470451 | 75508.8684222101 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.108254 | 0.1132729 | 0.12800592 | 73178.50469238867 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.10341449999999999 | 0.1073316 | 0.11227637999999998 | 77074.45411536157 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.11993200000000001 | 0.12290065 | 0.12732885 | 66520.59851908518 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.140848 | 0.14663895 | 0.16165939 | 112819.32098563472 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.17393599999999998 | 0.18229369999999998 | 0.1876572 | 91426.64885675548 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.155035 | 0.16114205 | 0.17002971999999997 | 102466.35229135897 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.13378099999999998 | 0.13761495 | 0.14866234 | 119622.39397007436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.1264655 | 0.131639 | 0.13489843999999998 | 125825.7115719233 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.1483305 | 0.15820365 | 0.16534527 | 213804.64429120888 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.2002335 | 0.21742815 | 0.22429269999999998 | 155480.91607800944 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1730985 | 0.18149215 | 0.19270163999999998 | 183260.90327924766 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.147064 | 0.15365695000000001 | 0.16238218 | 216322.00937188065 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.150475 | 0.1568381 | 0.15791422 | 212243.251991107 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.1739715 | 0.1844709 | 0.18777714 | 364737.16526959033 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.226174 | 0.2375868 | 0.24633797999999996 | 282861.6512172642 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.20700000000000002 | 0.21401645 | 0.22374852 | 316570.0897060706 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.176251 | 0.18554505 | 0.19354415 | 361122.62189468404 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.2102925 | 0.2150042 | 0.22271213999999998 | 311016.4757090301 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.215088 | 0.22521835 | 0.23311407999999997 | 590264.164426199 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.3067985 | 0.3450838499999999 | 0.35945599 | 429287.58183042996 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.255722 | 0.2900583 | 0.29881294999999997 | 487738.78052196885 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.233188 | 0.24431640000000002 | 0.25097672 | 556718.2848901438 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.232867 | 0.24874599999999997 | 0.2932325599999999 | 550822.26144727 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.17357699999999998 | 0.18219125 | 0.18616639999999998 | 5717.235807962623 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1881545 | 0.1969035 | 0.20043703 | 5291.664929236682 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.20102 | 0.21133355 | 0.21554493 | 4952.009089511725 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.221546 | 0.23410794999999998 | 0.24395707 | 4515.949792754032 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.424182 | 0.59268775 | 0.59503445 | 2332.4173308497134 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1906755 | 0.197686 | 0.20335888 | 10447.756359993327 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.196986 | 0.2033891 | 0.20605134 | 10108.221653488747 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.205983 | 0.2174051 | 0.2214986 | 9659.853378881473 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.2179065 | 0.23212324999999998 | 0.24051265 | 9131.190924618915 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.35274700000000003 | 0.52211685 | 0.52866762 | 4822.753674673048 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.20446999999999999 | 0.211735 | 0.21520077 | 19500.452995523086 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.2277385 | 0.23536405 | 0.23819229 | 17485.384841080584 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.206338 | 0.2230452 | 0.22499719 | 19245.951325449423 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.21905 | 0.2309486 | 0.24091657999999996 | 18170.823181900116 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.364554 | 0.45711005 | 0.4923044299999999 | 10357.305818252795 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.2178635 | 0.22733699999999998 | 0.23097102 | 36549.026315664436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.2548285 | 0.26273040000000003 | 0.27186397 | 31552.756089859406 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.2428015 | 0.25415825 | 0.25630972 | 32819.41682029572 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.23643350000000002 | 0.25559805 | 0.26155064 | 33779.61553899472 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.381836 | 0.44186755 | 0.44840957 | 20904.840954924774 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.2344325 | 0.2401654 | 0.24596826 | 68062.90339500314 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.2637635 | 0.30028679999999996 | 0.3060645 | 58427.32768813361 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.25875349999999997 | 0.2689808 | 0.27424147 | 62843.08396150232 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.257446 | 0.2686519 | 0.27745116999999997 | 62924.10849123091 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.3794155 | 0.45205249999999997 | 0.45905381 | 41836.38511459326 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.275309 | 0.282032 | 0.28800225 | 115796.050442786 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.333874 | 0.39592285 | 0.39937672999999996 | 91929.11231238201 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.28441 | 0.3259107 | 0.33241885 | 107946.95351770449 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.28911549999999997 | 0.30169995 | 0.30432511 | 112766.03887228604 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.474516 | 0.70356915 | 0.7202968399999999 | 67035.49596546867 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.378477 | 0.3872863 | 0.39327912 | 168669.7961941769 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.3951255 | 0.49919685 | 0.50627473 | 153169.24152937 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.357527 | 0.42866075 | 0.43801210999999995 | 181694.3921899574 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.333337 | 0.37515295 | 0.38216215 | 189294.63141495845 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.4078345 | 0.59428045 | 0.60634952 | 144231.35863776368 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.5598369999999999 | 0.5684222 | 0.5943521299999999 | 228028.6145982565 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.5602865 | 0.74897435 | 0.7569411699999999 | 236115.8006332626 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.425317 | 0.5233857 | 0.53355252 | 287351.5223255071 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.408377 | 0.48731275 | 0.49280799000000003 | 318855.4682965495 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.465796 | 0.55768235 | 0.5640158200000001 | 273396.69073810615 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 1 | ok | 58.217344 | 0.0439285 | 0.04724555 | 0.05091357999999999 | 22513.496841356395 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 2 | ok | 59.097531 | 0.050157 | 0.055398249999999996 | 0.059068619999999995 | 19657.503386987835 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 4 | ok | 58.408528 | 0.050288 | 0.0536089 | 0.05678882999999999 | 19582.24798723864 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 8 | ok | 58.916891 | 0.0499155 | 0.05397365 | 0.05872325999999999 | 19730.3804056803 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 64 | ok | 58.25497 | 0.0409145 | 0.04382575 | 0.04844139999999999 | 24183.785158991457 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 1 | ok | 58.519312 | 0.04658 | 0.0494791 | 0.05074694 | 42400.855140446474 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 2 | ok | 59.007935 | 0.045532500000000004 | 0.04922084999999999 | 0.05255324999999999 | 43405.339898535676 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 4 | ok | 58.898769 | 0.04721 | 0.0513321 | 0.05438393 | 41927.06870253332 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 8 | ok | 58.523792 | 0.049274 | 0.053404099999999996 | 0.057215529999999994 | 40179.97414016864 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 64 | ok | 58.371479 | 0.0441055 | 0.0486339 | 0.05020297 | 44823.04980513179 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 1 | ok | 59.543686 | 0.048143000000000005 | 0.05211795 | 0.05441409999999999 | 82169.87635898704 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 2 | ok | 58.799254 | 0.049543500000000004 | 0.0514314 | 0.05360817 | 80866.40263786205 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 4 | ok | 58.978247 | 0.052184499999999995 | 0.057224950000000004 | 0.061526529999999996 | 75443.57047259361 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 8 | ok | 59.058413 | 0.050791 | 0.0532331 | 0.05495025 | 78820.3432153025 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 64 | ok | 59.472267 | 0.0444065 | 0.04659505 | 0.05271803999999999 | 89218.91958564949 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 1 | ok | 59.152604 | 0.052595 | 0.05430945 | 0.05774638 | 151095.1185322318 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 2 | ok | 59.686234 | 0.055016999999999996 | 0.060000399999999995 | 0.06111539 | 144744.1412094893 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 4 | ok | 59.275342 | 0.0555405 | 0.059865299999999996 | 0.06264204 | 142595.6834860652 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 8 | ok | 59.473243 | 0.056479 | 0.059161349999999994 | 0.06368504 | 141620.7933313601 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 64 | ok | 64.543183 | 0.077055 | 0.0803323 | 0.08430383 | 103174.57860922732 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 1 | ok | 58.836666 | 0.053173 | 0.057869849999999994 | 0.06047254 | 297165.00867164636 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 2 | ok | 60.146203 | 0.06738050000000001 | 0.0726164 | 0.07529794 | 235366.73520742136 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 4 | ok | 59.268408 | 0.06146 | 0.06645749999999999 | 0.07264041999999998 | 255965.2706320806 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 8 | ok | 59.954817 | 0.0646165 | 0.06952005 | 0.07574459 | 245629.4838894692 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 64 | ok | 64.371181 | 0.078072 | 0.0812851 | 0.08580278999999999 | 203940.43511711535 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 1 | ok | 59.679122 | 0.06453149999999999 | 0.06870419999999999 | 0.07269752 | 492267.553568709 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 2 | ok | 59.789066 | 0.10438900000000001 | 0.1099112 | 0.11136191 | 304598.2339775142 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 4 | ok | 59.889923 | 0.095597 | 0.10894735 | 0.11055984999999999 | 325986.1488485354 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 8 | ok | 60.08155 | 0.086411 | 0.0981499 | 0.10065761999999999 | 364655.8252970407 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 64 | ok | 65.473338 | 0.12525150000000002 | 0.13004765 | 0.1352185 | 254876.42316622785 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 1 | ok | 60.169786 | 0.08244850000000001 | 0.08997915 | 0.09241245 | 760725.6371671528 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 2 | ok | 60.105171 | 0.147167 | 0.17500480000000002 | 0.18038364999999998 | 407850.25064309704 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 4 | ok | 60.380797 | 0.13405050000000002 | 0.13805175 | 0.14447527999999998 | 490749.75059944316 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 8 | ok | 60.053798 | 0.1238595 | 0.1436086 | 0.15095626999999998 | 494428.7154309348 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 64 | ok | 65.312149 | 15.115882 | 48.37276859999999 | 124.72161642999995 | 2946.7308442666845 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 1 | ok | 60.748493 | 0.13274049999999998 | 0.13894825 | 0.14071424 | 959537.9345076378 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 2 | ok | 60.784712 | 0.202876 | 0.24884085 | 0.2516834 | 584542.5753411446 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 4 | ok | 62.051833 | 0.1922305 | 0.20436785 | 0.2072753 | 699114.9968796532 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 8 | ok | 61.298142 | 0.19746350000000001 | 0.20963465 | 0.21381626999999997 | 679227.7096038341 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 64 | ok | 62.983554 | 6.497928 | 11.8720065 | 12.30765283 | 17567.41751748872 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 1 | ok | 58.464856 | 0.106379 | 0.1134841 | 0.11818199999999998 | 9307.763046691463 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 2 | ok | 58.672955 | 0.1179155 | 0.12432504999999999 | 0.12898446 | 8422.039871958044 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 4 | ok | 58.233185 | 0.11692150000000001 | 0.1256283 | 0.13082452 | 8444.789066903166 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 8 | ok | 59.357529 | 0.15067999999999998 | 0.16235185 | 0.16691885999999997 | 6637.3314316936285 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 64 | ok | 57.739588 | 0.2934055 | 0.31740045 | 0.32989286 | 3461.7928500545886 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 1 | ok | 59.930689 | 0.11799899999999999 | 0.1262979 | 0.13058034 | 16765.288476127487 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 2 | ok | 58.98661 | 0.12468000000000001 | 0.1326812 | 0.13815095 | 15917.263338547298 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 4 | ok | 59.04138 | 0.126666 | 0.13907424999999998 | 0.14347395999999998 | 15545.272185279746 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 8 | ok | 58.929791 | 0.1436735 | 0.1539248 | 0.15792783 | 13848.056356050143 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 64 | ok | 58.197656 | 0.311097 | 0.3816602499999997 | 0.42700723 | 6610.450951081935 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 1 | ok | 59.306019 | 0.1273745 | 0.14146904999999999 | 0.14620158 | 30738.37889478315 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 2 | ok | 59.066312 | 0.149625 | 0.15590795 | 0.15909671 | 26600.520040166783 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 4 | ok | 59.444355 | 0.13536399999999998 | 0.14957884999999999 | 0.15282185999999998 | 29181.386244244703 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 8 | ok | 58.868836 | 0.1421515 | 0.15983155 | 0.1678422 | 27775.602022286035 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 64 | ok | 58.105812 | 0.301959 | 0.34133725 | 0.35475665 | 12930.42641830938 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 1 | ok | 59.244854 | 0.1345265 | 0.1483019 | 0.15640615 | 58590.520083145806 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 2 | ok | 59.483964 | 0.174571 | 0.1815532 | 0.18755838 | 45763.43302630265 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 4 | ok | 59.360864 | 0.15734199999999998 | 0.16658599999999998 | 0.18574149999999995 | 50303.892100163604 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 8 | ok | 59.525438 | 0.157482 | 0.16818135 | 0.17182174 | 50682.105111645076 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 64 | ok | 64.113991 | 0.288053 | 0.3510279 | 0.35435348 | 26123.614199412128 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 1 | ok | 59.972435 | 0.1581295 | 0.16510075 | 0.16686252 | 100670.16126353132 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 2 | ok | 59.952422 | 0.218818 | 0.2248503 | 0.22903666 | 75526.06975833263 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 4 | ok | 59.634299 | 0.1807295 | 0.1913716 | 0.19797869999999998 | 90821.29355179034 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 8 | ok | 59.272817 | 0.17942 | 0.1966751 | 0.20173954 | 88603.57553083786 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 64 | ok | 60.122219 | 0.289197 | 0.3375958 | 0.34216101 | 55551.87524382066 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 1 | ok | 59.411443 | 0.201925 | 0.20891095 | 0.21246481999999997 | 158289.68000751873 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 2 | ok | 59.833429 | 0.2582005 | 0.31260645 | 0.31953314 | 119084.83305794965 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 4 | ok | 59.916931 | 0.2445045 | 0.2562216 | 0.26026066 | 135455.5836908091 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 8 | ok | 59.237524 | 0.212009 | 0.22374785 | 0.22985638 | 153558.18794812 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 64 | ok | 65.926806 | 0.358607 | 0.4396971499999999 | 3.0710680599999907 | 68860.99989700977 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 1 | ok | 59.542019 | 0.2955105 | 0.30286335000000003 | 0.30577137 | 215888.64895264318 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 2 | ok | 60.796293 | 0.3255555 | 0.4126273 | 0.42828579999999994 | 177937.64486626541 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 4 | ok | 61.100684 | 0.30503650000000004 | 0.36991835 | 0.37226012999999997 | 201106.58900600552 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 8 | ok | 60.21077 | 0.28821549999999996 | 0.31300485 | 0.32700060999999997 | 227655.89413204126 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 64 | ok | 65.119272 | 0.325762 | 0.38600365000000003 | 0.40092245 | 199717.51206153346 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 1 | ok | 60.337301 | 0.487777 | 0.49835665 | 0.5288302799999999 | 261243.57657151276 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 2 | ok | 61.167111 | 0.4026495 | 0.6534575999999999 | 0.67809769 | 278505.10643464286 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 4 | ok | 61.236682 | 0.36994 | 0.46397565 | 0.47088349999999995 | 322620.76136180846 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 8 | ok | 61.381293 | 0.367031 | 0.47746805000000003 | 0.47984612 | 336952.4305826661 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 64 | ok | 66.979161 | 0.3990045 | 0.4762369 | 0.47890754999999996 | 318656.59160826827 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 1 | ok | 1386.887355 | 0.059470499999999996 | 0.061395849999999995 | 0.06216884 | 16755.18430535184 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 2 | ok | 1382.201125 | 0.063535 | 0.0662643 | 0.07328599999999998 | 15626.342888842011 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 4 | ok | 1381.074488 | 0.0616 | 0.06336535 | 0.07044283999999998 | 16180.810852793455 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 8 | ok | 1323.996819 | 0.061726500000000004 | 0.0631186 | 0.06916528 | 16125.235025300492 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 64 | ok | 1405.612826 | 0.0591355 | 0.06096995 | 0.06540380999999999 | 16808.208591213275 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 1 | ok | 1367.539768 | 0.0641225 | 0.0657436 | 0.06986969999999999 | 31148.291095731474 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 2 | ok | 1376.34274 | 0.062202 | 0.0636687 | 0.06538659 | 32084.374203906467 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 4 | ok | 1372.363388 | 0.06490599999999999 | 0.06755239999999998 | 0.07694405999999997 | 30597.461328633563 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 8 | ok | 1351.804938 | 0.06451950000000001 | 0.0665761 | 0.07495654999999997 | 30811.798453863954 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 64 | ok | 1406.648248 | 0.0645435 | 0.06613785 | 0.07248660999999999 | 30854.712563668698 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 1 | ok | 1366.636882 | 0.06455749999999999 | 0.06669785 | 0.07036837 | 61660.4543758883 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 2 | ok | 1361.444199 | 0.063079 | 0.06585305 | 0.07033901 | 62935.51375518589 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 4 | ok | 1376.620349 | 0.0721535 | 0.0735402 | 0.074575 | 55389.14195572414 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 8 | ok | 1435.717698 | 0.069772 | 0.07187690000000001 | 0.07507008999999999 | 57140.66947151167 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 64 | ok | 1347.196783 | 0.0672085 | 0.0687445 | 0.07766337999999998 | 59074.80583588192 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 1 | ok | 1360.881601 | 0.0679285 | 0.06973135 | 0.07820070999999996 | 117071.13769146253 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 2 | ok | 1375.798283 | 0.0802165 | 0.08342455 | 0.08891107999999999 | 99099.4829979972 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 4 | ok | 1371.684137 | 0.0749815 | 0.07813004999999999 | 0.08964311999999998 | 105694.05587199182 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 8 | ok | 1355.812021 | 0.08431150000000001 | 0.08633955 | 0.087287 | 94657.97706437216 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 64 | ok | 1384.506038 | 0.105493 | 0.10944295 | 0.11193359 | 75680.51922890633 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 1 | ok | 1303.720779 | 0.1068125 | 0.1103418 | 0.11916602999999999 | 148907.6322234627 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 2 | ok | 1326.486845 | 0.148977 | 0.15344044999999998 | 0.15890464999999998 | 107026.28926878168 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 4 | ok | 1372.586357 | 0.122226 | 0.12492299999999999 | 0.13477369 | 130239.1124985918 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 8 | ok | 1359.277501 | 0.105768 | 0.10945555 | 0.11721667999999998 | 150305.81597081962 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 64 | ok | 1382.297539 | 0.12719550000000002 | 0.1368954 | 0.14269163999999998 | 124345.66977085269 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 1 | ok | 1417.449764 | 0.1133255 | 0.1164722 | 0.11945146 | 281005.9309814308 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 2 | ok | 1319.383299 | 0.1847425 | 0.1897581 | 0.19973204 | 172441.20159184784 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 4 | ok | 1304.136336 | 0.146121 | 0.14914739999999999 | 0.16131325999999999 | 217886.55219179575 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 8 | ok | 1361.772184 | 0.11452899999999999 | 0.1168432 | 0.1217147 | 278793.437760062 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 64 | ok | 1390.256669 | 0.16641 | 0.1717804 | 0.17552346 | 191394.3364501901 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 1 | ok | 1381.974964 | 0.1293425 | 0.13224115 | 0.13782655 | 493372.00958745147 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 2 | ok | 1376.227048 | 0.199419 | 0.2095676 | 0.21594443 | 318972.43031573785 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 4 | ok | 1314.634044 | 0.17781550000000002 | 0.181353 | 0.18497784 | 359605.553163804 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 8 | ok | 1374.193836 | 0.14777849999999998 | 0.15193585 | 0.15626125999999999 | 435289.93438276293 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 64 | ok | 1380.779921 | 0.1926485 | 0.19787459999999998 | 0.20624120999999998 | 332242.5395456874 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 1 | ok | 1378.160852 | 0.1725465 | 0.17676424999999998 | 0.18015372 | 740064.4896821713 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 2 | ok | 1353.661882 | 0.2557415 | 0.32133885 | 0.32421418999999996 | 468771.64177672367 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 4 | ok | 1298.332917 | 0.2104635 | 0.25560964999999997 | 0.25689868 | 560872.4511195277 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 8 | ok | 1372.124393 | 0.1964825 | 0.2000975 | 0.20883189 | 656628.8633118266 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 64 | ok | 1402.906261 | 0.2186395 | 0.22242184999999998 | 0.23123206 | 585837.0774272488 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 1 | ok | 1378.890245 | 0.1473665 | 0.1539683 | 0.15609265 | 6757.733888548751 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 2 | ok | 1364.469142 | 0.16045700000000002 | 0.17046545 | 0.17680005 | 6188.301856948491 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 4 | ok | 1307.832681 | 0.1596305 | 0.1698745 | 0.18298964999999995 | 6217.453960064298 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 8 | ok | 1395.275123 | 0.1762825 | 0.188054 | 0.19232224999999997 | 5628.656797604984 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 64 | ok | 1402.432656 | 0.34007299999999996 | 0.3987068 | 0.40775009 | 2897.608707847327 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 1 | ok | 1378.673453 | 0.15883 | 0.1662144 | 0.16790867 | 12518.814212910227 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 2 | ok | 1373.878127 | 0.181065 | 0.1884245 | 0.19225257 | 11012.736780723748 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 4 | ok | 1365.685939 | 0.17497449999999998 | 0.18829705 | 0.19051703 | 11323.411231986009 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 8 | ok | 1368.06967 | 0.1923165 | 0.20360409999999998 | 0.21395287 | 10378.030125345847 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 64 | ok | 1352.013668 | 0.35066200000000003 | 0.3897117 | 0.4231069299999999 | 5816.6507910877735 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 1 | ok | 1379.303068 | 0.168797 | 0.1783995 | 0.18175422 | 23539.591945881537 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 2 | ok | 1337.589296 | 0.19684600000000002 | 0.20910384999999998 | 0.21069138 | 20155.10766203472 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 4 | ok | 1399.504169 | 0.1794695 | 0.19394635 | 0.2033216 | 22104.852822021563 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 8 | ok | 1386.808324 | 0.19538450000000002 | 0.2095097 | 0.22023662 | 20431.853791288817 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 64 | ok | 1416.949 | 0.348139 | 0.4067011 | 0.40969875 | 11339.41190067877 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 1 | ok | 1399.714795 | 0.1746505 | 0.1797773 | 0.18419935999999998 | 45659.67152888899 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 2 | ok | 1361.772328 | 0.2185425 | 0.22445025 | 0.22556016 | 36519.78833861074 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 4 | ok | 1391.714861 | 0.2013635 | 0.2143281 | 0.21855262 | 39328.052690151875 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 8 | ok | 1359.693184 | 0.204375 | 0.21952129999999997 | 0.22494123 | 39336.13583633099 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 64 | ok | 1362.076979 | 0.375376 | 0.4003544 | 0.40497548 | 22070.54972607965 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 1 | ok | 1373.766868 | 0.19978200000000002 | 0.2103511 | 0.2128284 | 79774.42982223466 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 2 | ok | 1299.177282 | 0.2526615 | 0.26549865 | 0.26743824 | 62626.72125629204 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 4 | ok | 1417.612036 | 0.2292085 | 0.23936949999999999 | 0.24736123999999998 | 70630.0358111939 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 8 | ok | 1364.461352 | 0.2522275 | 0.26401365 | 0.26773463 | 63248.84157770124 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 64 | ok | 1405.397165 | 0.356132 | 0.4149616 | 0.42862942 | 43162.39657480486 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 1 | ok | 1377.076509 | 0.246295 | 0.2561121 | 0.25749306 | 129328.4933222445 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 2 | ok | 1353.515 | 0.3116875 | 0.3718344 | 0.3761206 | 96143.15716101274 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 4 | ok | 1384.223418 | 0.28957900000000003 | 0.3021735 | 0.30530314 | 113324.39959316542 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 8 | ok | 1304.861022 | 0.2517985 | 0.26182954999999997 | 0.27577118 | 127019.82358693029 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 64 | ok | 1392.111938 | 0.3670145 | 0.4199804 | 0.42538001 | 84774.53576404534 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 1 | ok | 1317.470357 | 0.345032 | 0.3561825 | 0.36057065 | 184617.74204193684 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 2 | ok | 1383.845726 | 0.4337955 | 0.5768471000000001 | 0.58175013 | 149691.2641067059 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 4 | ok | 1388.779036 | 0.3289455 | 0.3967027 | 0.40125825 | 182480.80487264806 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 8 | ok | 1363.342188 | 0.312163 | 0.36076664999999997 | 0.36234613 | 204334.61402181158 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 64 | ok | 1390.365236 | 0.3882495 | 0.45628575 | 0.47725967999999996 | 166703.34140177505 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 1 | ok | 1370.322226 | 0.5409495 | 0.54997115 | 0.55385626 | 236041.7364572621 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 2 | ok | 1369.718257 | 0.534358 | 0.74225885 | 0.7470179699999999 | 245168.46693744248 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 4 | ok | 1356.986478 | 0.4515045 | 0.6011215999999999 | 0.60526025 | 287728.9923780141 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 8 | ok | 1326.602521 | 0.35554549999999996 | 0.4423069 | 0.44952285 | 339803.38445232797 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 64 | ok | 1390.189156 | 0.41764049999999997 | 0.5266678499999999 | 2.4945073999999927 | 251626.65822459228 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0999565 | 0.10657915000000001 | 0.11215593999999998 | 9905.091395268813 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.1049625 | 0.1098131 | 0.11584053 | 9464.49866550569 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.095582 | 0.09968695 | 0.10305454 | 10392.277290899587 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0999765 | 0.10606764999999999 | 0.11410493999999999 | 9928.248547745445 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0986065 | 0.11909455 | 0.12272232 | 9562.377780858988 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1136465 | 0.1261392 | 0.13586382 | 17348.51837581089 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.11355799999999999 | 0.11987010000000001 | 0.12630252 | 17509.87513182747 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1107365 | 0.11728174999999999 | 0.12251492999999998 | 17930.645696896063 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.1181895 | 0.12498725 | 0.13306165999999997 | 16777.90698630369 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.112262 | 0.13810315 | 0.1414892 | 17401.99068332223 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.12414 | 0.13362685 | 0.14296458999999997 | 31911.622675697014 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.1301835 | 0.13764535 | 0.14323487 | 30550.461266139424 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.1199955 | 0.12824335 | 0.14091843999999998 | 32933.1895909385 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1194725 | 0.126368 | 0.13614278999999996 | 33188.80163368557 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.140673 | 0.14615794999999998 | 0.14942857 | 28298.553703368052 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.13443 | 0.1445463 | 0.15414754 | 58759.53336704172 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.15398699999999999 | 0.16309674999999998 | 0.1705966 | 51520.837151778716 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.133455 | 0.1384395 | 0.14717485 | 59543.85240984391 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.1397335 | 0.1439273 | 0.15254443999999998 | 58297.27646783798 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.146488 | 0.15036595 | 0.15251463 | 54510.743590592705 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.186059 | 0.20000669999999998 | 0.20650697999999998 | 85273.8735667727 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.349603 | 0.3575023 | 0.36846441 | 49689.907029562826 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.215313 | 0.25856485 | 0.26883825 | 69621.21706590084 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1920015 | 0.1977988 | 0.20816382999999997 | 85162.80786887312 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.22737000000000002 | 0.2316457 | 0.24086447 | 72577.52549829915 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.221558 | 0.230199 | 0.24033555999999998 | 143910.37405814033 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.3300995 | 0.41730295 | 0.42453052999999996 | 93669.10428801939 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.2916415 | 0.30341155 | 0.31370486 | 117300.51343167858 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.218983 | 0.23185304999999998 | 0.23687449 | 147072.77381457505 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.220807 | 0.24508254999999998 | 0.24965396 | 142035.6442000011 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.281677 | 0.29153375000000004 | 0.29677372999999996 | 226013.34495795303 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.34603249999999997 | 0.37015665 | 0.39159963 | 196063.18601819503 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.3315785 | 0.35199835 | 0.36278263 | 204845.04624504948 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.2667205 | 0.2769378 | 0.28854357 | 252725.6460931142 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.2454015 | 0.25276495 | 0.25545593 | 268557.4228222616 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.37494099999999997 | 0.388793 | 0.39292237999999996 | 340036.0278797665 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.4050205 | 0.5695381499999997 | 0.6270135299999999 | 288830.6524057134 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.3824385 | 0.47262785 | 0.47961612000000003 | 329952.8461294289 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.34158350000000004 | 0.42079839999999996 | 0.43303843 | 356941.49990945397 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.33118749999999997 | 0.3391548 | 0.34384448 | 398080.87697217194 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.2510905 | 0.26006514999999997 | 0.26149962 | 3967.1601655226164 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.2782565 | 0.28942874999999996 | 0.29242267 | 3578.092257101386 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.279678 | 0.2914696 | 0.30020189999999997 | 3603.8319112252225 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.301336 | 0.3223437 | 0.34280176999999995 | 3306.32637779249 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.4488895 | 0.63347275 | 0.64328285 | 2045.2606361529595 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.28286100000000003 | 0.29601145 | 0.3165903299999999 | 7016.837251827623 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.28777949999999997 | 0.30121005 | 0.30823166 | 6923.972565558942 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2842125 | 0.29974714999999996 | 0.30716842 | 7094.564946024195 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.315789 | 0.33428015 | 0.34200433 | 6365.262572491588 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.470614 | 0.6659544 | 0.68681788 | 4191.288214646599 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.28853249999999997 | 0.3016252 | 0.32452490999999994 | 13717.816288260705 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.3497495 | 0.36747535 | 0.3736038 | 11662.054421326819 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.2920735 | 0.31224355 | 0.32371715 | 13543.955383502176 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.3024375 | 0.3266366 | 0.3295354 | 13131.441792914535 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.495293 | 0.72326015 | 0.73063586 | 8129.738265515462 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.31216750000000004 | 0.32847945 | 0.3564833399999999 | 25310.166600375123 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.369516 | 0.42269505 | 0.42549756 | 21673.963658073066 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.3574885 | 0.3833472 | 0.39653696 | 22542.141547404175 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.3266545 | 0.3627256 | 0.37106875 | 24246.444743807213 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.47961200000000004 | 0.59804895 | 0.6259648499999999 | 15922.8639149649 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.3775565 | 0.38485445 | 0.38679486 | 42294.034791178754 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.38746800000000003 | 0.4508145 | 0.45588287 | 39074.43411061337 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.3702585 | 0.4395538 | 0.44548747 | 41319.283399668 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.35909599999999997 | 0.41872135 | 0.42442694999999997 | 42767.778819588995 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.4686695 | 0.6976766499999998 | 0.74276324 | 31896.279677743914 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.47650000000000003 | 0.48653359999999995 | 0.49355182999999997 | 66938.69996509983 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.490865 | 0.62326685 | 0.6412397799999999 | 62731.78414055809 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.440716 | 0.53181395 | 0.5389166599999999 | 71406.06576677172 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.40483250000000004 | 0.4846541 | 0.49872614 | 76178.43636998514 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.47018550000000003 | 0.67282675 | 0.68708636 | 61955.12164597721 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.7072555 | 0.71656045 | 0.7411820399999999 | 90275.37572399793 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.5499345 | 0.83090025 | 0.8395686099999999 | 106237.85527017781 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.513898 | 0.64609745 | 0.6833551499999998 | 118651.57527393126 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.447411 | 0.53377135 | 0.54064619 | 139796.16234196318 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.4683665 | 0.7517049499999994 | 0.9050249499999999 | 119628.78887739373 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.1227255 | 1.1319956 | 1.13694049 | 113875.99455779506 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.7378125 | 0.91428355 | 0.91686285 | 165944.95735409064 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.597961 | 0.75065415 | 0.9987664699999999 | 204177.83583234245 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.561547 | 0.7191413 | 0.73556946 | 228164.9445739221 | - |
| `full_mlp_capacity_search_hd256_depth4` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.5230585000000001 | 0.7766784999999999 | 0.78593884 | 217952.53954042186 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 1 | ok | 63.565808 | 0.051138 | 0.053431 | 0.058384559999999995 | 19515.096683643504 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 2 | ok | 63.642193 | 0.0600015 | 0.0670963 | 0.06904265 | 16229.966946049319 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 4 | ok | 63.648696 | 0.0552105 | 0.060984449999999996 | 0.06409572999999999 | 17830.200151128774 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 8 | ok | 63.047685 | 0.060091 | 0.0625752 | 0.06348149 | 16674.38134710326 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 1 | 64 | ok | 62.677776 | 0.056591 | 0.0742727 | 0.07761504999999999 | 17079.95022219307 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 1 | ok | 64.075962 | 0.061498 | 0.0644578 | 0.0681943 | 32568.473587293607 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 2 | ok | 63.32702 | 0.053988999999999995 | 0.059303049999999996 | 0.061452 | 36551.8184164144 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 4 | ok | 64.274139 | 0.0558625 | 0.06051469999999999 | 0.06195229 | 35485.984100859685 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 8 | ok | 64.416011 | 0.058543 | 0.0649953 | 0.0680607 | 33830.34622652935 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 2 | 64 | ok | 63.571715 | 0.052221500000000004 | 0.05653959999999999 | 0.05941203 | 37950.0736421179 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 1 | ok | 64.63312 | 0.058526999999999996 | 0.06161849999999999 | 0.06614606 | 68329.58090051555 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 2 | ok | 64.56071 | 0.06252350000000001 | 0.06663835 | 0.06935189 | 64107.24886305794 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 4 | ok | 63.845811 | 0.0611045 | 0.06584155 | 0.06862246999999999 | 64816.35420322714 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 8 | ok | 64.119673 | 0.06440599999999999 | 0.0694675 | 0.07203733999999999 | 61595.24287620218 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 4 | 64 | ok | 63.561323 | 0.0690065 | 0.0734204 | 0.07637305 | 57555.662800940576 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 1 | ok | 63.946519 | 0.058685 | 0.0609824 | 0.06635197999999999 | 135302.89243758307 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 2 | ok | 64.832363 | 0.071993 | 0.07726005 | 0.07906448999999999 | 110888.80984980942 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 4 | ok | 64.32228 | 0.06635 | 0.06987975 | 0.07189672999999999 | 119921.9068542565 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 8 | ok | 63.839159 | 0.062254500000000004 | 0.0686984 | 0.07981016999999999 | 125734.40681178722 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 8 | 64 | ok | 64.207723 | 0.08949850000000001 | 0.0951538 | 0.0971157 | 89305.24980911003 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 1 | ok | 63.931337 | 0.072133 | 0.07820309999999998 | 0.08264239 | 219292.41464796575 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 2 | ok | 65.670328 | 0.101732 | 0.1061313 | 0.10846737 | 163896.94487899897 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 4 | ok | 64.432443 | 0.084859 | 0.0897992 | 0.09117481 | 193821.26868618737 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 8 | ok | 64.637123 | 0.073293 | 0.08243845 | 0.08854316 | 212691.40232227105 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 16 | 64 | ok | 70.329184 | 0.1024545 | 0.1071345 | 0.10838587 | 155426.3315130967 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 1 | ok | 64.410024 | 0.09692049999999999 | 0.1016925 | 0.10542454999999999 | 331195.5351529906 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 2 | ok | 65.554858 | 0.1588275 | 0.1636741 | 0.16646586 | 208416.6185680189 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 4 | ok | 65.108022 | 0.134368 | 0.14003234999999997 | 0.14134423 | 241446.020215068 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 8 | ok | 64.543322 | 0.113312 | 0.12634235 | 0.13060641 | 280056.9355750024 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 32 | 64 | ok | 69.914952 | 0.175479 | 0.27139464999999996 | 0.3206430499999998 | 173378.716657241 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 1 | ok | 64.688214 | 0.1461465 | 0.1522995 | 0.15695488000000002 | 439174.71385708754 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 2 | ok | 65.958659 | 0.2339965 | 0.2406568 | 0.24184201 | 286454.3866595327 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 4 | ok | 65.671004 | 0.172206 | 0.2092658 | 0.21089301 | 355286.87972184585 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 8 | ok | 65.132023 | 0.16722399999999998 | 0.17521675 | 0.17913065 | 393385.0823213604 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 64 | 64 | ok | 70.679628 | 8.453034 | 18.34157104999999 | 27.576690549999974 | 6375.638201882104 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 1 | ok | 66.130169 | 0.240618 | 0.24629755 | 0.25188182 | 530134.4578368741 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 2 | ok | 67.346103 | 0.3076485 | 0.40776144999999997 | 0.41097268000000003 | 390020.6156365725 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 4 | ok | 66.859087 | 0.2790845 | 0.28449915000000003 | 0.28846114 | 482977.2407540422 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 8 | ok | 65.772171 | 0.232132 | 0.274622 | 0.27968111999999995 | 520020.4205518896 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `fp32` | 128 | 64 | ok | 73.197911 | 6.0500695 | 11.91222955 | 13.224101899999999 | 19645.235109768964 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 1 | ok | 63.272557 | 0.140272 | 0.1448737 | 0.14972524 | 7103.3644659591155 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 2 | ok | 64.081798 | 0.157418 | 0.1653087 | 0.17227171 | 6328.597222125535 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 4 | ok | 63.285429 | 0.156738 | 0.1681137 | 0.17199949 | 6316.451437794891 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 8 | ok | 62.48567 | 0.18126799999999998 | 0.19803715 | 0.20586059 | 5462.99004523028 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 1 | 64 | ok | 63.361589 | 0.3913605 | 0.48829995 | 0.49560558 | 2548.061406852991 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 1 | ok | 64.301099 | 0.1631015 | 0.1741767 | 0.18054329999999996 | 12147.239606609217 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 2 | ok | 63.992521 | 0.17487550000000002 | 0.18116865 | 0.18362178999999998 | 11485.299620456788 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 4 | ok | 63.780301 | 0.171224 | 0.17917695 | 0.18584039 | 11706.34219183464 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 8 | ok | 64.188915 | 0.2039025 | 0.2272506 | 0.22952706 | 9733.400221259655 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 2 | 64 | ok | 63.044879 | 0.39330849999999995 | 0.487347 | 0.49670359999999997 | 5153.245939435446 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 1 | ok | 64.180529 | 0.178807 | 0.19123315 | 0.20017670999999998 | 22087.310475691094 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 2 | ok | 63.891385 | 0.2338905 | 0.24051215 | 0.24464797 | 17709.94789024933 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 4 | ok | 63.930105 | 0.1822715 | 0.1917142 | 0.20206511999999996 | 21900.02419952674 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 8 | ok | 63.932595 | 0.19406800000000002 | 0.2172076 | 0.22109791999999998 | 20396.387473436254 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 4 | 64 | ok | 71.462392 | 0.4003015 | 0.4354531 | 0.43890386 | 10204.313312215507 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 1 | ok | 63.517462 | 0.19913399999999998 | 0.2081413 | 0.20893251 | 39950.59709163648 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 2 | ok | 63.98251 | 0.2822605 | 0.29395305 | 0.29664092 | 30232.311886808413 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 4 | ok | 64.037326 | 0.2333615 | 0.2508045 | 0.25711455 | 34204.43770084953 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 8 | ok | 64.51559 | 0.223466 | 0.258474 | 0.27220841999999995 | 35015.05997729623 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 8 | 64 | ok | 69.073419 | 0.32952400000000004 | 0.4459324 | 1.1042214699999975 | 21623.708017233013 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 1 | ok | 64.314992 | 0.2520025 | 0.26193259999999996 | 0.26532658 | 63174.35664419712 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 2 | ok | 65.127404 | 0.3160785 | 0.39811874999999997 | 0.40419096 | 48203.54428610249 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 4 | ok | 64.434122 | 0.2998215 | 0.31252854999999996 | 0.31601879 | 56967.18661567343 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 8 | ok | 64.357597 | 0.2502715 | 0.2779036999999999 | 0.29102166 | 64171.785302383076 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 16 | 64 | ok | 69.434559 | 0.370616 | 0.5196021 | 0.5293791999999999 | 40975.869003424865 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 1 | ok | 64.691493 | 0.365588 | 0.3764589 | 0.38192424999999997 | 87172.95703092395 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 2 | ok | 65.744611 | 0.3978795 | 0.50558755 | 0.51394559 | 79238.14504492928 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 4 | ok | 64.89545 | 0.341671 | 0.43658909999999995 | 0.44281911999999995 | 90807.23717789042 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 8 | ok | 64.762188 | 0.30371349999999997 | 0.37850255 | 0.385986 | 97952.73878306456 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 32 | 64 | ok | 72.447422 | 0.413281 | 0.4872409 | 0.49698154 | 76068.15497498334 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 1 | ok | 65.329513 | 0.5982995 | 0.60765325 | 0.61104614 | 106799.57517798983 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 2 | ok | 65.8982 | 0.49283299999999997 | 0.632369 | 0.63584848 | 125631.50935619074 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 4 | ok | 66.30268 | 0.4292455 | 0.529332 | 0.53564059 | 141091.3841840791 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 8 | ok | 64.754863 | 0.4086535 | 0.49739009999999995 | 0.53545375 | 154834.82632008183 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 64 | 64 | ok | 73.235149 | 0.3738075 | 0.48436925 | 0.8293452099999987 | 155451.66310690454 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 1 | ok | 65.815958 | 1.028705 | 1.0396013499999999 | 1.07059541 | 124121.08935417101 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 2 | ok | 67.296127 | 0.655727 | 1.0400322499999999 | 1.04430789 | 178944.95619595234 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 4 | ok | 66.410472 | 0.5178395 | 0.6989719 | 0.7110183800000001 | 236284.44482643763 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 8 | ok | 66.11075 | 0.503763 | 0.6331922999999999 | 0.6496413999999999 | 255072.8103344343 | - |
| `full_mlp_capacity_search_hd256_depth4` | `jit` | `bf16` | 128 | 64 | ok | 74.00631 | 0.5278555 | 0.6365188 | 0.64151173 | 241318.6465296814 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 1 | ok | 1362.162969 | 0.07912849999999999 | 0.08549439999999998 | 0.08948920999999999 | 12501.628337090904 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 2 | ok | 1326.329188 | 0.09475449999999999 | 0.1007212 | 0.10573165999999999 | 10481.314541314618 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 4 | ok | 1385.037178 | 0.0809865 | 0.08408144999999999 | 0.08996219 | 12287.3791904878 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 8 | ok | 1384.896718 | 0.0750395 | 0.0780918 | 0.08428416999999998 | 13244.015691509792 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 1 | 64 | ok | 1392.685404 | 0.080506 | 0.0832378 | 0.08969637999999999 | 12352.651397801326 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 1 | ok | 1328.343909 | 0.0844205 | 0.08908134999999999 | 0.10006648 | 23455.84320237936 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 2 | ok | 1318.933262 | 0.082932 | 0.09152044999999999 | 0.09429449999999999 | 23855.5252139781 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 4 | ok | 1359.615634 | 0.0849065 | 0.0871073 | 0.09167735999999999 | 23434.19724282609 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 8 | ok | 1385.595933 | 0.07845250000000001 | 0.08383594999999999 | 0.08671377999999999 | 25249.764356574142 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 2 | 64 | ok | 1383.518239 | 0.085479 | 0.09121484999999999 | 0.09696632 | 23221.84525426759 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 1 | ok | 1323.467727 | 0.086227 | 0.08903935 | 0.09575686 | 46137.038075974786 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 2 | ok | 1372.618358 | 0.1008385 | 0.10346069999999999 | 0.10670276999999999 | 39536.450927455946 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 4 | ok | 1366.227717 | 0.096623 | 0.0998272 | 0.11292336999999998 | 41121.69286495451 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 8 | ok | 1374.291696 | 0.0829815 | 0.08504329999999999 | 0.08965723999999999 | 48119.374544369675 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 4 | 64 | ok | 1442.553635 | 0.121148 | 0.1276677 | 0.13511188999999998 | 32739.230880166397 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 1 | ok | 1320.799474 | 0.1032555 | 0.10677199999999999 | 0.11822759999999997 | 76863.95080707148 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 2 | ok | 1399.133156 | 0.126374 | 0.13432414999999998 | 0.14249105999999997 | 62738.35674127563 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 4 | ok | 1366.262256 | 0.104196 | 0.10894235 | 0.12543580999999998 | 76004.76397860618 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 8 | ok | 1351.863588 | 0.095201 | 0.09774875 | 0.10508436999999997 | 83769.14229612894 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 8 | 64 | ok | 1397.856669 | 0.1452435 | 0.15010545 | 0.15625195 | 55045.76298312806 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 1 | ok | 1319.361349 | 0.1481575 | 0.15194444999999998 | 0.15507424 | 107755.70327365867 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 2 | ok | 1356.444613 | 0.2529905 | 0.3143058 | 0.32076696 | 57647.16691196738 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 4 | ok | 1366.3421 | 0.22572150000000002 | 0.2341653 | 0.24023717 | 74938.01923265528 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 8 | ok | 1369.853449 | 0.16325050000000002 | 0.16770035 | 0.16909662 | 97767.9216849617 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 16 | 64 | ok | 1327.184314 | 0.2205475 | 0.22972355 | 0.23871539999999997 | 74813.1028416448 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 1 | ok | 1358.040392 | 0.16467199999999999 | 0.1694636 | 0.17879467999999998 | 193227.54366851912 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 2 | ok | 1387.241512 | 0.34174400000000005 | 0.3753478 | 0.37665853 | 99144.58670246779 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 4 | ok | 1353.255906 | 0.23906349999999998 | 0.26112285 | 0.26461645 | 130712.96323755097 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 8 | ok | 1372.701851 | 0.2023325 | 0.20643589999999998 | 0.20902842 | 163502.43470453628 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 32 | 64 | ok | 1346.825968 | 0.238404 | 0.2683177 | 0.27344103999999997 | 129613.97394175861 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 1 | ok | 1308.658697 | 0.21507150000000003 | 0.2305946 | 0.23517663 | 294956.5653414053 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 2 | ok | 1366.107831 | 0.27053 | 0.33604775 | 0.33915727 | 214733.2797931689 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 4 | ok | 1409.369067 | 0.28006600000000004 | 0.32769550000000003 | 0.32928298 | 225154.19896398106 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 8 | ok | 1380.719565 | 0.23235699999999998 | 0.23971925 | 0.24348117 | 280482.24012750725 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 64 | 64 | ok | 1370.137874 | 0.29247 | 0.3072555 | 0.31224583 | 222318.32858301591 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 1 | ok | 1342.517167 | 0.3048805 | 0.313801 | 0.31708337999999997 | 418171.0479013629 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 2 | ok | 1374.524809 | 0.43469800000000003 | 0.56440535 | 0.56589561 | 311516.00187123765 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 4 | ok | 1374.988932 | 0.342029 | 0.43451660000000003 | 0.43839813 | 357268.27984087495 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 8 | ok | 1392.108496 | 0.2845835 | 0.35398789999999997 | 0.35788523 | 430587.4524797969 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `fp32` | 128 | 64 | ok | 1396.02576 | 0.288281 | 0.3359197 | 0.34857217999999995 | 435164.50102066476 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 1 | ok | 1373.057819 | 0.217858 | 0.22581855 | 0.23203252 | 4574.114700683199 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 2 | ok | 1351.915634 | 0.2460575 | 0.2535968 | 0.25919787 | 4052.1534845521373 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 4 | ok | 1310.940376 | 0.2301505 | 0.24052975000000001 | 0.24761154 | 4330.506321066857 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 8 | ok | 1374.072227 | 0.246207 | 0.264432 | 0.26758113 | 4054.6488820035547 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 1 | 64 | ok | 1394.73394 | 0.4888055 | 0.5735688 | 0.5897497399999999 | 2037.6116208739943 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 1 | ok | 1368.576616 | 0.23573850000000002 | 0.2466576 | 0.24818705 | 8441.46569506522 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 2 | ok | 1363.198982 | 0.2547865 | 0.26186645000000003 | 0.26617483000000003 | 8005.102772711436 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 4 | ok | 1361.850306 | 0.24488100000000002 | 0.26028249999999997 | 0.26910744 | 8231.047478163442 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 8 | ok | 1383.12481 | 0.270347 | 0.28706299999999996 | 0.29047689 | 7438.679616515237 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 2 | 64 | ok | 1347.756177 | 0.4547875 | 0.54252225 | 2.137625109999994 | 3781.279309641249 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 1 | ok | 1370.425154 | 0.260801 | 0.26982015 | 0.2738368 | 15263.669589176045 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 2 | ok | 1317.343386 | 0.3267315 | 0.33738735 | 0.33964203 | 12675.936452234822 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 4 | ok | 1365.754925 | 0.27031249999999996 | 0.28656054999999997 | 0.29322003999999996 | 14884.910615367507 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 8 | ok | 1393.446085 | 0.2888425 | 0.31104755 | 0.31800996 | 13886.873363518767 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 4 | 64 | ok | 1386.406788 | 0.442947 | 0.52881485 | 0.5374719299999999 | 8686.43058367124 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 1 | ok | 1380.576554 | 0.282697 | 0.29280905 | 0.29469744999999997 | 28210.49145356109 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 2 | ok | 1389.689173 | 0.33391950000000004 | 0.38483605000000004 | 0.38782524 | 22856.990041838006 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 4 | ok | 1437.357389 | 0.317824 | 0.32556535 | 0.33386231 | 25973.801654829866 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 8 | ok | 1326.094989 | 0.2863095 | 0.31292200000000003 | 0.31635714 | 27583.88224109665 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 8 | 64 | ok | 1337.649049 | 0.47262899999999997 | 0.55206415 | 0.5551744200000001 | 16859.403365946164 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 1 | ok | 1344.434211 | 0.3434355 | 0.35293630000000004 | 0.35510958 | 46477.51827249107 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 2 | ok | 1321.846408 | 0.413103 | 0.49500565 | 0.49883899 | 38664.59823954285 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 4 | ok | 1372.776465 | 0.3773915 | 0.39161155 | 0.39496142 | 44848.574953349074 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 8 | ok | 1408.075674 | 0.3421415 | 0.3624865 | 0.37911806 | 47406.165717022406 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 16 | 64 | ok | 1406.232977 | 0.484839 | 0.5818924 | 0.60113156 | 33008.484459584884 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 1 | ok | 1383.826744 | 0.465349 | 0.47691655 | 0.4991754299999999 | 68407.53291781172 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 2 | ok | 1352.624327 | 0.47163849999999996 | 0.5993010999999999 | 0.6035817099999999 | 67009.80433011493 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 4 | ok | 1381.998688 | 0.4085505 | 0.4453702 | 0.5250314700000001 | 78596.58333775029 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 8 | ok | 1355.483158 | 0.36378200000000005 | 0.43348355 | 0.44372561 | 83331.15457085444 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 32 | 64 | ok | 1393.70797 | 0.485357 | 0.5603829 | 0.7973637599999992 | 64202.396771775086 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 1 | ok | 1381.527064 | 0.682923 | 0.69738815 | 0.70861571 | 93400.0271794079 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 2 | ok | 1384.625857 | 0.5196565 | 0.8716233 | 0.87417128 | 107547.856443508 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 4 | ok | 1376.152629 | 0.4622335 | 0.5990836500000001 | 0.60555845 | 131685.0534666008 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 8 | ok | 1381.012896 | 0.481449 | 0.55923425 | 0.5607523600000001 | 131154.82437651764 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 64 | 64 | ok | 1417.638856 | 0.5384705000000001 | 0.6351591 | 0.64152794 | 120053.12200582355 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 1 | ok | 1390.053645 | 1.148639 | 1.1580577 | 1.17017022 | 111361.04124800787 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 2 | ok | 1391.139407 | 0.7051635 | 0.9491053499999993 | 1.1320526 | 168961.39116051025 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 4 | ok | 1365.746074 | 0.5422454999999999 | 0.73560015 | 0.7907363399999998 | 221694.78108662242 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 8 | ok | 1374.342255 | 0.5015769999999999 | 0.6505692 | 0.66828274 | 246297.35225343413 | - |
| `full_mlp_capacity_search_hd256_depth4` | `compile` | `bf16` | 128 | 64 | ok | 1397.362431 | 0.6195195 | 0.7699315 | 0.78271115 | 202738.85622910227 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.121454 | 0.13098155 | 0.13853630999999997 | 8127.027388569922 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.134858 | 0.14240909999999998 | 0.15151247999999998 | 7352.255686749207 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.1164325 | 0.12394029999999999 | 0.1301754 | 8528.844295542194 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.11125850000000001 | 0.11622865 | 0.11996934 | 8925.307316181512 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.122877 | 0.1496968 | 0.15360200999999998 | 7854.289730296257 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1465685 | 0.15491024999999997 | 0.16402411 | 13538.266587422679 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.144762 | 0.157972 | 0.16656209 | 13645.96419244412 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1370905 | 0.14666964999999998 | 0.15342166999999998 | 14468.400579083263 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.14450200000000002 | 0.1565473 | 0.16327761999999998 | 13683.649352653916 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.135927 | 0.1772317 | 0.18138528999999998 | 14028.324590180033 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.16531 | 0.17450635 | 0.18141422999999998 | 24002.10642485985 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.168546 | 0.1774811 | 0.1852865 | 23554.64571810695 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.154857 | 0.1605353 | 0.17332355999999996 | 25663.030044222534 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1416545 | 0.1494907 | 0.1610506 | 27954.98129811751 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.1732525 | 0.18149235 | 0.18303262 | 22965.421538924384 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.1779925 | 0.1871138 | 0.19058427 | 44575.9087049729 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.2050145 | 0.2137834 | 0.22209259 | 38806.50591071593 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.1714595 | 0.1793326 | 0.18706654 | 47441.00829221384 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.1624675 | 0.17012665 | 0.1784487 | 48901.244059874196 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.177558 | 0.18296735 | 0.18375677 | 44937.00898594134 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.2381585 | 0.24936334999999998 | 0.25520857 | 66855.29330545357 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.382412 | 0.42245835 | 0.42717601 | 42125.77619375498 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.3326675 | 0.3607084 | 0.37111953 | 49953.41219894795 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.2504075 | 0.25606819999999997 | 0.26500144 | 67595.99899281962 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.220051 | 0.2376857 | 0.24025617 | 71178.55041501099 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.280287 | 0.2929669 | 0.29610473 | 113614.15723290447 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.452434 | 0.6152384000000001 | 0.6263918199999999 | 72319.25733908253 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.331926 | 0.41250404999999996 | 0.42575987 | 92164.47049293994 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.2768595 | 0.29097095 | 0.30092285999999996 | 116687.81633337711 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.245803 | 0.25139195 | 0.25649241 | 129789.4223405397 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.3649905 | 0.37559205 | 0.37760702 | 174833.49431673676 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.412292 | 0.5133845 | 0.52033111 | 163547.5257354809 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.409285 | 0.42418904999999996 | 0.44422238 | 169568.33413101104 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.3097245 | 0.36891874999999996 | 0.38148426999999996 | 193453.63771731718 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.3233745 | 0.3673325 | 0.37325443 | 191163.71989783732 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.528655 | 0.5429396000000001 | 0.54874772 | 241157.83493921845 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.5197780000000001 | 0.6904985499999999 | 0.6930710999999999 | 248973.87750516346 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.441297 | 0.6233791499999998 | 0.65059406 | 269370.291386262 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.453094 | 0.5353921499999997 | 0.57906125 | 292503.5009012764 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.3907775 | 0.40286259999999996 | 0.41790755999999996 | 334041.9371906165 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.3208255 | 0.3313957 | 0.3549314099999999 | 3095.703998689898 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.3505865 | 0.3765903 | 0.38007431999999997 | 2808.854159966379 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.346797 | 0.3801299 | 0.3855569 | 2815.96476304445 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.374154 | 0.40986739999999994 | 0.41929363999999997 | 2702.787292640132 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.48688 | 0.8495318999999999 | 0.88386755 | 1757.0783638325238 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.3633805 | 0.37283355 | 0.3943902799999999 | 5472.335320192359 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.35419999999999996 | 0.38797755 | 0.39064692 | 5535.637715982592 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.35282650000000004 | 0.3893683 | 0.39456025 | 5535.426591801867 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.38464149999999997 | 0.42604 | 0.43422315 | 5192.718666032116 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.5330379999999999 | 0.6702084 | 0.6951103999999999 | 3568.3932706240553 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.375395 | 0.38314695 | 0.38658984 | 10620.074840729409 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.42640199999999995 | 0.43584 | 0.45332653999999994 | 9730.782337384519 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.382255 | 0.4033586 | 0.41923027999999996 | 10631.03879547779 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.3885075 | 0.42405 | 0.42991187999999997 | 10242.329941602844 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.5209315 | 0.9059562 | 0.9453950899999999 | 6779.746051052166 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.4282395 | 0.4401862 | 0.44804506 | 18604.005647059876 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.439381 | 0.56880065 | 0.57690035 | 17126.407228713964 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.4199545 | 0.4881988 | 0.5138323499999999 | 18361.46592986388 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.40580700000000003 | 0.46372204999999994 | 0.48603314 | 19657.7439347874 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.549417 | 0.804661 | 0.8883975799999998 | 13876.214797560768 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.501044 | 0.50985 | 0.52864128 | 31862.26072139185 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.530472 | 0.656895 | 0.6618415999999999 | 30874.344851226375 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.5032725 | 0.59063305 | 0.59588789 | 32971.7500106643 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.4729035 | 0.56462565 | 0.57041253 | 34228.441301817154 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.552828 | 0.6026374999999999 | 0.61760503 | 28490.603781066377 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.6792535 | 0.6862442 | 0.7050167199999999 | 47088.39365567477 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.574925 | 0.7249801 | 0.87141993 | 52408.39675540926 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.5143425 | 0.7090107999999996 | 0.7996226099999999 | 58304.229501772505 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.482491 | 0.57594885 | 0.6016782099999999 | 64557.068105245606 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.5460320000000001 | 0.7967465499999999 | 0.8249940299999999 | 54701.47184566526 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.011349 | 1.02490055 | 1.02929924 | 63172.0631500319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.715782 | 1.0662386 | 1.07097845 | 83518.18038105221 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.5735680000000001 | 0.86137265 | 0.8684057199999999 | 101857.14541721308 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.51686 | 0.6866356499999997 | 0.7356858900000001 | 115715.21458717568 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.6467125 | 0.77732265 | 0.9534212099999999 | 94136.41066044253 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.6879995 | 1.6966148 | 1.69745339 | 75812.66467858221 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.069571 | 1.2372896 | 1.23839842 | 118228.80103101427 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.7360635 | 1.02210005 | 1.02832651 | 163340.68034609134 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.6496999999999999 | 0.8066105 | 0.8341572899999999 | 193229.9822645068 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.6666814999999999 | 0.9266649499999999 | 0.9604151799999999 | 181727.06531319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 1 | ok | 67.554385 | 0.06515599999999999 | 0.0677858 | 0.07153347 | 15240.26893588175 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 2 | ok | 67.810452 | 0.06965450000000001 | 0.07210074999999999 | 0.07579568999999999 | 14308.644911230596 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 4 | ok | 67.493991 | 0.064436 | 0.06649205 | 0.07002768 | 15496.46169290166 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 8 | ok | 67.934362 | 0.0716985 | 0.08017575 | 0.08312417 | 13686.117650794327 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 64 | ok | 71.263557 | 0.089012 | 0.09202489999999999 | 0.10082881999999997 | 11452.694074490613 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 1 | ok | 68.474667 | 0.0701635 | 0.07561579999999998 | 0.08208272999999998 | 28358.81845818776 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 2 | ok | 68.767642 | 0.068185 | 0.0711627 | 0.07710534999999999 | 29162.009250772575 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 4 | ok | 68.307999 | 0.06911800000000001 | 0.0721362 | 0.07748105 | 28930.07889232514 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 8 | ok | 67.93033 | 0.0656365 | 0.06774175 | 0.07198906 | 30276.643749265797 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 64 | ok | 69.574385 | 0.0950415 | 0.0989032 | 0.10340408999999999 | 21881.732735641097 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 1 | ok | 68.100189 | 0.07179050000000001 | 0.07731729999999999 | 0.08020816 | 55229.395293350935 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 2 | ok | 68.809195 | 0.0772495 | 0.08723199999999999 | 0.09293973999999999 | 49976.47357506454 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 4 | ok | 68.323423 | 0.073604 | 0.0761751 | 0.08331295999999998 | 54244.79046593563 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 8 | ok | 69.115503 | 0.0753245 | 0.08000589999999999 | 0.08323478 | 52441.12107579813 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 64 | ok | 75.258766 | 0.1025475 | 0.10742094999999999 | 0.11034211 | 38822.761862925756 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 1 | ok | 68.715548 | 0.0831775 | 0.08891099999999999 | 0.09590573999999999 | 95772.82685470066 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 2 | ok | 69.227777 | 0.08499799999999999 | 0.1008761 | 0.10573397999999999 | 89195.92107052944 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 4 | ok | 68.884318 | 0.0828135 | 0.08640774999999999 | 0.08827635 | 96244.0981314449 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 8 | ok | 69.208166 | 0.082872 | 0.0865478 | 0.09050272999999999 | 98231.1275288069 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 64 | ok | 69.163822 | 0.1133355 | 0.11990685000000001 | 0.12410902999999998 | 70324.1556793964 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 1 | ok | 68.883861 | 0.096253 | 0.0999027 | 0.10403075 | 165270.20231964995 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 2 | ok | 69.990426 | 0.115261 | 0.12181455 | 0.12821919999999998 | 138856.9469696641 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 4 | ok | 69.736522 | 0.10096050000000001 | 0.10366070000000001 | 0.10867711 | 161619.52460828982 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 8 | ok | 70.212577 | 0.09662950000000001 | 0.10053664999999999 | 0.10409025 | 169502.44253019683 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 64 | ok | 75.35563 | 0.11800250000000001 | 0.1231877 | 0.12520963 | 135251.78980384095 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 1 | ok | 69.715224 | 0.13163550000000002 | 0.13649995 | 0.14212644 | 241841.91640371399 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 2 | ok | 71.176739 | 0.219691 | 0.2265053 | 0.22856224 | 159687.36407860692 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 4 | ok | 69.870172 | 0.148556 | 0.17231849999999999 | 0.17434606 | 203757.77733884958 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 8 | ok | 70.128051 | 0.145652 | 0.15134894999999998 | 0.15431300999999997 | 227305.72534819687 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 64 | ok | 76.73745 | 0.2226575 | 0.2558143 | 0.30813633999999995 | 137685.96533394395 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 1 | ok | 70.325465 | 0.2049225 | 0.21492409999999998 | 0.22463793999999998 | 308445.3889727689 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 2 | ok | 70.83706 | 0.25728399999999996 | 0.30266479999999985 | 0.32955711 | 254906.28888067702 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 4 | ok | 70.414601 | 0.2224645 | 0.2676558 | 0.27011492 | 280127.9344276531 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 8 | ok | 70.52726 | 0.1929595 | 0.19827565 | 0.20174785 | 330904.6312171293 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 64 | ok | 70.633132 | 15.784759000000001 | 43.7124738 | 65.29018288999997 | 3556.3146153470548 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 1 | ok | 71.490105 | 0.34284499999999996 | 0.3485734 | 0.35301467000000003 | 372355.46891567815 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 2 | ok | 73.184087 | 0.3437565 | 0.4473242 | 0.44993514 | 355544.33615650353 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 4 | ok | 72.164134 | 0.33454700000000004 | 0.42418290000000003 | 0.43306539 | 374508.7352406985 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 8 | ok | 72.329674 | 0.27551 | 0.33429495 | 0.3399356 | 436732.68540854944 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 64 | ok | 72.331916 | 5.3732215 | 10.1205824 | 15.023606789999983 | 24241.062610918725 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 1 | ok | 68.007326 | 0.1752385 | 0.1813293 | 0.18394759 | 5687.342012748063 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 2 | ok | 68.43879 | 0.1995165 | 0.20423144999999998 | 0.20968875 | 5039.005429326789 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 4 | ok | 67.675645 | 0.1950945 | 0.2042909 | 0.21643758999999996 | 5148.761094807845 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 8 | ok | 67.850118 | 0.2225915 | 0.24056804999999998 | 0.24832495999999998 | 4471.473608110752 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 64 | ok | 69.904034 | 0.429851 | 0.5299277 | 0.53802357 | 2292.6260160230713 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 1 | ok | 68.433246 | 0.206975 | 0.23057445 | 0.23231216 | 9542.226930569039 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 2 | ok | 69.238995 | 0.228369 | 0.23500465 | 0.24402913999999998 | 8899.087638838004 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 4 | ok | 68.846646 | 0.21637 | 0.2229223 | 0.23102193999999998 | 9313.5911279476 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 8 | ok | 68.610962 | 0.2295035 | 0.25176845 | 0.25836986 | 8622.530119359959 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 64 | ok | 69.074005 | 0.44211849999999997 | 0.59549045 | 0.59889265 | 4430.683136461097 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 1 | ok | 68.883554 | 0.23611700000000002 | 0.24478994999999998 | 0.25176527 | 16875.01829884797 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 2 | ok | 68.495399 | 0.26814150000000003 | 0.31481155 | 0.31944552 | 14037.183938766435 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 4 | ok | 69.058501 | 0.233244 | 0.25405695 | 0.26082621 | 17026.858507146342 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 8 | ok | 68.880588 | 0.2551115 | 0.2689448 | 0.27122805 | 15778.947050350698 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 64 | ok | 75.322511 | 0.39258550000000003 | 0.4928743 | 0.5434181599999998 | 9710.099634846847 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 1 | ok | 68.435825 | 0.265245 | 0.2735661 | 0.27672364 | 30043.170533898687 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 2 | ok | 69.186006 | 0.34452950000000004 | 0.40897279999999997 | 0.41562268 | 23263.160784266285 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 4 | ok | 69.121448 | 0.293366 | 0.33650985 | 0.34453077 | 27523.258357454604 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 8 | ok | 68.947272 | 0.2643645 | 0.30843875 | 0.32665934999999996 | 29487.43249313072 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 64 | ok | 75.379271 | 0.40185950000000004 | 0.54531345 | 0.55366749 | 19098.853085223953 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 1 | ok | 68.865078 | 0.35708300000000004 | 0.36893295000000004 | 0.37303708 | 44600.126140306755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 2 | ok | 70.625792 | 0.376767 | 0.4731321 | 0.47998563 | 40512.4911151043 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 4 | ok | 69.44219 | 0.317275 | 0.3672906 | 0.37207542 | 47953.58668252172 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 8 | ok | 69.012283 | 0.3227765 | 0.33976535 | 0.35575315999999996 | 50895.53557721995 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 64 | ok | 75.740889 | 0.475273 | 0.5740326499999999 | 0.5794699000000001 | 35219.15316162558 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 1 | ok | 68.88882 | 0.5245735 | 0.53279085 | 0.5374288 | 60905.10466999025 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 2 | ok | 71.127752 | 0.4885615 | 0.6902051499999999 | 0.69347775 | 63126.839654197494 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 4 | ok | 69.629807 | 0.394395 | 0.51633435 | 0.52150202 | 75687.06351033952 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 8 | ok | 69.536229 | 0.39213299999999995 | 0.5082691500000001 | 0.5170645700000001 | 82959.62614244479 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 64 | ok | 77.198856 | 0.4313825 | 0.6013447 | 3.6188098499999883 | 55631.007040486395 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 1 | ok | 69.851857 | 0.8787785 | 0.892582 | 0.898641 | 72697.66164790787 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 2 | ok | 71.846616 | 0.5811085 | 0.8896683 | 0.89585686 | 100304.2478598208 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 4 | ok | 70.480209 | 0.469426 | 0.7421686000000001 | 0.74538872 | 119781.41389232324 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 8 | ok | 70.060207 | 0.4624835 | 0.57768585 | 0.5828076600000001 | 133848.82419454693 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 64 | ok | 70.759499 | 0.5210695 | 0.6600286 | 0.66298141 | 120119.11461852252 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 1 | ok | 71.018522 | 1.563884 | 1.57496775 | 1.58180502 | 81742.8547325377 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 2 | ok | 72.604741 | 0.9464014999999999 | 0.96186685 | 1.0929303199999996 | 134266.8994770787 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 4 | ok | 72.645391 | 0.6494345 | 0.9332558999999999 | 0.9420869900000001 | 181832.9970748753 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 8 | ok | 71.838844 | 0.5622705 | 0.72685165 | 0.8417984399999996 | 218224.73902026092 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 64 | ok | 72.427943 | 0.6082989999999999 | 0.7962993999999999 | 0.8200748099999999 | 202768.84019550082 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 1 | ok | 1375.461248 | 0.1024645 | 0.1046668 | 0.10945028 | 9730.962248148102 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 2 | ok | 1364.642704 | 0.111322 | 0.11429625 | 0.12463345999999997 | 8930.532956345769 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 4 | ok | 1383.473461 | 0.102269 | 0.10749714999999999 | 0.11209614 | 9720.434472315523 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 8 | ok | 1350.188582 | 0.0951675 | 0.097664 | 0.10373156999999998 | 10477.567527922718 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 64 | ok | 1345.638422 | 0.095532 | 0.0985894 | 0.10421544999999999 | 10418.963179800881 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 1 | ok | 1353.132633 | 0.09922500000000001 | 0.1023857 | 0.10778495999999999 | 20072.796001980783 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 2 | ok | 1375.453258 | 0.09936500000000001 | 0.10160605 | 0.10871178999999997 | 20021.3146916207 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 4 | ok | 1360.103176 | 0.09518299999999999 | 0.10004489999999999 | 0.10728481 | 20832.99045703206 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 8 | ok | 1401.473581 | 0.10260749999999999 | 0.10966055 | 0.11856202999999997 | 19323.80965815601 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 64 | ok | 1361.996533 | 0.0998575 | 0.10491985 | 0.11132861999999999 | 19909.340825616473 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 1 | ok | 1335.668964 | 0.1134965 | 0.1202567 | 0.12486517 | 35085.56492145219 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 2 | ok | 1380.20325 | 0.13507950000000002 | 0.13771535 | 0.14340669 | 29526.139417410883 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 4 | ok | 1392.402594 | 0.114679 | 0.11714509999999999 | 0.12321112999999999 | 34758.86989876653 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 8 | ok | 1371.863951 | 0.1058515 | 0.109162 | 0.11623271999999998 | 37609.60143036836 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 64 | ok | 1373.330025 | 0.15007500000000001 | 0.15469345 | 0.15530343 | 26630.513082106398 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 1 | ok | 1417.67505 | 0.135943 | 0.14121135 | 0.14784174 | 58557.38609197667 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 2 | ok | 1382.494802 | 0.16635850000000002 | 0.16979914999999998 | 0.17460857999999999 | 48608.66801920285 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 4 | ok | 1364.588726 | 0.13822099999999998 | 0.14028549999999998 | 0.14517639 | 57761.716169901476 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 8 | ok | 1346.997397 | 0.115201 | 0.1182967 | 0.12436872999999998 | 69268.02743706567 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 64 | ok | 1437.546119 | 0.1789115 | 0.18803705 | 0.19150631999999998 | 44637.979845728914 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 1 | ok | 1428.986983 | 0.184714 | 0.1892476 | 0.19416151 | 86400.54363222054 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 2 | ok | 1397.521325 | 0.3496595 | 0.4847433 | 0.48912603 | 41095.42738371201 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 4 | ok | 1330.57062 | 0.25535149999999995 | 0.3188622 | 0.32212297 | 58681.93766291114 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 8 | ok | 1369.826239 | 0.2162115 | 0.21920945 | 0.22354738999999998 | 78564.79405515916 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 64 | ok | 1446.424207 | 0.290406 | 0.32362035 | 0.33572918999999996 | 53255.20794328054 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 1 | ok | 1297.695496 | 0.219919 | 0.22536774999999998 | 0.2289655 | 145098.4080074733 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 2 | ok | 1389.372477 | 0.40799399999999997 | 0.5683082 | 0.57157566 | 76978.72365812132 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 4 | ok | 1365.012357 | 0.3230535 | 0.36809325 | 0.374101 | 99727.6065087721 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 8 | ok | 1431.519925 | 0.2476495 | 0.2518759 | 0.26019816 | 135314.6098507683 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 64 | ok | 1344.155891 | 0.299327 | 0.33926915 | 0.34587615 | 103929.36804253778 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 1 | ok | 1375.276147 | 0.294672 | 0.3021028 | 0.30378618999999996 | 216797.95250593702 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 2 | ok | 1362.204496 | 0.3529515 | 0.47819904999999996 | 0.482352 | 161801.90693659944 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 4 | ok | 1362.455154 | 0.36658999999999997 | 0.46840994999999996 | 0.4745889 | 161481.5039842281 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 8 | ok | 1325.608825 | 0.311713 | 0.3182395 | 0.32401532 | 218573.96471246288 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 64 | ok | 1447.105067 | 0.3604235 | 0.4026068 | 0.41236565999999997 | 186002.32526156868 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 1 | ok | 1367.061445 | 0.42539400000000005 | 0.43149805 | 0.43580038 | 300302.3011883807 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 2 | ok | 1302.570085 | 0.469472 | 0.6270308 | 0.63767164 | 265098.1032810613 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 4 | ok | 1372.834881 | 0.43835599999999997 | 0.59614975 | 0.60504307 | 294718.17597447295 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 8 | ok | 1364.381258 | 0.3458785 | 0.44043495 | 0.4439638 | 333227.27334024047 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 64 | ok | 1384.799813 | 0.360178 | 0.42547865 | 0.43139079999999996 | 356939.708254214 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 1 | ok | 1352.609062 | 0.27100349999999995 | 0.27924794999999997 | 0.28338742 | 3678.9268129015545 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 2 | ok | 1312.492177 | 0.303371 | 0.31503945 | 0.31815439999999995 | 3369.7490884828712 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 4 | ok | 1370.361536 | 0.29309799999999997 | 0.31168534999999997 | 0.3166413 | 3402.0761373752903 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 8 | ok | 1369.342504 | 0.3025095 | 0.3263643 | 0.34120188 | 3283.497718363106 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 64 | ok | 1390.53249 | 0.5468755 | 0.6537725999999999 | 0.66360425 | 1787.5201279235205 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 1 | ok | 1384.859954 | 0.3087855 | 0.31851175 | 0.32151549 | 6448.152533369512 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 2 | ok | 1331.407621 | 0.320917 | 0.34468624999999997 | 0.34899579999999997 | 6155.9512997767915 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 4 | ok | 1363.712101 | 0.309497 | 0.32588405 | 0.33169244000000003 | 6530.067237490318 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 8 | ok | 1364.362294 | 0.32987299999999997 | 0.36118265 | 0.36790652 | 6015.897369753416 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 64 | ok | 1344.726004 | 0.54305 | 0.6636479999999999 | 0.70306836 | 3554.7207491360696 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 1 | ok | 1362.200899 | 0.319902 | 0.3298795 | 0.33205887 | 12439.673028170322 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 2 | ok | 1364.312488 | 0.3912 | 0.42560595 | 0.43232143 | 10120.118213100846 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 4 | ok | 1341.25783 | 0.3242025 | 0.34938505 | 0.35673815 | 12235.75435475086 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 8 | ok | 1359.362487 | 0.3364505 | 0.38145305 | 0.3893354 | 11703.194580063338 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 64 | ok | 1394.628208 | 0.570071 | 0.6713226 | 0.6847836599999999 | 7193.107679311407 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 1 | ok | 1329.297938 | 0.3644585 | 0.3768735 | 0.38159535 | 21881.519667164583 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 2 | ok | 1365.41627 | 0.444681 | 0.5391232499999999 | 0.54355383 | 17802.208737840312 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 4 | ok | 1369.037081 | 0.378222 | 0.44719525 | 0.45740643 | 20258.443035334672 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 8 | ok | 1359.852375 | 0.363389 | 0.40837285 | 0.41496295 | 21981.70814138723 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 64 | ok | 1379.444993 | 0.575577 | 0.6729117499999999 | 1.684457289999996 | 12799.116963322467 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 1 | ok | 1303.390372 | 0.45844 | 0.46758755 | 0.47164881000000003 | 34822.33643948574 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 2 | ok | 1368.87957 | 0.47882749999999996 | 0.58471175 | 0.5865735400000001 | 33011.77389674858 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 4 | ok | 1376.402513 | 0.442716 | 0.5245970499999999 | 0.54812472 | 35712.48095855156 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 8 | ok | 1321.864968 | 0.424784 | 0.4873967 | 0.49904498999999997 | 38577.04704278002 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 64 | ok | 1448.344316 | 0.568048 | 0.65974625 | 0.7625842299999999 | 27495.346154894414 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 1 | ok | 1368.875628 | 0.628893 | 0.6399922 | 0.64429832 | 50737.28891385164 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 2 | ok | 1314.69564 | 0.540177 | 0.7913554999999995 | 0.86554701 | 54640.60619381324 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 4 | ok | 1375.525724 | 0.49825600000000003 | 0.61094495 | 0.61345604 | 63367.10456768308 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 8 | ok | 1393.293466 | 0.478789 | 0.59465415 | 0.6166625 | 66820.0576774033 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 64 | ok | 1405.378325 | 0.5510865 | 0.64188595 | 0.704991 | 55840.325705451774 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 1 | ok | 1370.027936 | 0.990094 | 1.00343095 | 1.00978105 | 64523.355387886695 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 2 | ok | 1306.537883 | 0.6561185 | 0.9782964999999997 | 1.0438039799999999 | 89807.65361592134 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 4 | ok | 1411.202192 | 0.5412254999999999 | 0.87031265 | 0.88217846 | 107090.36971444185 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 8 | ok | 1379.236476 | 0.5495335 | 0.66827255 | 0.68683684 | 117438.49755058157 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 64 | ok | 1452.966894 | 0.6407134999999999 | 0.7540727500000001 | 1.306181639999998 | 95937.41330967916 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 1 | ok | 1400.686231 | 1.717608 | 1.7251796 | 1.72968086 | 74514.74687008664 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 2 | ok | 1388.163133 | 1.038504 | 1.1812558999999998 | 1.2279734199999999 | 121708.64870215316 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 4 | ok | 1364.72746 | 0.662192 | 0.8224635 | 1.0738341799999997 | 178573.70558547578 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 8 | ok | 1316.98101 | 0.7228215 | 0.78635335 | 0.8819594299999997 | 173001.99522660463 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 64 | ok | 1405.31949 | 0.703945 | 0.90000545 | 0.9206096100000001 | 170920.49956109485 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.056061 | 0.06010615 | 0.06142974 | 17651.65863810393 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.05742 | 0.06180214999999999 | 0.07198954999999997 | 17152.888151848343 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.058189 | 0.06271145 | 0.06786426999999999 | 16935.54096013001 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0562815 | 0.05982605 | 0.06626920999999998 | 17588.299418988117 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.057111499999999996 | 0.06040225 | 0.06510642 | 17357.018952823277 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.067559 | 0.07051195 | 0.08164708999999998 | 29280.879175965787 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0716415 | 0.07712754999999999 | 0.08181514999999999 | 27586.808098714428 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.07053999999999999 | 0.0734548 | 0.07997046999999997 | 28126.752648274396 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.06761600000000001 | 0.0707009 | 0.07796500999999997 | 29319.090510084738 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.062228000000000006 | 0.0654263 | 0.06807179 | 31938.177907787456 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.072799 | 0.07540915 | 0.08238242999999998 | 54500.828276337736 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0703355 | 0.07327204999999999 | 0.08105090999999998 | 56451.69484923447 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0667645 | 0.07108575 | 0.07742999999999998 | 59363.52799821435 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.06770899999999999 | 0.07149905 | 0.07451372 | 58573.47236723225 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.062148499999999995 | 0.06458515 | 0.06921330999999999 | 63941.25588938929 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.070931 | 0.07531924999999999 | 0.08147046 | 111795.45456451058 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.068663 | 0.07259009999999999 | 0.07809812 | 115711.24374833846 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.07176650000000001 | 0.0748269 | 0.08078349999999998 | 110610.78167533301 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.071857 | 0.07461105 | 0.08117846999999999 | 110425.2476286178 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.0769175 | 0.0810516 | 0.0823305 | 103300.95623112659 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0730575 | 0.0770223 | 0.08226353999999998 | 217434.73220874276 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0693675 | 0.07266265 | 0.08315621999999998 | 228543.02393845335 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.07243050000000001 | 0.07683845 | 0.07894800999999999 | 219503.18746066088 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.072183 | 0.0758929 | 0.07835782 | 220335.5379740037 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.0625645 | 0.0653482 | 0.06596666 | 254525.30076936638 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.071524 | 0.0763195 | 0.08201439999999999 | 443888.5972787409 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.071105 | 0.0746728 | 0.07631858 | 447641.0992722476 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.070716 | 0.0740919 | 0.08295619999999998 | 447871.78326588665 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.071209 | 0.07451735 | 0.07944839 | 445977.9893138099 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.07057250000000001 | 0.074636 | 0.07910516999999999 | 458095.4338765009 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.078178 | 0.0810491 | 0.08877865999999998 | 811970.0626637896 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.097402 | 0.10311935 | 0.11014001999999999 | 650766.0126061512 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.0893075 | 0.0926258 | 0.10023337999999997 | 711973.548402743 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.080915 | 0.0846173 | 0.09524608999999998 | 799280.447776889 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.0663475 | 0.0699795 | 0.07455662999999998 | 956423.5497779604 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.083756 | 0.08977379999999999 | 0.09627905999999999 | 1512361.6661688474 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.12344050000000001 | 0.1271512 | 0.13316050999999998 | 1033530.8076962518 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1192915 | 0.12372195 | 0.12598908 | 1071515.8114714809 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.104326 | 0.11344705 | 0.12445995 | 1211078.7210630244 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.12286 | 0.1277909 | 0.13138172 | 1038153.7732834371 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.069821 | 0.07406164999999999 | 0.07832081999999999 | 14186.19609351881 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0748095 | 0.07785755 | 0.08019045 | 13309.075325373618 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.08000099999999999 | 0.1009399 | 0.10601772999999999 | 11683.34581114666 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0700685 | 0.07404079999999999 | 0.07594479 | 14189.485250881451 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.07078699999999999 | 0.07508709999999999 | 0.08003770999999998 | 14010.323366669498 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.11593200000000001 | 0.11889305 | 0.12484473 | 17177.32860590347 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.12105250000000001 | 0.12815585 | 0.13759753 | 16355.620919885918 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.12444949999999999 | 0.13549075 | 0.13778899 | 16007.055910245237 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1307585 | 0.1400672 | 0.14297077 | 15275.937663786692 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.22278599999999998 | 0.2400019 | 0.24353239 | 9077.780969122108 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.11343800000000001 | 0.1194369 | 0.12249593 | 34988.511522241664 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.12353149999999999 | 0.13884565 | 0.14066831 | 31670.480056386124 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.12138499999999999 | 0.13315644999999998 | 0.13553865 | 32448.52163724208 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.134863 | 0.14507405 | 0.15006537 | 29621.502367054258 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.232886 | 0.24327655 | 0.24863455 | 17552.83777167076 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.114856 | 0.11942599999999999 | 0.12342169 | 69284.06866536193 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1232555 | 0.1401335 | 0.14398906 | 63360.61535829637 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.135296 | 0.1593758 | 0.16884038999999998 | 57658.13138933177 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1352655 | 0.1454019 | 0.14920704999999998 | 58947.22322473437 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.2427785 | 0.25870515 | 0.26530298999999996 | 33036.57404780541 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1246425 | 0.1374307 | 0.15495987 | 125605.7533715332 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.136505 | 0.15462135 | 0.16143730999999997 | 114557.03443913808 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.13833299999999998 | 0.16762975 | 0.16934181 | 113024.15481596629 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.14425549999999998 | 0.21618154999999997 | 0.22690534999999995 | 104046.55250852337 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.24390699999999998 | 0.2625955 | 1.7410817899999942 | 53527.6995474166 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1255735 | 0.13091855 | 0.13543838 | 253723.79310727605 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.14033600000000002 | 0.1477425 | 0.15558055 | 225976.98678613693 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.141367 | 0.17410765 | 0.1791933 | 218047.96646422276 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1546235 | 0.22045055 | 0.24107120999999992 | 194683.88872793666 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.2207425 | 0.23307695 | 0.5257919999999989 | 137579.11659073445 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1366675 | 0.14820999999999998 | 0.15782137 | 463482.77176602 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.17094399999999998 | 0.20143409999999998 | 0.21199021999999998 | 363647.6863029599 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.159446 | 0.16685065000000002 | 0.17088981 | 399352.54967883317 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.172481 | 0.24584329999999996 | 0.25893404999999997 | 350775.42195542826 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.24445650000000002 | 0.2619596 | 0.2699647 | 259825.94746952667 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1590635 | 0.17328005 | 0.18679358000000001 | 793989.2039799206 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.21486899999999998 | 0.2338253 | 0.24230850999999998 | 599802.2152195313 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.208441 | 0.2511189 | 0.27156493 | 597287.159054024 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2018665 | 0.21345945 | 0.21554227 | 633076.8613671474 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.28107550000000003 | 0.9509042999999998 | 0.99394821 | 338884.23952655756 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 1 | ok | 53.074647 | 0.0382925 | 0.04542465 | 0.050937439999999994 | 25491.930529390916 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 2 | ok | 53.44557 | 0.0383575 | 0.04189835 | 0.04917004 | 25500.017085011445 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 4 | ok | 53.301496 | 0.0389235 | 0.04505449999999998 | 0.05111041999999999 | 25106.56481435453 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 8 | ok | 53.122194 | 0.038476 | 0.04540939999999999 | 0.05121605 | 25340.68010330888 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 64 | ok | 54.542385 | 0.03412 | 0.040425949999999995 | 0.04370328999999999 | 28751.412413134796 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 1 | ok | 53.720308 | 0.0404075 | 0.0424791 | 0.043710309999999995 | 50690.50607373644 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 2 | ok | 53.583751 | 0.037599999999999995 | 0.039868850000000004 | 0.04501747999999999 | 53132.19609180818 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 4 | ok | 54.161256 | 0.036985000000000004 | 0.039151849999999995 | 0.040997599999999995 | 53579.68558368906 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 8 | ok | 53.833374 | 0.041693 | 0.0439178 | 0.0454871 | 49912.15460789011 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 64 | ok | 52.61342 | 0.0369125 | 0.0383065 | 0.04031710999999999 | 53766.90969309848 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 1 | ok | 53.701475 | 0.041278999999999996 | 0.043894499999999996 | 0.048367819999999985 | 95747.93016911957 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 2 | ok | 54.110457 | 0.041688 | 0.0458224 | 0.04801058999999999 | 94471.30316077362 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 4 | ok | 53.588152 | 0.041762 | 0.044042399999999995 | 0.04525643 | 95135.39431480397 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 8 | ok | 53.543564 | 0.041354 | 0.0431539 | 0.047415849999999996 | 95917.64894332322 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 64 | ok | 51.298582 | 0.038037 | 0.0401428 | 0.04272520999999999 | 104266.37138430291 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 1 | ok | 53.852904 | 0.042401 | 0.044784449999999996 | 0.04666071 | 187063.79062324046 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 2 | ok | 54.819463 | 0.036691 | 0.03915215 | 0.039886830000000005 | 216575.73193122205 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 4 | ok | 53.237692 | 0.041315 | 0.043430149999999994 | 0.049130699999999985 | 191880.38944043842 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 8 | ok | 53.594331 | 0.0414125 | 0.043435499999999995 | 0.046958679999999996 | 191506.31204804513 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 64 | ok | 53.226161 | 0.0367755 | 0.03838845 | 0.039697939999999994 | 216267.5359179827 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 1 | ok | 54.016025 | 0.043504 | 0.04560635 | 0.05342430999999998 | 364153.2957921632 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 2 | ok | 53.891583 | 0.042855000000000004 | 0.04426855 | 0.046266579999999995 | 371108.63602268026 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 4 | ok | 53.713654 | 0.0425495 | 0.0465632 | 0.04976388999999999 | 371285.69273788395 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 8 | ok | 54.524842 | 0.0446775 | 0.04738465 | 0.056415089999999966 | 352030.8660663367 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 64 | ok | 52.176814 | 0.037895 | 0.03890785 | 0.040530939999999994 | 420600.5966219463 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 1 | ok | 53.920313 | 0.0448775 | 0.04686125 | 0.051144199999999994 | 707037.9440750662 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 2 | ok | 54.365594 | 0.0458195 | 0.047982 | 0.05301248999999999 | 693886.3839108563 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 4 | ok | 54.107229 | 0.046143500000000004 | 0.047761599999999994 | 0.049210609999999995 | 690534.4132140665 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 8 | ok | 53.843823 | 0.045733499999999996 | 0.04715705 | 0.05271344999999998 | 694168.1631052224 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 64 | ok | 53.362699 | 0.039825 | 0.041930049999999996 | 0.047542689999999985 | 793233.3231369801 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 1 | ok | 54.236535 | 0.048718 | 0.051395249999999996 | 0.05539137999999999 | 1300963.2006296662 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 2 | ok | 54.472484 | 0.063472 | 0.06554064999999999 | 0.07425565999999997 | 999341.683665885 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 4 | ok | 54.194333 | 0.059654 | 0.061713199999999996 | 0.06484664 | 1071172.3747322487 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 8 | ok | 54.395105 | 0.0488475 | 0.0519454 | 0.05490705 | 1298489.4509905446 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 64 | ok | 55.422153 | 0.043513 | 0.04549845 | 0.05042252999999999 | 1455057.3701838737 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 1 | ok | 55.095094 | 0.0582655 | 0.060879499999999996 | 0.062425949999999994 | 2180938.1237184857 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 2 | ok | 55.853829 | 0.08896699999999999 | 0.09403365 | 0.10258996 | 1424283.6354038634 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 4 | ok | 55.471261 | 0.0837125 | 0.08901975 | 0.09421977 | 1519295.0470981465 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 8 | ok | 55.258477 | 0.07544200000000001 | 0.07961645 | 0.08362838999999998 | 1687151.053006332 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 64 | ok | 61.551731 | 0.0937895 | 0.0978437 | 0.10268501 | 1360438.367252888 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 1 | ok | 53.442452 | 0.0446515 | 0.0507207 | 0.05222768 | 22003.46862679433 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 2 | ok | 52.968102 | 0.0430225 | 0.04909375 | 0.0505227 | 22815.058851444308 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 4 | ok | 53.426346 | 0.0453095 | 0.04953035 | 0.05557888999999998 | 21704.525871143705 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 8 | ok | 53.293171 | 0.043242 | 0.04878565 | 0.05000102 | 22762.492511139964 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 64 | ok | 51.767221 | 0.043225 | 0.0481406 | 0.04921562 | 22843.98472194302 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 1 | ok | 53.901057 | 0.0763455 | 0.08587424999999999 | 0.08795027999999999 | 25856.7304422923 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 2 | ok | 53.518497 | 0.074918 | 0.07915885 | 0.08201864 | 26499.915067772206 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 4 | ok | 53.546207 | 0.0809825 | 0.08586465 | 0.09536640999999998 | 24484.719086817917 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 8 | ok | 53.616713 | 0.089448 | 0.10136975 | 0.10670681 | 22201.594474111942 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 64 | ok | 51.921362 | 0.1897345 | 0.20535584999999998 | 0.25269036999999983 | 10609.952368740831 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 1 | ok | 53.488084 | 0.077955 | 0.08243239999999999 | 0.08680940999999999 | 50894.522120795096 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 2 | ok | 53.659987 | 0.078449 | 0.09599225 | 0.09830536999999999 | 49145.60368002281 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 4 | ok | 53.403577 | 0.0792245 | 0.08710065 | 0.09047891 | 49821.66335601714 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 8 | ok | 53.271726 | 0.0881725 | 0.09657665 | 0.10445927999999997 | 44796.83626823539 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 64 | ok | 51.721651 | 0.18893 | 0.2007515 | 0.20612316 | 21075.586220065157 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 1 | ok | 53.824868 | 0.07953350000000001 | 0.08417495 | 0.08574867 | 100008.42570986606 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 2 | ok | 53.725444 | 0.0855885 | 0.1008519 | 0.10514219999999999 | 90681.50095113559 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 4 | ok | 53.525988 | 0.0835005 | 0.11299129999999999 | 0.12246996999999997 | 89116.05561643915 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 8 | ok | 53.663547 | 0.092154 | 0.10548149999999999 | 0.10617046 | 86508.29592930888 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 64 | ok | 55.222514 | 0.1827445 | 0.1924689 | 0.19991536999999998 | 43583.34410205127 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 1 | ok | 53.659908 | 0.0783225 | 0.08498834999999999 | 0.08866255999999999 | 202098.28544867082 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 2 | ok | 54.481922 | 0.094834 | 0.11576659999999998 | 0.11764754 | 162811.99102173274 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 4 | ok | 54.462617 | 0.088591 | 0.11323465 | 0.11645305999999998 | 172562.51670351237 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 8 | ok | 53.699005 | 0.100994 | 0.15135759999999998 | 0.16813299999999998 | 147258.13633415964 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 64 | ok | 55.005271 | 0.1989735 | 0.2088313 | 0.21334368999999997 | 82474.7794701717 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 1 | ok | 53.588466 | 0.084696 | 0.0907923 | 0.09520676 | 373494.9321406395 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 2 | ok | 54.121381 | 0.0998295 | 0.10400975 | 0.10588968 | 319635.99852487986 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 4 | ok | 53.959921 | 0.09829650000000001 | 0.12415145 | 0.12686445999999998 | 311849.01518081006 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 8 | ok | 53.580924 | 0.1039355 | 0.16673525 | 0.17749723999999997 | 283195.3640918898 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 64 | ok | 55.271672 | 0.1998695 | 0.2113239 | 0.21464765 | 159465.01083514915 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 1 | ok | 54.229641 | 0.09583249999999999 | 0.10484705 | 0.11295568 | 657485.2954440995 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 2 | ok | 54.292197 | 0.12394050000000001 | 0.1438883 | 0.14628138 | 499242.08810499555 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 4 | ok | 54.681359 | 0.12384700000000001 | 0.1322425 | 0.1376555 | 513354.4343395615 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 8 | ok | 54.289987 | 0.123496 | 0.20603944999999996 | 0.21974031 | 473843.8912415732 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 64 | ok | 53.700534 | 0.197597 | 0.21247144999999998 | 0.21938892999999998 | 321573.2327264677 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 1 | ok | 54.558441 | 0.117215 | 0.12998774999999999 | 0.14660754999999998 | 1072627.7956409412 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 2 | ok | 55.121019 | 0.1653155 | 0.1883832 | 0.19511385999999997 | 780346.105444999 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 4 | ok | 55.333162 | 0.16120600000000002 | 0.2027451 | 0.20669481999999997 | 759040.4684797772 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 8 | ok | 55.163198 | 0.165495 | 0.17440795 | 0.17621398 | 773278.1180537121 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 64 | ok | 60.433297 | 0.235545 | 0.7250042999999998 | 0.9476994299999995 | 399161.8349893664 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 1 | ok | 1376.389106 | 0.043704999999999994 | 0.04549875 | 0.049960109999999995 | 22725.051869930896 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 2 | ok | 1359.247893 | 0.0476915 | 0.0535217 | 0.05462769 | 20686.339651493097 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 4 | ok | 1411.864856 | 0.0439345 | 0.0461124 | 0.04940895 | 22565.16254814842 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 8 | ok | 1420.270801 | 0.044215500000000005 | 0.04718675 | 0.048300869999999996 | 22504.792395540633 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 64 | ok | 1569.863265 | 0.0490025 | 0.054091799999999995 | 0.06993734999999995 | 19924.017765848163 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 1 | ok | 1304.448309 | 0.0474745 | 0.04918375 | 0.0494821 | 42005.56405701499 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 2 | ok | 1334.910633 | 0.049429 | 0.0511559 | 0.05349404 | 40257.09793022157 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 4 | ok | 1395.155421 | 0.0490175 | 0.0499726 | 0.051487109999999996 | 40802.20397184975 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 8 | ok | 1440.897371 | 0.0465835 | 0.0481271 | 0.04996849 | 42780.766965033996 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 64 | ok | 1581.601928 | 0.046620999999999996 | 0.04881705 | 0.054038019999999985 | 42609.472256120425 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 1 | ok | 1396.43806 | 0.049598 | 0.05173255 | 0.057794739999999976 | 79939.40593030483 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 2 | ok | 1319.554613 | 0.049897 | 0.05146655 | 0.05175233 | 79905.96665843636 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 4 | ok | 1336.785469 | 0.0466505 | 0.0481373 | 0.04873329 | 85745.20298461903 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 8 | ok | 1429.682461 | 0.045917 | 0.04885615 | 0.05936038 | 86047.48616571541 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 64 | ok | 1576.376851 | 0.0456645 | 0.04775895 | 0.05362127999999998 | 86842.38064491758 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 1 | ok | 1406.863616 | 0.046391 | 0.0480239 | 0.05137101999999999 | 171284.2766600765 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 2 | ok | 1322.270654 | 0.046721 | 0.04789215 | 0.05572368999999997 | 169992.13786362382 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 4 | ok | 1360.602491 | 0.051178 | 0.0531498 | 0.057887279999999985 | 155424.8518898302 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 8 | ok | 1342.31212 | 0.050308 | 0.05212595 | 0.059129849999999984 | 157848.15086810564 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 64 | ok | 1606.542215 | 0.0469435 | 0.0494558 | 0.05868303999999997 | 168651.69714202834 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 1 | ok | 1378.033521 | 0.050287 | 0.05192665 | 0.058150879999999974 | 316316.93216860166 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 2 | ok | 1370.255644 | 0.047661499999999996 | 0.049431249999999996 | 0.05309440999999999 | 334198.4896734755 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 4 | ok | 1374.003065 | 0.046481999999999996 | 0.048959199999999994 | 0.05138971999999999 | 342900.5442688889 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 8 | ok | 1333.299835 | 0.047144000000000005 | 0.0487972 | 0.049324810000000004 | 338991.38199159136 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 64 | ok | 1564.940479 | 0.0520705 | 0.05384515 | 0.05589552999999999 | 306291.37805085356 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 1 | ok | 1348.809885 | 0.0498475 | 0.051616550000000004 | 0.05846256999999999 | 637901.050383817 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 2 | ok | 1370.857874 | 0.0521155 | 0.0541042 | 0.058067189999999984 | 610354.1236478272 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 4 | ok | 1384.417426 | 0.0517045 | 0.0543364 | 0.06174613999999998 | 613618.5739274235 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 8 | ok | 1339.486592 | 0.049919000000000005 | 0.05193585 | 0.058732799999999974 | 635382.8002670195 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 64 | ok | 1552.586414 | 0.048654 | 0.05179225 | 0.05809049999999998 | 650022.8320519758 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 1 | ok | 1363.579923 | 0.056305 | 0.05740355 | 0.05868465 | 1136494.80142417 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 2 | ok | 1366.160492 | 0.07089000000000001 | 0.0751133 | 0.07882969999999999 | 897183.348949931 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 4 | ok | 1414.11617 | 0.06516250000000001 | 0.06735375 | 0.07140430999999998 | 977057.1709737077 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 8 | ok | 1402.833721 | 0.0531635 | 0.0544066 | 0.055559159999999996 | 1200938.8339334275 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 64 | ok | 1486.299206 | 0.0517075 | 0.05425455 | 0.05759977999999999 | 1228417.7557037931 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 1 | ok | 1376.605447 | 0.060328 | 0.06307195 | 0.06957360999999998 | 2100533.502688847 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 2 | ok | 1406.691687 | 0.1015685 | 0.10473874999999999 | 0.10723519 | 1255786.6748083502 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 4 | ok | 1382.060002 | 0.0984005 | 0.10307604999999999 | 0.10758084999999998 | 1295834.3178637037 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 8 | ok | 1455.332064 | 0.07887 | 0.0803624 | 0.08642890999999998 | 1619161.153085615 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 64 | ok | 1507.720441 | 0.0978445 | 0.1022108 | 0.10421318 | 1299835.4895708512 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 1 | ok | 1392.758758 | 0.0567425 | 0.058523900000000004 | 0.0592188 | 17547.707830664618 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 2 | ok | 1364.026454 | 0.056426000000000004 | 0.05913145 | 0.06546114999999998 | 17564.752459943582 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 4 | ok | 1400.020914 | 0.055807499999999996 | 0.0575963 | 0.06397339999999997 | 17793.632299570745 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 8 | ok | 1422.336658 | 0.057832999999999996 | 0.059733 | 0.06232568999999999 | 17275.805821324633 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 64 | ok | 1558.435976 | 0.053198 | 0.05486165 | 0.05534269 | 18806.155179365585 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 1 | ok | 1396.209246 | 0.08830750000000001 | 0.0927643 | 0.09998924 | 22408.386114419456 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 2 | ok | 1381.054135 | 0.0958815 | 0.0985351 | 0.10461721999999998 | 20775.529750039215 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 4 | ok | 1409.444412 | 0.095344 | 0.10432045 | 0.10863908 | 20813.569114994552 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 8 | ok | 1357.851079 | 0.108834 | 0.1209649 | 0.1253625 | 18183.200105026164 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 64 | ok | 1534.883341 | 0.21498899999999999 | 0.22691224999999998 | 0.23422516 | 9477.145862751973 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 1 | ok | 1384.328783 | 0.096765 | 0.10111459999999999 | 0.10765857999999999 | 41152.669820607276 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 2 | ok | 1369.119251 | 0.09829199999999999 | 0.1129196 | 0.11544834999999999 | 39532.30125036716 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 4 | ok | 1326.809067 | 0.0984845 | 0.11016645 | 0.12098552999999995 | 39879.88179603036 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 8 | ok | 1341.489099 | 0.1123965 | 0.12371905 | 0.12969210999999997 | 35378.47449786889 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 64 | ok | 1532.160368 | 0.21518949999999998 | 0.22760095 | 0.28968267999999975 | 18627.444223378076 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 1 | ok | 1355.256799 | 0.097583 | 0.10494585 | 0.10831674999999999 | 81427.83712396878 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 2 | ok | 1366.12398 | 0.1059345 | 0.11919145 | 0.12113536 | 74139.33041434432 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 4 | ok | 1323.907428 | 0.103321 | 0.13054044999999997 | 0.14182656999999999 | 73235.98318208882 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 8 | ok | 1430.282185 | 0.10830100000000001 | 0.1206418 | 0.12205545 | 73174.26102685806 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 64 | ok | 1619.461984 | 0.209129 | 0.22179179999999998 | 0.26263858999999984 | 37659.24446893969 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 1 | ok | 1381.34699 | 0.09467300000000001 | 0.09915399999999999 | 0.10596788 | 167951.07249356117 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 2 | ok | 1310.132209 | 0.10762 | 0.12208429999999999 | 0.12311124999999999 | 145558.79201486302 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 4 | ok | 1327.336274 | 0.1095615 | 0.13460615 | 0.14456783999999998 | 139693.11866458965 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 8 | ok | 1390.383133 | 0.112607 | 0.17103624999999997 | 0.18930468999999997 | 131418.41372389352 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 64 | ok | 1580.178897 | 0.213308 | 0.2284176 | 0.23763907999999997 | 74448.6126361444 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 1 | ok | 1375.823278 | 0.1063545 | 0.11172075 | 0.11793808 | 298801.4141523928 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 2 | ok | 1318.417168 | 0.118977 | 0.12259355 | 0.12908292 | 268840.1044914276 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 4 | ok | 1324.382945 | 0.121914 | 0.15114315 | 0.15315635 | 252501.58050208047 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 8 | ok | 1441.122244 | 0.12917050000000002 | 0.20347744999999998 | 0.22440261999999994 | 228726.53783930335 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 64 | ok | 1585.899616 | 0.215136 | 0.2361708 | 0.23907524 | 148183.98671975112 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 1 | ok | 1317.210753 | 0.1157375 | 0.122695 | 0.12490048 | 548097.8094243363 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 2 | ok | 1371.329861 | 0.141754 | 0.16392959999999998 | 0.16822064999999997 | 440342.2890698513 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 4 | ok | 1388.425449 | 0.13725900000000002 | 0.14760589999999998 | 0.15002658 | 467944.6234332629 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 8 | ok | 1374.427704 | 0.134852 | 0.18577564999999996 | 0.20817247999999994 | 448488.4397900065 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 64 | ok | 1578.188771 | 0.22308899999999998 | 0.2375157 | 0.24758584 | 285314.87037030933 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 1 | ok | 1312.714891 | 0.14031 | 0.1478645 | 0.15314379 | 906708.8664084906 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 2 | ok | 1373.246424 | 0.1904765 | 0.212828 | 0.21439041 | 669385.9074603268 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 4 | ok | 1452.547122 | 0.1575405 | 0.19285595 | 0.21383992 | 781760.4536946793 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 8 | ok | 1393.038366 | 0.14632699999999998 | 0.1572227 | 0.16447698 | 869826.4016776776 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 64 | ok | 1517.319683 | 0.23326750000000002 | 0.84720905 | 3.1037705999999914 | 302987.6093217168 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0906585 | 0.09629735 | 0.1002294 | 10917.781463135565 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.069499 | 0.07415479999999999 | 0.07953747999999998 | 14233.919087571625 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.069479 | 0.0739981 | 0.08018866999999999 | 14233.63544699024 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0713495 | 0.07520635 | 0.07957892999999999 | 13874.05136173814 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0953245 | 0.0991348 | 0.10115292000000001 | 11171.616586543832 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0846095 | 0.0890516 | 0.09668771999999998 | 23429.618130025945 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.088507 | 0.0911139 | 0.09910003999999999 | 22471.071865633778 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0826735 | 0.08667245 | 0.09353956 | 23972.462353045812 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.08513499999999999 | 0.0880243 | 0.09571928999999998 | 23308.675862094686 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.0733695 | 0.0757645 | 0.07764091999999999 | 27157.139082300528 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.09027450000000001 | 0.09416089999999999 | 0.10191950999999998 | 43933.671578670816 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.089055 | 0.09199059999999999 | 0.09740529999999999 | 44691.01969712002 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0848175 | 0.087633 | 0.09313799999999998 | 46953.07415862438 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.092142 | 0.0953724 | 0.10126127999999998 | 44168.16764823058 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.07426250000000001 | 0.076611 | 0.08004940999999999 | 53627.152929088275 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.085698 | 0.08934874999999999 | 0.09804962999999997 | 92585.1986357571 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.091062 | 0.09375315 | 0.09970709999999999 | 87410.86823030142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.08854000000000001 | 0.09417185 | 0.09882765999999998 | 89608.48484821331 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.08670749999999999 | 0.09101184999999999 | 0.09873706999999998 | 91585.68012098469 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.079001 | 0.08183710000000001 | 0.08518614 | 100819.02858345483 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0919375 | 0.09636965 | 0.10158396 | 172842.88301064647 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0907365 | 0.09400354999999999 | 0.10100266999999997 | 175264.7538417486 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0867825 | 0.0921729 | 0.09563922 | 183057.73228996526 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.086384 | 0.09072255 | 0.09658268999999998 | 183852.1295707076 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.09764149999999999 | 0.10123525 | 0.10459672 | 163097.282636174 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.09392400000000001 | 0.09777279999999999 | 0.11060714999999996 | 337518.5635209937 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0895175 | 0.0938449 | 0.10064335999999999 | 354650.7233433988 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.09945799999999999 | 0.10496799999999999 | 0.11874841999999998 | 317688.5615637743 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0888605 | 0.09267524999999999 | 0.10086462999999998 | 357254.97778990463 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.084329 | 0.08747255 | 0.09827670999999999 | 376432.67925018375 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0987625 | 0.1076852 | 0.11082088 | 640267.7599772225 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.11836550000000001 | 0.12326514999999999 | 0.12760136 | 539030.1634541592 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.109606 | 0.11837165000000001 | 0.12426648 | 578430.158625433 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1099485 | 0.11525769999999999 | 0.12214103 | 578105.1699764694 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.1269915 | 0.13199795 | 0.13780072 | 501600.41883634974 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.111805 | 0.11835815 | 0.13096084 | 1133097.1297941515 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1441555 | 0.1509417 | 0.15748353999999998 | 880771.4787478788 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1388105 | 0.14391225 | 0.15520419 | 918781.706941367 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.129428 | 0.13312100000000002 | 0.13947088 | 984408.0531346551 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.1448505 | 0.15116495 | 0.15507922 | 878787.1090169738 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.12899650000000001 | 0.13648849999999998 | 0.14267776 | 7688.593510211989 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.131179 | 0.13909495 | 0.14211969 | 7551.636960792354 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.14455649999999998 | 0.157217 | 0.16150411 | 6853.526701477099 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.1527685 | 0.16054285000000001 | 0.16271046 | 6556.901512585382 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.2529055 | 0.2706826 | 0.271892 | 4002.494995280258 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1399285 | 0.14810405000000001 | 0.15405629 | 14179.527852917457 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.136459 | 0.14452235 | 0.14728745 | 14583.230946899394 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.149137 | 0.1582332 | 0.16298010000000002 | 13349.685808469654 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.15284 | 0.160466 | 0.16716852000000001 | 13078.349119990586 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.265236 | 0.28146135 | 0.29013066 | 7594.5004576825695 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.13416299999999998 | 0.1402024 | 0.14630743 | 29625.53472238578 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.140581 | 0.15593015 | 0.16010467 | 27928.198277919368 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1441925 | 0.15416635 | 0.15707590999999999 | 27492.119383978585 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1565055 | 0.1654867 | 0.16965284 | 25479.557114338237 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.2394545 | 0.2563958 | 0.26066914 | 16990.62550732946 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.166873 | 0.17370325 | 0.17581404 | 47699.98995915212 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1873965 | 0.19302029999999998 | 0.19792242 | 42509.83597898611 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.189096 | 0.2053559 | 0.20823887 | 41811.530553932746 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.2044875 | 0.22254905 | 0.22486556 | 38734.57250395384 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.35593949999999996 | 0.4107556 | 0.41573396 | 21680.45182061594 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.177413 | 0.18585105 | 0.19375953999999998 | 89485.75890075695 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1955275 | 0.20448945000000002 | 0.21474336999999996 | 81430.07497674153 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1885095 | 0.2021703 | 0.20357797 | 84489.47498768302 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.2074145 | 0.22053135000000001 | 0.22230076999999998 | 76968.91783433326 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.34938400000000003 | 0.3918259 | 0.40730392 | 46536.53848312874 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1834565 | 0.1914626 | 0.19652018 | 173512.21140988107 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.2091285 | 0.21496795 | 0.21770989999999998 | 153189.67716041487 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.19724150000000001 | 0.205305 | 0.20615523 | 163165.40893330614 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.222099 | 0.23730645 | 0.23969536 | 144201.9418413741 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.344611 | 0.4127582999999999 | 1.680348049999995 | 78100.92523725475 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.20364 | 0.21129825000000002 | 0.21663441 | 313406.55293926375 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.24425 | 0.2528023 | 0.25890248 | 262355.9184338729 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.225657 | 0.2399934 | 0.24405104 | 283616.41057729675 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2407785 | 0.25197235 | 0.27003244 | 266290.04378141183 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.37707650000000004 | 0.41139439999999994 | 0.43084267 | 174640.17302475142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.228586 | 0.2396401 | 0.24035767 | 556662.9378947687 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.27791849999999996 | 0.30928295 | 0.31445928 | 447498.87286221405 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.28884449999999995 | 0.30134015000000003 | 0.30780342 | 446187.6749020827 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.27835299999999996 | 0.3044244 | 0.30806466 | 456612.79491835623 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.3988435 | 0.4684780499999999 | 0.47671705 | 323864.75913697126 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 1 | ok | 57.954685 | 0.043963 | 0.04884515 | 0.06070902999999998 | 22300.203556258057 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 2 | ok | 58.126989 | 0.043296 | 0.0449473 | 0.046316569999999994 | 23436.756615376107 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 4 | ok | 58.493178 | 0.044549000000000005 | 0.04788864999999999 | 0.05916404999999997 | 22150.444935987427 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 8 | ok | 58.207351 | 0.0425055 | 0.0469425 | 0.05664650999999998 | 23217.553584952795 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 64 | ok | 57.462298 | 0.038705500000000004 | 0.04252315 | 0.05170830999999997 | 25229.461956494317 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 1 | ok | 58.547589 | 0.0450905 | 0.049217050000000005 | 0.05253727999999999 | 43747.67043654925 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 2 | ok | 59.13479 | 0.0459815 | 0.049103799999999996 | 0.05108181 | 43009.64276190722 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 4 | ok | 58.974606 | 0.046681 | 0.0499559 | 0.05090311 | 42834.67912756039 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 8 | ok | 58.275699 | 0.040999 | 0.0467803 | 0.051244979999999996 | 46871.16242357657 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 64 | ok | 56.900325 | 0.0411835 | 0.043779849999999995 | 0.04552367 | 48125.65029784966 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 1 | ok | 58.257496 | 0.0446705 | 0.047375999999999995 | 0.053875949999999985 | 88495.61437859043 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 2 | ok | 58.68625 | 0.043576000000000004 | 0.04760755 | 0.05044314 | 90371.38119100445 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 4 | ok | 58.376273 | 0.0468305 | 0.0498055 | 0.054258439999999984 | 84615.7556229285 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 8 | ok | 59.076703 | 0.0497395 | 0.052177799999999996 | 0.05320099 | 80491.15704025967 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 64 | ok | 59.227358 | 0.042981 | 0.045921449999999996 | 0.0487228 | 92260.71031520871 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 1 | ok | 58.72789 | 0.044675 | 0.04783945 | 0.05177067999999999 | 176917.09576433935 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 2 | ok | 59.046687 | 0.046148999999999996 | 0.04967175 | 0.05181586 | 171455.19597114582 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 4 | ok | 58.467401 | 0.0480725 | 0.052111149999999995 | 0.05459902999999999 | 164414.1246530348 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 8 | ok | 58.204126 | 0.047937 | 0.051757149999999995 | 0.053711699999999994 | 165122.22346981236 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 64 | ok | 56.316818 | 0.0420715 | 0.0440773 | 0.044888040000000004 | 189326.34845325106 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 1 | ok | 59.218415 | 0.048198000000000005 | 0.0521881 | 0.0535567 | 326202.5456846665 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 2 | ok | 58.780432 | 0.0492985 | 0.052962949999999995 | 0.056027999999999994 | 321649.0302482769 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 4 | ok | 58.794007 | 0.045897 | 0.04897844999999999 | 0.05316158999999999 | 345755.7615875721 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 8 | ok | 58.695893 | 0.045716 | 0.0482963 | 0.05269801999999999 | 346986.680916253 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 64 | ok | 60.112742 | 0.045046 | 0.04845245 | 0.050450919999999996 | 352671.6419818559 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 1 | ok | 59.383039 | 0.04875 | 0.05203649999999999 | 0.05709343999999999 | 649002.4426829437 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 2 | ok | 58.928744 | 0.049656 | 0.0524838 | 0.05405168 | 644594.4111247326 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 4 | ok | 59.381481 | 0.0582545 | 0.06016675 | 0.06288976 | 549806.6398773381 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 8 | ok | 59.051192 | 0.051444000000000004 | 0.05574225 | 0.06062806999999999 | 614677.2637026929 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 64 | ok | 56.548664 | 0.0448765 | 0.0468303 | 0.049334109999999994 | 707715.0226225529 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 1 | ok | 59.056005 | 0.0554915 | 0.059823549999999996 | 0.06538593 | 1148810.030260374 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 2 | ok | 59.144816 | 0.072685 | 0.07814594999999999 | 0.0812156 | 872597.255736168 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 4 | ok | 59.277948 | 0.063393 | 0.06784 | 0.07120708999999999 | 1003993.3836836015 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 8 | ok | 59.532124 | 0.0616595 | 0.0670696 | 0.07226327999999999 | 1027850.2447889597 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 64 | ok | 63.348136 | 0.0727885 | 0.0755054 | 0.07730265 | 876877.8544771875 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 1 | ok | 59.812728 | 0.063523 | 0.0671007 | 0.07102126 | 2010019.9494479983 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 2 | ok | 60.654545 | 0.10020899999999999 | 0.10771435 | 0.10977041 | 1309595.013880684 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 4 | ok | 60.903868 | 0.094356 | 0.09905104999999999 | 0.10272092999999999 | 1352490.2303713516 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 8 | ok | 59.990181 | 0.0804085 | 0.08378275 | 0.08652464 | 1587056.7586051952 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 64 | ok | 67.622966 | 0.102024 | 0.1115737 | 0.11613544999999999 | 1234900.2094506528 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 1 | ok | 58.501987 | 0.0824525 | 0.08944579999999999 | 0.09705571 | 11993.295268213258 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 2 | ok | 58.993341 | 0.086069 | 0.0928372 | 0.09421254 | 11516.056607406575 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 4 | ok | 58.144793 | 0.09002650000000001 | 0.09730565 | 0.09881506999999999 | 10950.714775054797 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 8 | ok | 57.631298 | 0.09642400000000001 | 0.10433184999999999 | 0.10846969999999999 | 10333.298475673142 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 64 | ok | 59.191544 | 0.187319 | 0.20150605 | 0.20791707 | 5287.504906804553 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 1 | ok | 58.717405 | 0.08317350000000001 | 0.08793744999999999 | 0.09535998999999999 | 23834.35693951603 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 2 | ok | 58.50539 | 0.0899655 | 0.09480809999999999 | 0.09925553 | 22099.183788745948 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 4 | ok | 58.826411 | 0.0916215 | 0.10196715 | 0.10528923 | 21514.60217564263 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 8 | ok | 58.592966 | 0.10079550000000001 | 0.1121557 | 0.11422556 | 19949.60331211294 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 64 | ok | 56.151022 | 0.20557350000000002 | 0.2165128 | 0.22530712 | 9687.452743394586 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 1 | ok | 58.377656 | 0.0875735 | 0.0942015 | 0.09760117 | 45118.057035991806 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 2 | ok | 58.885361 | 0.0962915 | 0.11372135 | 0.11805283 | 40243.23818020882 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 4 | ok | 58.7952 | 0.093707 | 0.102989 | 0.10477908999999999 | 41937.95279883412 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 8 | ok | 59.213288 | 0.1051375 | 0.12169139999999999 | 0.1265192 | 37718.12865276502 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 64 | ok | 57.123398 | 0.193337 | 0.2175112 | 0.21961284 | 20508.323405595776 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 1 | ok | 57.951896 | 0.1073205 | 0.1141391 | 0.11591097 | 73893.26819395431 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 2 | ok | 58.975354 | 0.1241945 | 0.1289923 | 0.13409067 | 64151.42816315056 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 4 | ok | 58.528918 | 0.12176899999999999 | 0.1353724 | 0.13772028 | 64912.25080795467 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 8 | ok | 59.201286 | 0.144396 | 0.15705135 | 0.16001558 | 55267.94729869617 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 64 | ok | 57.623077 | 0.284341 | 0.3359132 | 0.33947221 | 27955.075076149624 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 1 | ok | 58.488341 | 0.11627699999999999 | 0.12381995 | 0.12642404 | 136552.30354348107 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 2 | ok | 58.431822 | 0.12795250000000002 | 0.1368516 | 0.13948101 | 124213.70782110059 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 4 | ok | 58.603246 | 0.1288435 | 0.1453566 | 0.14666928 | 122316.50956385145 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 8 | ok | 58.368741 | 0.14581650000000002 | 0.1575044 | 0.16095078999999998 | 109573.7758178073 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 64 | ok | 59.003086 | 0.3069155 | 0.32367144999999997 | 0.33038235 | 54059.00337497116 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 1 | ok | 58.845149 | 0.121876 | 0.12992575 | 0.13354868 | 259870.23704550928 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 2 | ok | 59.086973 | 0.1486635 | 0.15693825 | 0.16037853999999999 | 213767.10955203633 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 4 | ok | 59.255726 | 0.1466885 | 0.15797985 | 0.16281381 | 215075.4444332426 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 8 | ok | 58.643513 | 0.1561465 | 0.16739205 | 0.16983109 | 205461.28948789416 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 64 | ok | 56.985031 | 0.29493899999999995 | 0.3240479 | 0.33738286 | 109193.68516539635 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 1 | ok | 58.798321 | 0.1366325 | 0.1471399 | 0.15547783999999998 | 463728.26219427807 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 2 | ok | 59.338175 | 0.165798 | 0.1808691 | 0.18394968 | 376868.60582052363 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 4 | ok | 59.29793 | 0.171686 | 0.1858142 | 0.19240711 | 367011.7035444614 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 8 | ok | 59.017689 | 0.167792 | 0.1836952 | 0.1919413 | 377084.7483831313 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 64 | ok | 64.126434 | 0.2968545 | 0.42945425 | 1.0952112099999973 | 182629.09651331377 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 1 | ok | 60.165271 | 0.172128 | 0.179205 | 0.18161882 | 741848.4475488226 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 2 | ok | 60.229033 | 0.239386 | 0.2466311 | 0.25282546 | 556766.7649215015 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 4 | ok | 59.942933 | 0.210668 | 0.22475285 | 0.22625079 | 601316.9216407008 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 8 | ok | 60.206867 | 0.217107 | 0.23475919999999997 | 0.23915329999999999 | 584217.6166805814 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 64 | ok | 68.355939 | 0.3198085 | 0.3691301 | 0.38617806999999993 | 394129.1507034097 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 1 | ok | 1405.660712 | 0.059655 | 0.06385205000000001 | 0.06448379 | 16679.420863820535 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 2 | ok | 1319.911332 | 0.055929999999999994 | 0.05992569999999999 | 0.06557631 | 17688.62709720785 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 4 | ok | 1401.281206 | 0.054393 | 0.057032799999999995 | 0.06349838999999999 | 18237.545488997835 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 8 | ok | 1382.231444 | 0.0546715 | 0.05806355 | 0.06529898999999999 | 18091.93743663299 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 64 | ok | 1596.284589 | 0.05608 | 0.06136975 | 0.06832811999999998 | 17629.8002860964 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 1 | ok | 1327.627643 | 0.06268599999999999 | 0.06518275 | 0.06977385999999998 | 31754.126528127646 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 2 | ok | 1385.63099 | 0.059841000000000005 | 0.0625076 | 0.07040770999999998 | 33168.07890818644 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 4 | ok | 1336.391622 | 0.057821 | 0.0602448 | 0.06423207999999998 | 34355.14701941615 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 8 | ok | 1362.422242 | 0.0590155 | 0.06091305 | 0.06769407999999998 | 33635.14145763267 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 64 | ok | 1545.57959 | 0.057679999999999995 | 0.05980145 | 0.06597008999999998 | 34416.358783657044 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 1 | ok | 1348.478142 | 0.0578695 | 0.06031635 | 0.06506872999999999 | 68662.83253409796 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 2 | ok | 1360.232055 | 0.058954 | 0.061979099999999995 | 0.06577344999999998 | 67385.28581637442 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 4 | ok | 1348.513046 | 0.058017 | 0.06064195 | 0.06470673999999998 | 68547.77762963841 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 8 | ok | 1432.331094 | 0.058253 | 0.0603189 | 0.06502364999999999 | 68342.35281534614 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 64 | ok | 1541.646209 | 0.058138999999999996 | 0.060279799999999994 | 0.06393276999999999 | 68480.60372500244 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 1 | ok | 1369.802408 | 0.0648575 | 0.06649635 | 0.06761930999999999 | 122957.74864362234 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 2 | ok | 1319.681429 | 0.0589355 | 0.0605215 | 0.0616385 | 135375.28906854693 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 4 | ok | 1323.911444 | 0.0592505 | 0.062271499999999994 | 0.06283759 | 134674.10381959326 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 8 | ok | 1400.771883 | 0.058168 | 0.060539499999999996 | 0.06347737999999999 | 136789.92350707477 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 64 | ok | 1553.726548 | 0.062516 | 0.06418454999999999 | 0.06478057 | 127645.04465981 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 1 | ok | 1332.939326 | 0.0616345 | 0.06416949999999999 | 0.07083649999999998 | 257230.9219863629 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 2 | ok | 1316.757936 | 0.059965500000000005 | 0.0619996 | 0.06673089 | 265025.18898780586 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 4 | ok | 1335.56296 | 0.059798000000000004 | 0.06098485 | 0.0647483 | 266671.4667530682 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 8 | ok | 1416.624452 | 0.0607775 | 0.06313315 | 0.06840231 | 261332.6921971285 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 64 | ok | 1626.446035 | 0.06602150000000001 | 0.06807525 | 0.07028483999999999 | 243002.36714680897 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 1 | ok | 1335.455785 | 0.062812 | 0.06466045 | 0.06528514 | 507877.3361960686 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 2 | ok | 1317.499887 | 0.062422000000000005 | 0.0639345 | 0.06863103999999999 | 511431.94231303403 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 4 | ok | 1383.274931 | 0.0734545 | 0.07526655 | 0.07841216999999999 | 434426.6857452386 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 8 | ok | 1432.036706 | 0.062023499999999995 | 0.06449534999999999 | 0.06904649999999998 | 511358.8786922509 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 64 | ok | 1536.537135 | 0.0626515 | 0.0645847 | 0.06769170999999999 | 509097.0876782913 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 1 | ok | 1381.924027 | 0.07163649999999999 | 0.07317845 | 0.07780296999999999 | 891167.8029984455 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 2 | ok | 1363.067029 | 0.0908005 | 0.0942001 | 0.09786666999999999 | 701337.121138235 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 4 | ok | 1390.509792 | 0.080571 | 0.0827625 | 0.08665379999999999 | 792009.6130166779 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 8 | ok | 1454.115168 | 0.077491 | 0.08018415 | 0.08602841999999998 | 821016.3772241846 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 64 | ok | 1589.808601 | 0.114992 | 0.1206525 | 0.12814518999999996 | 553507.4817779283 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 1 | ok | 1317.86209 | 0.0822615 | 0.08424915 | 0.08991413999999999 | 1549461.5984100585 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 2 | ok | 1365.308195 | 0.1221975 | 0.12567214999999998 | 0.12907712 | 1044898.9827581871 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 4 | ok | 1389.16784 | 0.1249315 | 0.1300522 | 0.13456224 | 1019662.7624735427 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 8 | ok | 1421.365121 | 0.0954145 | 0.10060424999999999 | 0.10329331999999998 | 1326953.7268428435 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 64 | ok | 1531.229865 | 0.12514 | 0.12805445 | 0.13409503999999997 | 1019654.3148823167 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 1 | ok | 1386.897853 | 0.1050355 | 0.10827640000000001 | 0.11173833999999999 | 9486.825361016396 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 2 | ok | 1370.168652 | 0.1098285 | 0.11620625 | 0.12413262 | 9010.335395318554 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 4 | ok | 1386.255667 | 0.113816 | 0.12630015 | 0.13017884999999998 | 8668.570053488545 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 8 | ok | 1407.3466 | 0.12788300000000002 | 0.1400653 | 0.14282055 | 7779.930052204887 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 64 | ok | 1512.471671 | 0.235016 | 0.2533854 | 0.25840552 | 4345.93390512496 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 1 | ok | 1401.255483 | 0.1130005 | 0.11713029999999999 | 0.12195168 | 17647.443953041562 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 2 | ok | 1307.63496 | 0.111399 | 0.1140581 | 0.12588840999999998 | 17875.928542763242 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 4 | ok | 1385.461888 | 0.1184955 | 0.12843975 | 0.13094362999999998 | 16739.525451192985 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 8 | ok | 1394.446372 | 0.12674400000000002 | 0.1385392 | 0.14071556 | 15722.694435879943 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 64 | ok | 1558.17551 | 0.232796 | 0.2456021 | 0.24975728 | 8576.578684406004 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 1 | ok | 1386.370296 | 0.11018049999999999 | 0.11335200000000001 | 0.11740532 | 36160.159856834696 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 2 | ok | 1320.893211 | 0.1174285 | 0.13434764999999999 | 0.13779486999999999 | 33150.447837687454 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 4 | ok | 1404.604187 | 0.1198055 | 0.13135435 | 0.13579377999999998 | 32830.630715812804 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 8 | ok | 1381.304508 | 0.12789699999999998 | 0.13733805 | 0.14058891 | 31223.445435389873 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 64 | ok | 1546.525435 | 0.2314275 | 0.2470147 | 0.25038037 | 17546.973467309155 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 1 | ok | 1402.868216 | 0.1505645 | 0.1559237 | 0.16192847 | 52882.41555240112 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 2 | ok | 1320.112648 | 0.155077 | 0.16582125 | 0.17349341 | 51218.27795954576 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 4 | ok | 1430.847058 | 0.154471 | 0.163164 | 0.16637370999999998 | 51449.22167617448 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 8 | ok | 1413.772583 | 0.1806215 | 0.19597485 | 0.19954649 | 43982.32702135627 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 64 | ok | 1568.849713 | 0.332569 | 0.37270125 | 0.37603628 | 24027.46868274749 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 1 | ok | 1405.702358 | 0.15360849999999998 | 0.16168155 | 0.16944359 | 103458.99635462226 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 2 | ok | 1370.597507 | 0.16154849999999998 | 0.17042174999999998 | 0.17471351 | 98566.31597182137 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 4 | ok | 1415.552546 | 0.170047 | 0.18706894999999998 | 0.19615135 | 92773.63002340215 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 8 | ok | 1399.287258 | 0.179004 | 0.19105709999999998 | 0.19897361 | 89105.12723154045 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 64 | ok | 1487.971461 | 0.3403845 | 0.39015195 | 0.40320617 | 47813.514860530064 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 1 | ok | 1392.67825 | 0.15769450000000002 | 0.1648931 | 0.17516203999999996 | 201543.343344579 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 2 | ok | 1348.85227 | 0.189602 | 0.19762065 | 0.20419236999999998 | 168173.1830621856 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 4 | ok | 1414.049043 | 0.1741495 | 0.1861531 | 0.19156312 | 182546.43838778182 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 8 | ok | 1392.047615 | 0.186839 | 0.2014356 | 0.20674612 | 169738.1672075833 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 64 | ok | 1540.354941 | 0.5290435 | 0.8070586499999999 | 3.38520700999999 | 53422.219002049584 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 1 | ok | 1368.445415 | 0.1700525 | 0.1772264 | 0.18462180999999997 | 374346.9400647153 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 2 | ok | 1366.304471 | 0.2158605 | 0.22663474999999997 | 0.23037984 | 300233.46905137104 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 4 | ok | 1401.823665 | 0.2030835 | 0.21305235 | 0.21533232 | 316573.6599634832 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 8 | ok | 1424.4082 | 0.20758949999999998 | 0.22429135 | 0.22751807999999998 | 308271.97757629637 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 64 | ok | 1526.201137 | 0.341084 | 0.39856025 | 0.41139075999999997 | 181018.4300520462 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 1 | ok | 1404.474877 | 0.203057 | 0.2112193 | 0.21286935 | 628613.8543939765 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 2 | ok | 1382.475687 | 0.273973 | 0.27945634999999996 | 0.28766081 | 481839.4703620542 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 4 | ok | 1416.889963 | 0.22473349999999997 | 0.2343057 | 0.23759618 | 582471.2909911897 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 8 | ok | 1349.427921 | 0.2490865 | 0.2641975 | 0.27174568 | 514227.8820142691 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 64 | ok | 1573.134066 | 0.3786945 | 0.42306205 | 0.44031279999999995 | 342075.1346653597 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0782045 | 0.0829873 | 0.09058362999999998 | 12640.984904641466 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.08138300000000001 | 0.08443555 | 0.09301364999999999 | 12180.20565546437 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.08253250000000001 | 0.08536669999999999 | 0.09376224999999998 | 12044.91017026444 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.07875 | 0.10178519999999992 | 0.117684 | 12280.109292972706 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.07934250000000001 | 0.0822619 | 0.09211968999999998 | 12499.15943152823 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0995985 | 0.1040449 | 0.10954421999999998 | 19953.528232745935 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0970785 | 0.10400519999999999 | 0.10911760999999998 | 20419.98185072013 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.09914300000000001 | 0.1042001 | 0.11119629999999998 | 20035.113539990187 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.10130700000000001 | 0.10605735 | 0.11240719999999998 | 19585.231795135343 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.088301 | 0.09257015 | 0.09579156999999999 | 22517.263422934782 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1010505 | 0.10494415 | 0.11256060999999998 | 39283.26118635056 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.100543 | 0.10812885 | 0.11353657999999998 | 39372.589905340414 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.101272 | 0.10515635 | 0.11198668999999999 | 39274.62130919201 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0995425 | 0.10370114999999999 | 0.11022845999999999 | 39922.50243826684 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.1020615 | 0.10533094999999999 | 0.11027333 | 39011.122656236126 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.107018 | 0.1149997 | 0.12037777999999999 | 73953.83062354173 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1021415 | 0.10651039999999999 | 0.11087965 | 77891.9327325269 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.103465 | 0.109776 | 0.11566428999999999 | 76667.817322825 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.114802 | 0.11925469999999999 | 0.12361387999999998 | 76449.19482752393 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.1018295 | 0.10636754999999999 | 0.10929797 | 78027.70920009114 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.1020165 | 0.11064419999999998 | 0.11374925999999999 | 155292.04221458876 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1049165 | 0.111636 | 0.12573737999999998 | 150747.82225927268 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1038625 | 0.11328925 | 0.11726087999999998 | 152594.52658692587 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.105096 | 0.10985084999999999 | 0.11349999 | 151563.0029405117 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.09389549999999999 | 0.1003987 | 0.1041358 | 169055.27684914775 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.10328999999999999 | 0.10785975 | 0.11567287999999999 | 307619.42555148475 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.104935 | 0.11515524999999999 | 0.12296307999999999 | 301330.75193322514 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.11962 | 0.12540235 | 0.14236136 | 265106.11949426366 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1073755 | 0.11249374999999999 | 0.11598624 | 295791.9163027193 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.09587 | 0.10143364999999999 | 0.10611918 | 331481.25090250943 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.1097465 | 0.1171865 | 0.12381562 | 577524.0281579059 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.136184 | 0.1424354 | 0.14944335999999997 | 466973.9939264195 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1260885 | 0.13542475 | 0.14077188 | 503052.3487278357 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.12202650000000001 | 0.1281607 | 0.13545456999999997 | 520144.6327169348 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.118558 | 0.12473295 | 0.12688096000000001 | 540676.8057084657 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.12750450000000002 | 0.13504804999999998 | 0.13787453 | 997599.3706394971 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.160935 | 0.1686793 | 0.17576573999999998 | 790520.7666717432 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1721625 | 0.1794787 | 0.18588209 | 741756.621076064 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.146783 | 0.15433745000000001 | 0.15951412 | 869599.7178148916 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.15211999999999998 | 0.1587077 | 0.16678201999999998 | 838229.1152596467 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.143747 | 0.1547992 | 0.1583141 | 6890.093093425804 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.155459 | 0.16522155 | 0.17214168 | 6385.028996970686 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1608275 | 0.17393625 | 0.1772637 | 6159.729169027896 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.16568149999999998 | 0.1837417 | 0.19061909 | 5959.945116057414 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.2680935 | 0.29623554999999996 | 0.29833271 | 3652.939480217397 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.14944600000000002 | 0.157804 | 0.16317410999999998 | 13270.398028337342 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.15840100000000001 | 0.1670568 | 0.17715706 | 12518.082369983442 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1655395 | 0.1745766 | 0.18507940999999997 | 11981.224941268036 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.17859049999999999 | 0.19097969999999997 | 0.19834764 | 11171.798804840964 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.2751935 | 0.3000895 | 0.30715898999999997 | 7203.336354892822 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1523525 | 0.16074335 | 0.16510803 | 26068.41054774814 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.15927750000000002 | 0.1786955 | 0.18415653999999998 | 24580.03474387911 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.164463 | 0.1715545 | 0.17363868999999998 | 24304.939492853133 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.18218800000000002 | 0.1999454 | 0.20527277 | 21758.746417694387 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.27710049999999997 | 0.30500385 | 0.31087548 | 14214.545104350753 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.2099655 | 0.21776605 | 0.21959448 | 37996.5050814626 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.2351185 | 0.24309195 | 0.24860828999999998 | 34018.15855285393 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.245527 | 0.2557797 | 0.26518317999999996 | 32436.115246490554 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.26336950000000003 | 0.28531075 | 0.29516097999999996 | 30180.816288465827 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.466038 | 0.5627508499999999 | 0.5678197700000001 | 17155.680769150928 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.218973 | 0.2277906 | 0.23196691 | 72801.32491131207 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.250173 | 0.26283025 | 0.2685817 | 63711.66140819815 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.24283 | 0.25338745 | 0.25886285 | 65983.92004860376 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.271861 | 0.29659445 | 0.30974147999999996 | 58417.10102215323 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.451738 | 0.5422544999999999 | 0.58318763 | 34536.69242960019 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.235013 | 0.2451489 | 0.24802845999999998 | 135621.16059672972 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.26500049999999997 | 0.27163265 | 0.2827404 | 121101.11185958327 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.25327849999999996 | 0.26786814999999997 | 0.27329546 | 126708.68643480561 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.282204 | 0.29670185 | 0.31912893999999997 | 114099.23538536766 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.455037 | 0.5552298999999999 | 0.56924562 | 68884.34950592916 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.2599455 | 0.26943435 | 0.27255165000000003 | 245123.6855816869 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.296425 | 0.30361755 | 0.30896907 | 218477.77114986014 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.28713500000000003 | 0.3040916 | 0.3132018 | 223541.4583481488 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.3105065 | 0.33338009999999996 | 0.33446841 | 205512.19676354257 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.4718755 | 0.56117205 | 0.5724844299999999 | 133792.4476922747 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.296936 | 0.30613250000000003 | 0.31028596999999997 | 429648.8882432221 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.351448 | 0.37971259999999996 | 0.40080090999999995 | 370021.8902637643 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.32064950000000003 | 0.3694794 | 0.3760415 | 389198.66670266754 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.330527 | 0.3617177 | 0.37429109 | 386492.7545894203 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.463981 | 0.5329811499999999 | 0.7080232 | 264086.1270688363 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 1 | ok | 62.323616 | 0.0457585 | 0.05023275 | 0.051227659999999994 | 21515.12986962692 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 2 | ok | 62.271355 | 0.044775 | 0.0488735 | 0.05369306999999999 | 21889.070568174604 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 4 | ok | 62.530536 | 0.0460515 | 0.05109345 | 0.053100459999999995 | 21305.353438769263 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 8 | ok | 63.086606 | 0.046489 | 0.05121555 | 0.05170644 | 21345.655753486706 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 64 | ok | 60.337652 | 0.041096 | 0.04467445 | 0.04523135 | 24004.93541472127 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 1 | ok | 62.593992 | 0.045713000000000004 | 0.047605749999999995 | 0.04888478 | 43590.70593840546 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 2 | ok | 63.008193 | 0.044738 | 0.047619049999999996 | 0.04901009 | 44257.14819329044 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 4 | ok | 62.674639 | 0.0463405 | 0.04802195 | 0.05013778 | 43336.212302370615 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 8 | ok | 62.728729 | 0.051200499999999996 | 0.05424895 | 0.057773649999999996 | 39718.09683589754 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 64 | ok | 63.411757 | 0.044649999999999995 | 0.047143899999999996 | 0.049488569999999996 | 44457.05048809396 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 1 | ok | 62.505591 | 0.049481 | 0.052340849999999994 | 0.056901379999999994 | 80198.47518639128 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 2 | ok | 62.969319 | 0.046361 | 0.049651549999999996 | 0.053667579999999986 | 85277.7367970876 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 4 | ok | 61.660139 | 0.0460295 | 0.0491212 | 0.05166243 | 86004.60038607466 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 8 | ok | 62.428527 | 0.04836650000000001 | 0.05127629999999999 | 0.056878219999999986 | 81888.57994147424 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 64 | ok | 60.679356 | 0.045833 | 0.048898199999999996 | 0.05002832 | 86560.22644155238 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 1 | ok | 62.659985 | 0.0487335 | 0.0501486 | 0.055628719999999986 | 163192.43570422023 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 2 | ok | 62.990941 | 0.0486645 | 0.05127795 | 0.053339700000000004 | 162916.6644605035 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 4 | ok | 62.891288 | 0.0522615 | 0.0559066 | 0.06023774999999999 | 152971.27581868315 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 8 | ok | 63.309317 | 0.052106 | 0.05449145 | 0.05701946999999999 | 153650.83998993586 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 64 | ok | 61.532003 | 0.047924999999999995 | 0.0493239 | 0.05168341 | 166463.85819942702 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 1 | ok | 63.252589 | 0.048911 | 0.052264 | 0.054879979999999995 | 323373.039500421 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 2 | ok | 62.67411 | 0.053818 | 0.055900649999999996 | 0.05797862 | 303059.6906366678 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 4 | ok | 63.099938 | 0.0552165 | 0.05820085 | 0.05926816 | 287902.0326963141 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 8 | ok | 63.761105 | 0.053365499999999996 | 0.0552265 | 0.0576184 | 300826.4077453775 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 64 | ok | 64.230959 | 0.0483565 | 0.051122049999999995 | 0.054125749999999986 | 328880.4293534005 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 1 | ok | 63.062627 | 0.051473500000000005 | 0.05525814999999999 | 0.059404889999999995 | 614393.5513252853 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 2 | ok | 63.33926 | 0.0531275 | 0.05970879999999998 | 0.06918266999999997 | 592290.7436802577 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 4 | ok | 63.29335 | 0.06357550000000001 | 0.06541855 | 0.06942609 | 501956.2174963741 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 8 | ok | 63.869248 | 0.055299 | 0.057569199999999994 | 0.05975629 | 581390.0673140688 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 64 | ok | 61.167172 | 0.050799 | 0.05383945 | 0.057456419999999994 | 625429.00520826 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 1 | ok | 64.060358 | 0.058964 | 0.06299835 | 0.06824476 | 1073917.759377987 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 2 | ok | 64.296651 | 0.07461100000000001 | 0.0835239 | 0.08521313 | 827906.2354793007 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 4 | ok | 63.996932 | 0.0675775 | 0.07176729999999999 | 0.07503665 | 939047.5944535153 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 8 | ok | 64.171373 | 0.067684 | 0.07428894999999999 | 0.07585723 | 935992.1751054161 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 64 | ok | 68.130798 | 0.091322 | 0.09740879999999999 | 0.10703114999999998 | 695137.0169367963 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 1 | ok | 64.139531 | 0.069963 | 0.07511744999999999 | 0.07699073 | 1817519.8691840076 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 2 | ok | 64.414961 | 0.09397749999999999 | 0.10114104999999998 | 0.10376337 | 1350129.2221335967 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 4 | ok | 65.188812 | 0.11097950000000001 | 0.1172343 | 0.12044811999999999 | 1197728.9561829576 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 8 | ok | 65.366183 | 0.0861035 | 0.0911927 | 0.09319445 | 1479471.1815195556 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 64 | ok | 68.945518 | 0.1101665 | 0.12103475 | 0.12466220999999998 | 1146532.5987881152 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 1 | ok | 62.821756 | 0.0905735 | 0.09599284999999999 | 0.0983956 | 10960.477613772498 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 2 | ok | 62.05441 | 0.09114749999999999 | 0.09644029999999999 | 0.10058206999999998 | 10905.705778017413 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 4 | ok | 62.084257 | 0.093303 | 0.10343975 | 0.10765667999999999 | 10599.640756975467 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 8 | ok | 61.949561 | 0.1050335 | 0.11591049999999997 | 0.1214262 | 9541.757114572649 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 64 | ok | 62.284578 | 0.1993285 | 0.2237172 | 0.23973380999999996 | 4906.386152215724 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 1 | ok | 62.509049 | 0.0939855 | 0.10212099999999999 | 0.10451134999999999 | 21071.351811504115 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 2 | ok | 62.807754 | 0.098633 | 0.10499915 | 0.10599625 | 20145.446091692804 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 4 | ok | 62.884739 | 0.1015195 | 0.1115372 | 0.11524096999999998 | 19353.945930881255 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 8 | ok | 62.891316 | 0.109213 | 0.11745795 | 0.12339620999999999 | 18303.248588682258 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 64 | ok | 62.100876 | 0.20171699999999998 | 0.22465875 | 0.23290629 | 9803.052749050477 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 1 | ok | 62.70858 | 0.0997905 | 0.10733089999999999 | 0.10877060999999999 | 39717.76555794525 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 2 | ok | 62.81361 | 0.105712 | 0.1216888 | 0.1234328 | 36879.22808825273 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 4 | ok | 62.52903 | 0.100944 | 0.10719554999999999 | 0.11019052 | 39406.55307333677 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 8 | ok | 63.256082 | 0.1231875 | 0.13269635 | 0.13987897 | 32688.86153194388 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 64 | ok | 63.720498 | 0.1916115 | 0.21385579999999998 | 0.2408436599999999 | 20409.08583928845 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 1 | ok | 63.394044 | 0.1442185 | 0.1520712 | 0.15494549999999999 | 55176.59199296609 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 2 | ok | 63.394876 | 0.1545045 | 0.1621516 | 0.16580864 | 51738.71843711818 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 4 | ok | 62.931859 | 0.15509 | 0.1666413 | 0.17333467 | 51193.25723370324 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 8 | ok | 63.143487 | 0.1883195 | 0.20249535 | 0.21362019999999998 | 42162.08877733737 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 64 | ok | 60.69594 | 0.384743 | 0.441225 | 0.46762013999999996 | 21517.58468124415 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 1 | ok | 62.447541 | 0.141458 | 0.15041745 | 0.1519809 | 112197.90363826946 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 2 | ok | 63.207545 | 0.1663945 | 0.17606855 | 0.18172407 | 95948.37593581149 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 4 | ok | 62.602089 | 0.164115 | 0.17398365 | 0.17787313 | 97232.64941103145 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 8 | ok | 63.827363 | 0.19065949999999998 | 0.20344445 | 0.20716713 | 83913.45501903156 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 64 | ok | 61.051971 | 0.3651295 | 0.43250849999999996 | 0.4927708199999999 | 41959.24498534574 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 1 | ok | 63.385598 | 0.1588325 | 0.16648015 | 0.17051092999999998 | 200345.69649930956 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 2 | ok | 63.082784 | 0.1835055 | 0.1905015 | 0.19176171 | 173707.03071607143 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 4 | ok | 63.935132 | 0.184008 | 0.19522425000000002 | 0.19898825 | 174391.46371224275 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 8 | ok | 62.925361 | 0.2017545 | 0.2121978 | 0.21505949 | 159669.14955520665 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 64 | ok | 60.839953 | 0.388953 | 0.45820025000000003 | 0.47380131999999997 | 83774.17310309748 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 1 | ok | 63.256226 | 0.1704995 | 0.18341365 | 0.18672717 | 371495.43397283956 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 2 | ok | 63.921269 | 0.2347245 | 0.2451976 | 0.25050911 | 278980.2261431149 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 4 | ok | 63.884122 | 0.212844 | 0.22355845000000002 | 0.22676137 | 301712.482335677 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 8 | ok | 63.695727 | 0.206694 | 0.2299115 | 0.24670078 | 303122.8186405756 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 64 | ok | 66.585116 | 0.39840200000000003 | 0.49268595000000004 | 0.50139187 | 159171.7102129792 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 1 | ok | 63.754559 | 0.212405 | 0.2204438 | 0.22818852999999997 | 600073.6777962518 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 2 | ok | 64.232675 | 0.26226550000000004 | 0.300169 | 0.3042459 | 467973.33107731194 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 4 | ok | 64.974769 | 0.265147 | 0.29269205 | 0.29972232 | 484339.5613851203 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 8 | ok | 64.543048 | 0.25477550000000004 | 0.2783772 | 0.29448925 | 498231.7831863432 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 64 | ok | 72.847665 | 0.42443200000000003 | 0.5240126 | 0.5284196 | 296367.7126826164 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 1 | ok | 1372.197283 | 0.0653745 | 0.0670975 | 0.07479615999999997 | 15199.383999365275 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 2 | ok | 1357.460542 | 0.066341 | 0.06981615 | 0.07289471 | 14999.45251998302 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 4 | ok | 1320.901168 | 0.07182749999999999 | 0.07645335 | 0.08012401 | 13840.72317224946 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 8 | ok | 1379.835448 | 0.0661935 | 0.07195800000000001 | 0.07353116999999999 | 14949.26965343108 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 64 | ok | 1564.061989 | 0.066279 | 0.06878255 | 0.07080513999999999 | 15024.020403821632 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 1 | ok | 1362.446868 | 0.070157 | 0.0724054 | 0.07614930999999998 | 28337.450292570007 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 2 | ok | 1323.768735 | 0.074988 | 0.07728934999999999 | 0.08165848 | 26529.231100642086 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 4 | ok | 1345.177633 | 0.0725185 | 0.0740272 | 0.08033981 | 27494.74987751089 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 8 | ok | 1414.16465 | 0.0695655 | 0.07206975 | 0.07716323999999998 | 28646.277201430592 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 64 | ok | 1576.858424 | 0.072823 | 0.0744663 | 0.07829979999999999 | 27393.845424748422 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 1 | ok | 1322.566958 | 0.07020799999999999 | 0.072413 | 0.07598187999999999 | 56716.88095722214 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 2 | ok | 1365.708645 | 0.074306 | 0.07608325 | 0.07670769000000001 | 53778.28771545263 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 4 | ok | 1365.269564 | 0.0696205 | 0.0712411 | 0.07230069 | 57388.72684544211 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 8 | ok | 1380.578321 | 0.0741105 | 0.07616325 | 0.07965150999999998 | 53728.589492890766 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 64 | ok | 1647.761649 | 0.0728785 | 0.07441195 | 0.07498938999999999 | 54796.742443255236 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 1 | ok | 1376.952816 | 0.070601 | 0.0726478 | 0.07466481999999999 | 113004.08651027842 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 2 | ok | 1367.630189 | 0.070016 | 0.07216209999999999 | 0.07901485999999999 | 113515.17328942569 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 4 | ok | 1403.368796 | 0.074742 | 0.07641025 | 0.07958617999999999 | 106693.26885503452 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 8 | ok | 1347.820687 | 0.070227 | 0.07205310000000001 | 0.07328703 | 113578.05808211524 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 64 | ok | 1633.87862 | 0.071138 | 0.0734812 | 0.07891536 | 111924.24071994149 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 1 | ok | 1374.679988 | 0.07272500000000001 | 0.07591165 | 0.07978472 | 218860.88381496407 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 2 | ok | 1310.773305 | 0.07051199999999999 | 0.07257225 | 0.07538260999999999 | 226573.24674081465 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 4 | ok | 1352.874761 | 0.07729349999999999 | 0.07946805 | 0.08107782 | 206838.1199552506 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 8 | ok | 1401.744051 | 0.07199 | 0.0750986 | 0.08252098999999997 | 221088.65157478684 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 64 | ok | 1577.846272 | 0.070477 | 0.07254419999999999 | 0.07426732 | 225905.74082963882 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 1 | ok | 1396.986616 | 0.0753805 | 0.07865865 | 0.08600795999999998 | 420342.91575066943 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 2 | ok | 1310.361331 | 0.078492 | 0.0805766 | 0.08352894 | 406685.3996128863 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 4 | ok | 1404.267559 | 0.087614 | 0.08955724999999999 | 0.09527803 | 363662.3159016348 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 8 | ok | 1395.798836 | 0.078813 | 0.08026865 | 0.08506552999999999 | 404148.2789724429 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 64 | ok | 1608.626756 | 0.0810225 | 0.08297645 | 0.08636836999999999 | 394110.6081274475 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 1 | ok | 1361.031788 | 0.0841355 | 0.08677595 | 0.09045044999999999 | 756625.4975699317 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 2 | ok | 1374.18236 | 0.104019 | 0.10535894999999999 | 0.10966849999999999 | 614781.1840070823 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 4 | ok | 1412.718004 | 0.0949565 | 0.0970635 | 0.09877045 | 672348.8549268673 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 8 | ok | 1403.403868 | 0.094624 | 0.0978152 | 0.10569135999999998 | 671878.9072645437 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 64 | ok | 1498.367226 | 0.129995 | 0.13496334999999998 | 0.13623529 | 490152.67949620885 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 1 | ok | 1375.323648 | 0.089336 | 0.09201725000000001 | 0.09640193999999999 | 1426545.2884664035 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 2 | ok | 1363.255982 | 0.124487 | 0.12989565 | 0.13389024 | 1022511.2236583534 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 4 | ok | 1393.182627 | 0.1367345 | 0.1415743 | 0.14972686 | 932466.5473983674 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 8 | ok | 1439.710623 | 0.1128275 | 0.11735444999999999 | 0.12462327999999999 | 1129015.0155468895 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 64 | ok | 1598.004984 | 0.1387945 | 0.1437313 | 0.14466065 | 920574.4211780446 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 1 | ok | 1425.116412 | 0.12617 | 0.13281754999999998 | 0.13905235 | 7879.866704174831 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 2 | ok | 1361.595895 | 0.1223755 | 0.1277232 | 0.13396097 | 8121.729785501868 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 4 | ok | 1333.211058 | 0.132067 | 0.14596815 | 0.14866373 | 7454.6700153044385 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 8 | ok | 1392.392797 | 0.136877 | 0.15008095 | 0.15319872999999998 | 7277.29476755229 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 64 | ok | 1549.249921 | 0.2513615 | 0.2778303 | 0.29699350999999996 | 4014.7069958437346 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 1 | ok | 1393.443759 | 0.12535849999999998 | 0.1332087 | 0.13566361999999998 | 15830.166376631634 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 2 | ok | 1315.855751 | 0.1266965 | 0.13071885 | 0.13758644999999997 | 15696.312763472417 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 4 | ok | 1375.794855 | 0.1279765 | 0.13933664999999998 | 0.14247996 | 15481.074386562428 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 8 | ok | 1417.505392 | 0.14019900000000002 | 0.1521767 | 0.16054058999999998 | 14138.034437141521 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 64 | ok | 1535.66082 | 0.2522485 | 0.27061235 | 0.2744348 | 7911.264097378169 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 1 | ok | 1374.400051 | 0.127382 | 0.13304844999999998 | 0.13767947 | 31278.590586708164 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 2 | ok | 1367.189657 | 0.1375595 | 0.1535628 | 0.15575782999999999 | 28515.18847256397 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 4 | ok | 1378.676845 | 0.1405085 | 0.1523961 | 0.15340502 | 28072.91771940108 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 8 | ok | 1350.041575 | 0.1489925 | 0.15826455 | 0.16744218 | 26755.67029576119 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 64 | ok | 1583.841728 | 0.25397349999999996 | 0.26902509999999996 | 0.26974561 | 15837.145994255232 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 1 | ok | 1362.105953 | 0.1873325 | 0.1977861 | 0.19941674999999998 | 42411.89220975183 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 2 | ok | 1368.972862 | 0.1971635 | 0.20393275 | 0.20756022 | 40461.27880906272 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 4 | ok | 1405.802313 | 0.2068955 | 0.2183084 | 0.22159952 | 38509.57916154248 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 8 | ok | 1387.873171 | 0.24078349999999998 | 0.2598872 | 0.26388153999999997 | 33060.67915726676 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 64 | ok | 1548.778458 | 0.439556 | 0.5182658 | 0.5570255000000001 | 18436.5016922404 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 1 | ok | 1334.296015 | 0.188125 | 0.19581965 | 0.19988985 | 84690.70478228296 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 2 | ok | 1367.099209 | 0.220111 | 0.2320333 | 0.23669208 | 72649.88079062686 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 4 | ok | 1399.774006 | 0.223358 | 0.2405432 | 0.24818705999999996 | 70848.80051209513 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 8 | ok | 1389.432103 | 0.247785 | 0.2671548 | 0.26916216 | 64239.649868212364 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 64 | ok | 1600.059376 | 0.42095550000000004 | 0.49817155 | 0.50793333 | 37168.282371901376 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 1 | ok | 1367.990268 | 0.204721 | 0.2126698 | 0.21380638 | 155848.50745832513 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 2 | ok | 1364.852185 | 0.225957 | 0.23149409999999998 | 0.23400428 | 142535.1534003409 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 4 | ok | 1389.192042 | 0.2363135 | 0.25055425 | 0.26099861999999996 | 134680.18002699665 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 8 | ok | 1364.470246 | 0.24755100000000002 | 0.26066754999999997 | 0.26617711 | 129929.00922552809 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 64 | ok | 1583.008721 | 0.435291 | 0.5140123 | 0.51613243 | 72106.62478713898 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 1 | ok | 1396.568598 | 0.22107500000000002 | 0.2293639 | 0.23289917 | 288120.14033611736 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 2 | ok | 1371.956701 | 0.2589915 | 0.27797485 | 0.28235694 | 242510.64602789914 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 4 | ok | 1431.123674 | 0.2606195 | 0.27706815 | 0.28090169 | 246635.6774076247 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 8 | ok | 1398.89784 | 0.268977 | 0.2878128 | 0.29379422 | 238283.71980842884 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 64 | ok | 1578.888548 | 0.427023 | 0.5019971 | 0.5069496 | 144763.41857365327 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 1 | ok | 1337.22099 | 0.25970649999999995 | 0.2684317 | 0.27164245000000004 | 491646.1634852551 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 2 | ok | 1364.64498 | 0.315135 | 0.3575437 | 0.35942461000000003 | 395089.1163628195 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 4 | ok | 1324.354247 | 0.289293 | 0.3112506 | 0.3128546 | 439472.0622116651 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 8 | ok | 1435.886057 | 0.3109465 | 0.33888925 | 0.34597470999999996 | 410739.5294632129 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 64 | ok | 1545.158586 | 0.4503505 | 0.5304656 | 0.53819261 | 277585.34751863853 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.09543850000000001 | 0.09936555 | 0.10903704999999997 | 10412.096190276478 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.091258 | 0.0970207 | 0.10676073 | 10842.104053407338 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.090028 | 0.09520295 | 0.10990801999999997 | 10977.184580480545 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.09157 | 0.09612559999999999 | 0.09767607 | 10879.53891643709 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0946205 | 0.09784725 | 0.10057315 | 10538.136250514262 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.11773249999999999 | 0.12350865 | 0.13401264 | 16844.692607942474 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.11806749999999999 | 0.12308909999999999 | 0.12781362 | 16846.12847329265 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1155395 | 0.11999194999999999 | 0.12927748 | 17194.101460329428 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.1110525 | 0.117031 | 0.12525663 | 17851.12447803312 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.0974585 | 0.10253505 | 0.10640638999999999 | 20393.542344029676 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.111293 | 0.11615505 | 0.12733962999999998 | 35638.05845283071 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.112598 | 0.1174141 | 0.12613401999999996 | 35257.860122901846 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.1141475 | 0.11937255 | 0.13206753999999998 | 34724.9109913719 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.11279249999999999 | 0.11728204999999998 | 0.12317816 | 35233.784971733694 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.10299849999999999 | 0.1088268 | 0.11017048 | 38581.36327247123 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.119502 | 0.1274722 | 0.12843024 | 66474.14645949556 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1164435 | 0.1210028 | 0.12761625 | 68384.01728747957 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.11291100000000001 | 0.117695 | 0.12263314999999998 | 70488.09658701923 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.114117 | 0.11825269999999999 | 0.12518752 | 69761.47157638603 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.100268 | 0.10428979999999999 | 0.10731269999999998 | 79525.8351604394 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.1201815 | 0.12494444999999998 | 0.13101227999999998 | 132539.29044428165 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1150365 | 0.12308699999999999 | 0.12833471999999999 | 137969.19010015874 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1198145 | 0.1258019 | 0.13194651999999998 | 132849.8484515354 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.115716 | 0.12408019999999999 | 0.13152694999999998 | 136878.86083936854 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.1047815 | 0.10841764999999999 | 0.11230306 | 151911.18512561632 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.119665 | 0.12435774999999999 | 0.12960211 | 265889.8694131761 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.119668 | 0.12455904999999999 | 0.13107588999999997 | 265712.0959619235 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.135782 | 0.14236125 | 0.14485745 | 234448.03605810797 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1203305 | 0.12718455 | 0.13753958 | 263503.7434000547 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.1097545 | 0.11573834999999999 | 0.12018663 | 289353.84032225335 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.1289765 | 0.13649194999999997 | 0.14416816999999998 | 492063.85574163945 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.153528 | 0.15947884999999998 | 0.16728849999999998 | 414405.9956259447 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1457435 | 0.15189999999999998 | 0.16049622 | 436355.68278960005 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1359195 | 0.1445063 | 0.14820098999999998 | 467618.718894222 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.1667145 | 0.1728701 | 0.17550541 | 382503.0665988042 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.14893 | 0.156495 | 0.16218699 | 853982.1320251038 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1794375 | 0.1856623 | 0.19521477999999998 | 711116.3259641682 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.19877699999999998 | 0.21034175 | 0.21729294 | 641016.2029865946 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1708305 | 0.18247215 | 0.185288 | 744396.6415149773 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.1705795 | 0.17684945 | 0.18565850999999997 | 746805.6845448447 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.16075299999999998 | 0.17028085 | 0.17212270999999998 | 6177.561145809099 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1739775 | 0.1797203 | 0.18497743 | 5724.938024683414 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.175087 | 0.1819554 | 0.19023605999999998 | 5715.069495245062 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.188952 | 0.20275649999999998 | 0.20544208 | 5272.558589198299 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.28080150000000004 | 0.3119829 | 0.31638082 | 3470.110515385672 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1671475 | 0.17662904999999998 | 0.18404152000000001 | 11885.090664819882 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.16891250000000002 | 0.1755506 | 0.17945531999999997 | 11793.781021329643 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.181235 | 0.1920141 | 0.20172895 | 10963.619968041048 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.188817 | 0.2009381 | 0.20935703 | 10515.456090557425 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.304146 | 0.31911735 | 0.32070178 | 6680.150327430909 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.17692049999999998 | 0.1851252 | 0.19145679 | 22475.64005846813 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.17587 | 0.1924204 | 0.19816974999999998 | 22367.906885087785 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.182409 | 0.1894061 | 0.19125022 | 21847.150345097587 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.19150250000000002 | 0.2027901 | 0.20724025 | 20818.74936567873 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.300083 | 0.31513175 | 0.3198315 | 13452.889159291963 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.256942 | 0.26675005 | 0.27403999 | 31048.196425793725 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.2748305 | 0.2862852 | 0.28988524 | 28952.02360168964 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.2976205 | 0.30933245 | 0.31327094 | 26793.92351326212 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.3160575 | 0.34380095 | 0.3518517 | 25107.943757954905 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.5467805 | 0.6374136499999999 | 0.6457554799999999 | 14490.342766160065 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.273997 | 0.28110535 | 0.28557461 | 58229.04935181243 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.288378 | 0.29908714999999997 | 0.3008257 | 55250.053454426714 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.29854 | 0.31240575 | 0.31492567 | 53724.09734618405 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.327623 | 0.34866555 | 0.35496303 | 48264.80485181951 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.5318970000000001 | 0.62652155 | 0.6429911 | 29513.63339869004 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.2773155 | 0.2851242 | 0.28976836 | 115031.19034779033 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.312148 | 0.31882435 | 0.324142 | 103745.79375647439 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.322238 | 0.3343205 | 0.33756008 | 99593.8624535309 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.341389 | 0.35562895 | 0.35891943 | 94281.07498338885 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.5313135 | 0.6270675499999999 | 2.690480059999992 | 51613.371076041054 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.320154 | 0.32932405 | 0.33382214 | 200309.5784534999 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.335317 | 0.36234205 | 0.36630916 | 187157.9250268747 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.335094 | 0.3616102 | 0.3627421 | 189950.0627221043 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.3490475 | 0.37789255 | 0.39988268 | 182422.0922274792 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.5848150000000001 | 0.6707926 | 3.26864043999999 | 95219.12226120419 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.3374505 | 0.34654694999999996 | 0.34909736999999996 | 378048.58526835626 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.410642 | 0.45937945 | 0.46143718 | 312306.18107609975 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.388264 | 0.43388059999999995 | 0.44524236 | 329365.4743259098 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.393141 | 0.435059 | 0.4511802 | 325912.26024379156 | - |
| `full_mlp_capacity_search_hd32_depth5` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.560569 | 0.66288605 | 0.67030211 | 223714.36597181455 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 1 | ok | 66.436856 | 0.047109 | 0.055904499999999996 | 0.06018889999999999 | 20777.062123415748 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 2 | ok | 66.51028 | 0.045453 | 0.054041499999999985 | 0.059995829999999986 | 21433.98504079316 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 4 | ok | 66.676291 | 0.045151 | 0.05253379999999999 | 0.05777167999999999 | 21761.497469573074 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 8 | ok | 66.035699 | 0.045078999999999994 | 0.05473255 | 0.05811828999999999 | 21684.560766210943 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 1 | 64 | ok | 66.350803 | 0.045866000000000004 | 0.055739449999999996 | 0.06329541999999999 | 21241.290009031796 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 1 | ok | 67.098284 | 0.0510635 | 0.0532517 | 0.05588778999999999 | 38883.31777352567 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 2 | ok | 66.896681 | 0.0493745 | 0.05102325 | 0.054187659999999985 | 40359.033966162984 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 4 | ok | 66.653447 | 0.0560205 | 0.057349149999999995 | 0.06009708 | 36215.049308600384 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 8 | ok | 67.104706 | 0.0543125 | 0.05683725 | 0.05781586 | 37512.782480630274 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 2 | 64 | ok | 67.098507 | 0.0492905 | 0.051997249999999995 | 0.054420069999999994 | 40347.21196730585 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 1 | ok | 67.276664 | 0.051111000000000004 | 0.0541073 | 0.05813869 | 77541.68059186015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 2 | ok | 66.875971 | 0.0505695 | 0.05246765 | 0.05700941999999999 | 78475.38031131183 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 4 | ok | 67.846983 | 0.0535165 | 0.057194249999999995 | 0.060697879999999996 | 73699.57106849637 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 8 | ok | 66.564769 | 0.049103499999999994 | 0.05170655 | 0.05521696999999999 | 80656.21899776583 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 4 | 64 | ok | 65.483901 | 0.049905 | 0.0516987 | 0.056976949999999985 | 79632.67041789633 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 1 | ok | 67.192764 | 0.053687 | 0.0556223 | 0.059108959999999995 | 148008.41723868836 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 2 | ok | 67.480964 | 0.051957 | 0.05574199999999999 | 0.059215319999999995 | 152311.55633471376 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 4 | ok | 67.268317 | 0.0535 | 0.0600068 | 0.06163523 | 144978.27862940435 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 8 | ok | 66.788709 | 0.055848499999999995 | 0.05822835 | 0.059750689999999995 | 144030.63243490623 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 8 | 64 | ok | 64.958396 | 0.050807 | 0.05380214999999999 | 0.05435573 | 156700.26842755981 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 1 | ok | 67.072429 | 0.0537865 | 0.0556606 | 0.05827669 | 298423.057955996 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 2 | ok | 67.405941 | 0.052762500000000004 | 0.055440249999999996 | 0.05813590999999999 | 300996.18449711625 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 4 | ok | 67.340042 | 0.0529005 | 0.05614305 | 0.058114309999999995 | 297720.05978218804 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 8 | ok | 67.515296 | 0.056468 | 0.060384799999999995 | 0.0627534 | 279015.229697544 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 16 | 64 | ok | 65.107285 | 0.0512675 | 0.05274265 | 0.05413601999999999 | 310637.99219133746 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 1 | ok | 67.79307 | 0.054602 | 0.05637175 | 0.058579 | 583433.3374538313 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 2 | ok | 67.146264 | 0.054393 | 0.057375050000000004 | 0.05873033 | 579615.0124635341 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 4 | ok | 68.049527 | 0.060742500000000005 | 0.0623752 | 0.06733773 | 523689.0753221997 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 8 | ok | 67.918768 | 0.056791499999999995 | 0.058712099999999996 | 0.06664315999999998 | 560491.5791394842 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 32 | 64 | ok | 65.332318 | 0.054568 | 0.05683845 | 0.06076575999999999 | 582173.6252788521 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 1 | ok | 67.345118 | 0.065234 | 0.06725995 | 0.07135016999999999 | 976425.7264988822 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 2 | ok | 68.768908 | 0.08284549999999999 | 0.0857007 | 0.08761134 | 791036.5669012995 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 4 | ok | 67.626406 | 0.07205049999999999 | 0.07437474999999999 | 0.07633961999999998 | 888767.9177000909 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 8 | ok | 67.746298 | 0.06808149999999999 | 0.07004144999999999 | 0.07523090999999998 | 942352.19353087 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 64 | 64 | ok | 72.786272 | 0.0985805 | 0.1011973 | 0.10270446 | 648142.7469585901 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 1 | ok | 68.841756 | 0.0751415 | 0.08098040000000001 | 0.08319298 | 1689489.737141795 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 2 | ok | 69.548634 | 0.0987405 | 0.1031878 | 0.10520347 | 1288785.1327358068 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 4 | ok | 69.249994 | 0.1099885 | 0.1140239 | 0.11839788999999999 | 1161708.1393447132 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 8 | ok | 69.336774 | 0.0911055 | 0.09544535 | 0.09772560999999999 | 1402845.1892383362 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `fp32` | 128 | 64 | ok | 76.246607 | 0.1246205 | 0.1280933 | 0.12980432 | 1028038.2989243026 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 1 | ok | 67.181764 | 0.1005665 | 0.10446734999999999 | 0.1055406 | 9918.895177909271 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 2 | ok | 66.477101 | 0.101742 | 0.10641374999999999 | 0.10926546 | 9769.070887113696 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 4 | ok | 66.826022 | 0.105752 | 0.1154834 | 0.1173327 | 9324.831821995675 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 8 | ok | 66.91121 | 0.116263 | 0.1247727 | 0.12748811999999998 | 8610.856947316883 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 1 | 64 | ok | 65.361333 | 0.206401 | 0.23397035 | 0.2876524499999998 | 4703.015394756533 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 1 | ok | 66.631956 | 0.09899250000000001 | 0.1053316 | 0.10788211 | 19953.376939443297 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 2 | ok | 67.131051 | 0.1042265 | 0.1097655 | 0.11174434999999999 | 19082.581972570315 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 4 | ok | 67.592456 | 0.1103365 | 0.1175072 | 0.12210581 | 17878.686674585733 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 8 | ok | 67.001136 | 0.1207645 | 0.12910844999999999 | 0.13661832 | 16552.161571607878 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 2 | 64 | ok | 65.425287 | 0.22784 | 0.24290265000000003 | 0.24755261 | 8877.646237693474 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 1 | ok | 67.618537 | 0.1105655 | 0.11787774999999999 | 0.12172965999999999 | 35817.11280760856 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 2 | ok | 74.584507 | 0.1109615 | 0.12839845 | 0.13253505999999998 | 35062.52349189074 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 4 | ok | 67.048224 | 0.111736 | 0.12078439999999997 | 0.12601666 | 35470.40136887373 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 8 | ok | 66.837994 | 0.12319 | 0.1355653 | 0.13890331 | 32239.708681992346 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 4 | 64 | ok | 64.885985 | 0.235931 | 0.2500032 | 0.26081105 | 17233.639029366983 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 1 | ok | 67.383305 | 0.1703825 | 0.17839795 | 0.18082541 | 46724.71979477569 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 2 | ok | 66.793453 | 0.183501 | 0.1906766 | 0.19524857999999998 | 43390.136445707816 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 4 | ok | 66.770036 | 0.1879785 | 0.1935238 | 0.19989776 | 42790.03890684288 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 8 | ok | 67.612323 | 0.223134 | 0.24662235 | 0.251873 | 35815.57984886542 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 8 | 64 | ok | 65.08605 | 0.45994599999999997 | 0.57200105 | 0.57824644 | 18058.049949107903 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 1 | ok | 67.851257 | 0.1770825 | 0.1823605 | 0.18686339999999999 | 90074.41835851758 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 2 | ok | 67.302432 | 0.2013135 | 0.20698435 | 0.20982958 | 79381.80624568115 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 4 | ok | 67.377535 | 0.199569 | 0.21094279999999999 | 0.21357168999999998 | 80682.81055746679 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 8 | ok | 67.385861 | 0.22955550000000002 | 0.249462 | 0.25041445 | 69503.18428838818 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 16 | 64 | ok | 66.869318 | 0.42241150000000005 | 0.5204791000000001 | 0.52671917 | 36297.999680759094 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 1 | ok | 67.292807 | 0.18167250000000001 | 0.18819875 | 0.18985947 | 175839.49898051558 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 2 | ok | 67.229533 | 0.2163525 | 0.2220558 | 0.22632087999999997 | 150191.8842125707 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 4 | ok | 67.417785 | 0.2180295 | 0.2289646 | 0.23149096 | 146710.07695585393 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 8 | ok | 67.033848 | 0.23429450000000002 | 0.2560743 | 0.2800644599999999 | 134218.5268879136 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 32 | 64 | ok | 67.957363 | 0.42403749999999996 | 0.5062243 | 0.5158517699999999 | 73854.70167570317 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 1 | ok | 67.169642 | 0.20293250000000002 | 0.20897385000000002 | 0.2101707 | 314446.4357201717 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 2 | ok | 67.984668 | 0.2681025 | 0.27632255 | 0.28249647 | 248875.723363123 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 4 | ok | 68.288013 | 0.24382500000000001 | 0.26654115 | 0.26808113 | 257850.3132196391 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 8 | ok | 67.959597 | 0.257119 | 0.2833693 | 0.29402889 | 243978.86838026243 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 64 | 64 | ok | 72.57033 | 0.4136565 | 0.54534525 | 1.6770556099999965 | 135056.17746111608 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 1 | ok | 68.888379 | 0.259175 | 0.2658664 | 0.26874154 | 492187.977047429 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 2 | ok | 68.943339 | 0.31040199999999996 | 0.3575165 | 0.36380094 | 411236.1276078555 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 4 | ok | 68.768567 | 0.3085325 | 0.32926574999999997 | 0.33952119999999997 | 429698.12967846426 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 8 | ok | 69.733709 | 0.31089049999999996 | 0.3587424 | 0.36682785999999995 | 407239.00418100826 | - |
| `full_mlp_capacity_search_hd32_depth5` | `jit` | `bf16` | 128 | 64 | ok | 76.309177 | 0.491653 | 0.5952275 | 0.61901058 | 258212.96776186893 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 1 | ok | 1419.750553 | 0.07244800000000001 | 0.0764811 | 0.07966622999999999 | 13705.757130694263 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 2 | ok | 1355.126168 | 0.0781645 | 0.0825639 | 0.08522879 | 12706.557803656237 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 4 | ok | 1382.687501 | 0.0766795 | 0.0781037 | 0.08152735 | 13015.759220949541 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 8 | ok | 1410.613031 | 0.076195 | 0.0801585 | 0.08438923999999999 | 13091.105981143048 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 1 | 64 | ok | 1522.483991 | 0.073738 | 0.07568689999999999 | 0.07943733 | 13491.383997437715 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 1 | ok | 1372.507635 | 0.080111 | 0.08183649999999999 | 0.08664380999999999 | 24841.412423090987 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 2 | ok | 1375.019116 | 0.07875750000000001 | 0.08003335 | 0.08458859999999999 | 25333.714690185137 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 4 | ok | 1376.828935 | 0.078011 | 0.07995555 | 0.08572615999999998 | 25494.439152931962 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 8 | ok | 1409.251282 | 0.0818235 | 0.0838334 | 0.08770504999999998 | 24319.722638427254 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 2 | 64 | ok | 1574.375149 | 0.08079549999999999 | 0.0826966 | 0.08647775999999999 | 24677.33764100939 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 1 | ok | 1360.656659 | 0.080368 | 0.08241765 | 0.08841709999999998 | 49525.48395683854 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 2 | ok | 1318.261471 | 0.079184 | 0.08128869999999999 | 0.08467056999999999 | 50317.554083823 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 4 | ok | 1399.781543 | 0.083291 | 0.08515009999999999 | 0.09091429999999999 | 47832.082604093375 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 8 | ok | 1394.022292 | 0.0836995 | 0.0852684 | 0.08935546 | 47650.803475935514 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 4 | 64 | ok | 1530.470811 | 0.08321500000000001 | 0.08504329999999999 | 0.08943375999999999 | 47916.71308598247 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 1 | ok | 1351.888603 | 0.0789495 | 0.0809604 | 0.08672567999999997 | 100888.93238322863 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 2 | ok | 1362.488141 | 0.08296200000000001 | 0.0856761 | 0.09229333999999999 | 95925.88382512136 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 4 | ok | 1381.526356 | 0.0822865 | 0.0837667 | 0.08744394999999998 | 96980.65284466076 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 8 | ok | 1376.966058 | 0.0831755 | 0.08541875 | 0.09273648999999999 | 95798.44772985221 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 8 | 64 | ok | 1540.178268 | 0.079815 | 0.08185855 | 0.08672775999999999 | 99824.78255042835 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 1 | ok | 1361.740572 | 0.0831655 | 0.08835879999999999 | 0.09003828999999999 | 191037.06620556684 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 2 | ok | 1330.114364 | 0.08074500000000001 | 0.08293235 | 0.08852777999999999 | 196904.36797882096 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 4 | ok | 1356.503589 | 0.081164 | 0.08324555 | 0.09332507999999996 | 195946.11971573118 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 8 | ok | 1407.495371 | 0.0860475 | 0.08760024999999999 | 0.09318063999999998 | 185458.5648474418 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 16 | 64 | ok | 1597.660772 | 0.085992 | 0.08950129999999999 | 0.09501434999999998 | 185235.2244228939 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 1 | ok | 1317.061485 | 0.08597350000000001 | 0.08824425 | 0.09405224999999998 | 370661.7516918392 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 2 | ok | 1364.232293 | 0.0844145 | 0.0871843 | 0.08978973 | 377603.487545811 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 4 | ok | 1379.787464 | 0.10090350000000001 | 0.102918 | 0.10657587999999998 | 316625.28110388236 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 8 | ok | 1400.192581 | 0.08938750000000001 | 0.0909244 | 0.09481906999999999 | 357179.53182692867 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 32 | 64 | ok | 1505.911506 | 0.08498 | 0.08724155 | 0.09413382999999997 | 374654.4398190232 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 1 | ok | 1391.721493 | 0.09100150000000001 | 0.09320475 | 0.09650570999999998 | 701392.9225070414 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 2 | ok | 1355.623831 | 0.11354449999999999 | 0.11604975000000001 | 0.12305974 | 561305.4280339787 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 4 | ok | 1384.788946 | 0.1041835 | 0.10643409999999999 | 0.11200245 | 612248.4902334886 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 8 | ok | 1397.130887 | 0.100326 | 0.10428074999999999 | 0.10949927 | 634854.0004432074 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 64 | 64 | ok | 1543.930773 | 0.1445865 | 0.14912775 | 0.15230362 | 442754.16891099216 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 1 | ok | 1367.503652 | 0.10222149999999999 | 0.10424535 | 0.10915857999999998 | 1249736.627769265 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 2 | ok | 1370.773677 | 0.136622 | 0.14101135 | 0.14963509999999997 | 931376.7436391334 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 4 | ok | 1358.047577 | 0.16257349999999998 | 0.16629069999999999 | 0.17037159999999998 | 786126.3440150298 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 8 | ok | 1413.94153 | 0.119219 | 0.1222429 | 0.12853309 | 1069238.0058644363 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `fp32` | 128 | 64 | ok | 1610.311511 | 0.15498800000000001 | 0.16007995 | 0.17017845999999998 | 823040.7162100312 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 1 | ok | 1365.830154 | 0.13718249999999999 | 0.14454329999999999 | 0.14931053 | 7258.282498742139 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 2 | ok | 1364.889967 | 0.13631 | 0.13919865 | 0.14299613 | 7330.537759453388 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 4 | ok | 1396.527179 | 0.1428485 | 0.15450875 | 0.15738332 | 6929.114740458227 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 8 | ok | 1395.608972 | 0.14870899999999998 | 0.1582139 | 0.17639500999999994 | 6716.695299938731 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 1 | 64 | ok | 1619.448266 | 0.261476 | 0.27416755 | 0.27788791 | 3916.8892504264704 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 1 | ok | 1363.488953 | 0.132475 | 0.137019 | 0.14118818 | 15036.011246936414 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 2 | ok | 1405.034361 | 0.1440075 | 0.14789945 | 0.16038454 | 13796.108145484377 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 4 | ok | 1348.782695 | 0.141028 | 0.1574334 | 0.16119336999999997 | 13950.157204321536 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 8 | ok | 1402.499774 | 0.15056 | 0.1574627 | 0.16168614 | 13240.329726579248 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 2 | 64 | ok | 1520.705364 | 0.2754795 | 0.2918445 | 1.8675000899999938 | 5989.381425670428 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 1 | ok | 1353.687815 | 0.144399 | 0.1495891 | 0.15847888999999998 | 27529.867497371244 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 2 | ok | 1312.057829 | 0.14669949999999998 | 0.16234835 | 0.16658524 | 26754.53569643662 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 4 | ok | 1410.990255 | 0.14334950000000002 | 0.1539888 | 0.16017531 | 27655.177000737975 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 8 | ok | 1412.144979 | 0.16057 | 0.17140385 | 0.17319961 | 24872.05503986804 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 4 | 64 | ok | 1549.278584 | 0.2673335 | 0.2808314 | 0.28391853 | 15086.466195189454 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 1 | ok | 1376.985148 | 0.21679199999999998 | 0.22519935 | 0.22975644999999997 | 36738.82662065986 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 2 | ok | 1359.814728 | 0.2555995 | 0.2645393 | 0.26751644999999996 | 31228.120797865995 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 4 | ok | 1393.169742 | 0.2503255 | 0.26853604999999997 | 0.27319509 | 31711.88609851598 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 8 | ok | 1352.638127 | 0.2884 | 0.307061 | 0.31767701 | 27726.89123045566 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 8 | 64 | ok | 1538.06183 | 0.4911215 | 0.6793856 | 0.6882435499999999 | 15230.2561527284 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 1 | ok | 1334.318681 | 0.2275575 | 0.23615185 | 0.24027550999999997 | 69961.85242544187 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 2 | ok | 1367.676146 | 0.26410999999999996 | 0.2740567 | 0.27904204 | 60417.36618648454 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 4 | ok | 1375.70276 | 0.271131 | 0.28410294999999997 | 0.28870867 | 58945.96798895 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 8 | ok | 1343.234472 | 0.30287600000000003 | 0.3235537 | 0.32951501 | 52752.041009436674 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 16 | 64 | ok | 1582.681108 | 0.48320850000000004 | 0.6782657999999999 | 0.68797682 | 30902.262632767706 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 1 | ok | 1360.202372 | 0.25130450000000004 | 0.25742885 | 0.25986892 | 127125.8925330144 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 2 | ok | 1362.894321 | 0.28798 | 0.2949771 | 0.30090481999999996 | 111642.77104335403 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 4 | ok | 1398.882398 | 0.2821305 | 0.29481265 | 0.29858729 | 113495.71922521011 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 8 | ok | 1409.759586 | 0.29561099999999996 | 0.31836 | 0.33667940999999996 | 108048.59377456417 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 32 | 64 | ok | 1573.279665 | 0.5105744999999999 | 0.60535415 | 0.60986765 | 61916.55018938144 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 1 | ok | 1350.572485 | 0.2568165 | 0.26559105 | 0.26706452 | 248549.7124279827 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 2 | ok | 1362.693367 | 0.31053450000000005 | 0.33267445 | 0.33768893 | 205779.8412922974 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 4 | ok | 1448.755958 | 0.2978775 | 0.31070175 | 0.31976061 | 214347.9421726763 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 8 | ok | 1422.820562 | 0.31465350000000003 | 0.33086675 | 0.33435506 | 204929.6073202905 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 64 | 64 | ok | 1483.822152 | 0.5513765 | 0.8090588 | 3.4346898799999903 | 93465.3387150286 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 1 | ok | 1366.474784 | 0.304769 | 0.31349130000000003 | 0.3166547 | 418639.36841448385 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 2 | ok | 1362.384886 | 0.36702 | 0.40217565 | 0.40381118 | 356047.83124311926 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 4 | ok | 1352.725524 | 0.336904 | 0.3482554 | 0.36359561999999995 | 383871.1958589415 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 8 | ok | 1396.842677 | 0.355877 | 0.38940325 | 0.39639323 | 360167.6017424233 | - |
| `full_mlp_capacity_search_hd32_depth5` | `compile` | `bf16` | 128 | 64 | ok | 1554.626248 | 0.535182 | 0.66838285 | 0.68795106 | 231449.0427864295 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.058896000000000004 | 0.063538 | 0.06968044999999998 | 16774.59440708183 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0554105 | 0.0589024 | 0.06470570999999999 | 17836.79616894157 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.060219 | 0.06404085 | 0.06838102999999998 | 16483.33897063526 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.056534 | 0.0607958 | 0.06380728 | 17509.681978650096 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.055925 | 0.0601479 | 0.06125152 | 17706.448334354405 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.06691 | 0.07064954999999999 | 0.07747449999999999 | 29576.750780974104 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.06765850000000001 | 0.07114405 | 0.07770327999999999 | 29288.51175699439 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0701515 | 0.0730064 | 0.08383655999999996 | 28276.512623766295 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.070052 | 0.07380355 | 0.07803658 | 28364.9720959587 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.059382000000000004 | 0.06795584999999997 | 0.08068131999999997 | 33006.33952763307 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0680425 | 0.07054105 | 0.07887926999999997 | 58286.74279692432 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.07226450000000001 | 0.07633380000000001 | 0.08196417999999998 | 54943.062504326765 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0698745 | 0.07271459999999999 | 0.07440374999999999 | 57045.16322617781 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.07228000000000001 | 0.07557395 | 0.08053995999999998 | 54940.05902210541 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.05901 | 0.06193799999999999 | 0.06297786 | 67312.98349653927 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0687855 | 0.07257155 | 0.08118522999999997 | 115096.35435175001 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.072386 | 0.07499494999999999 | 0.08033705999999999 | 109937.25605953542 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.071898 | 0.07484629999999999 | 0.08272725999999997 | 110466.72190002761 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.082829 | 0.0863548 | 0.09217715 | 101097.43796345277 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.06651 | 0.06941415 | 0.07139939999999999 | 119505.58150818433 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.07501250000000001 | 0.0773369 | 0.08518349999999998 | 211614.40364019095 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0688695 | 0.0718234 | 0.07846378999999998 | 230412.49020026874 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0710665 | 0.0747263 | 0.07971174999999998 | 223307.99531946442 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0729195 | 0.07591469999999999 | 0.08337164999999998 | 217859.82136583948 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.069074 | 0.07311834999999998 | 0.07779315999999999 | 229821.5923706126 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0752475 | 0.07795845 | 0.08740127999999997 | 422373.21478021145 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0764805 | 0.08040445 | 0.08631732 | 414761.571718238 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0722835 | 0.0759009 | 0.08638128999999997 | 438359.88744013995 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.07971249999999999 | 0.0831707 | 0.09123108999999997 | 405449.7514719727 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.079218 | 0.08198124999999999 | 0.08635519 | 402344.15764849953 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.07818549999999999 | 0.08154 | 0.08821785999999998 | 812498.2543982816 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0965785 | 0.1004755 | 0.10519584 | 658699.6938693172 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.087805 | 0.09154135 | 0.09850790999999998 | 723936.0402508437 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.09002399999999999 | 0.09337065 | 0.09815971999999999 | 727116.397247319 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.06852849999999999 | 0.07288575 | 0.0751278 | 927234.4103643945 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0862955 | 0.091611 | 0.09711112 | 1474861.6741062684 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1178205 | 0.1225159 | 0.12569468 | 1081594.6558408055 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.115965 | 0.1217634 | 0.1341265 | 1095603.3778821644 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.108103 | 0.11408655 | 0.12159131999999999 | 1173236.2089291709 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.11876149999999999 | 0.12266709999999999 | 0.12562987 | 1071800.9504864805 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0700515 | 0.07417295 | 0.07751074 | 14155.724291789113 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.07119149999999999 | 0.0741514 | 0.07730748 | 13987.629899621972 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0725935 | 0.07637795 | 0.08009934999999999 | 13622.846057752693 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0704935 | 0.074406 | 0.07772544999999999 | 14095.362456607429 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.089868 | 0.09622644999999999 | 0.10148743999999998 | 11854.50445089224 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.1171915 | 0.12297545 | 0.12484938999999999 | 16967.229831647754 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1196235 | 0.12657425 | 0.13064708 | 16600.987625955844 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.12178449999999999 | 0.130828 | 0.14143284999999997 | 16277.281537721296 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.13646350000000002 | 0.14872485 | 0.15320467 | 14581.723268055819 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.236441 | 0.2528236 | 0.25776319 | 8607.826149456825 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.11169499999999999 | 0.11740474999999999 | 0.12460091999999999 | 35527.480950608806 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.1181825 | 0.13428934999999997 | 0.13889441 | 33071.446063861295 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.12976100000000002 | 0.13855605000000001 | 0.14166832999999998 | 30677.65560742295 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.13823950000000002 | 0.15325825 | 0.15687490999999998 | 28707.156005020308 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.237278 | 0.2548805 | 0.30297519999999983 | 16880.5887409174 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.114312 | 0.12013979999999999 | 0.12623017999999997 | 69401.22531333352 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.12226899999999999 | 0.14305459999999998 | 0.14885300999999998 | 63669.631018570835 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.134498 | 0.16157285 | 0.16762202 | 57526.7661256514 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.139656 | 0.15570124999999999 | 0.16500032999999997 | 57030.21432239694 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.2315895 | 0.25054875 | 0.30137594999999984 | 34607.41692316531 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.124917 | 0.13310925 | 0.13864781 | 126910.94167442805 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.127647 | 0.14598 | 0.15062087999999998 | 121780.48549921701 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.13045099999999998 | 0.15756025 | 0.16868016999999996 | 118474.20039170534 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1409665 | 0.21528034999999998 | 0.2269204 | 104862.63780916783 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.229728 | 0.2557306 | 0.3193664199999999 | 68992.3588375374 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1274805 | 0.13636125000000002 | 0.14004124999999998 | 248209.7869118979 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.142036 | 0.15034539999999996 | 0.1602198 | 223660.13947166724 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1459775 | 0.1712569 | 0.19436287 | 211857.84232850836 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.15526600000000002 | 0.2181957 | 0.22364417 | 196614.78494029053 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.24462099999999998 | 0.25811660000000003 | 0.25842321 | 133334.88890703724 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.132857 | 0.13834585 | 0.14766086999999997 | 477425.2456502085 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.16558450000000002 | 0.1871088 | 0.19148161 | 377286.8594523917 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1608615 | 0.1698264 | 0.17581785 | 397119.4937520693 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.163234 | 0.2336356 | 0.24305681999999998 | 368895.47162363806 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.250324 | 0.27609815 | 0.28112925 | 253460.82941096096 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.16087200000000001 | 0.16943785 | 0.18261608999999998 | 787648.1993931418 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.206752 | 0.2285942 | 0.23357466999999998 | 612618.5249332867 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2111865 | 0.2466349 | 0.24977048 | 595019.7774347117 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.2102105 | 0.23333394999999998 | 0.24033644 | 604739.2279407665 | - |
| `full_mlp_capacity_search_hd512_depth2` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.29529 | 0.8956733499999997 | 1.0574179299999997 | 335161.6751841373 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 1 | ok | 53.680311 | 0.038977 | 0.046273049999999996 | 0.05061782999999999 | 24974.039485954352 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 2 | ok | 53.492811 | 0.039189 | 0.04311275 | 0.05118501999999999 | 24979.142416082574 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 4 | ok | 54.051987 | 0.037609000000000004 | 0.04427239999999999 | 0.05077444999999999 | 25854.998960629044 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 8 | ok | 53.283473 | 0.038148 | 0.04339529999999999 | 0.05080234999999999 | 25540.61826685451 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 1 | 64 | ok | 53.30077 | 0.035824499999999995 | 0.043381199999999995 | 0.04837209 | 27028.97019082994 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 1 | ok | 53.518781 | 0.036577 | 0.03901275 | 0.04000079 | 54298.07252702143 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 2 | ok | 53.658268 | 0.0391285 | 0.0421435 | 0.04264573 | 51272.53299643861 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 4 | ok | 53.431066 | 0.039652999999999994 | 0.042997499999999994 | 0.04422466 | 50539.739144190375 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 8 | ok | 53.607746 | 0.0367435 | 0.0429075 | 0.04479771 | 51917.19829877724 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 2 | 64 | ok | 52.111856 | 0.036563 | 0.0387123 | 0.03978514 | 54253.73713304807 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 1 | ok | 53.963443 | 0.04284 | 0.0453519 | 0.04639769 | 92892.25612995999 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 2 | ok | 61.980144 | 0.0384745 | 0.04159614999999999 | 0.04419914999999999 | 102981.46796993147 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 4 | ok | 53.659732 | 0.041892 | 0.0442844 | 0.04621797 | 94658.11146655233 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 8 | ok | 53.862824 | 0.041641 | 0.0430425 | 0.04490292 | 95675.46880979718 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 4 | 64 | ok | 53.760383 | 0.036685499999999996 | 0.03934955 | 0.04200502999999999 | 108014.63166200493 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 1 | ok | 53.650876 | 0.0425905 | 0.0450445 | 0.04838127999999999 | 186296.49553004844 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 2 | ok | 53.971004 | 0.0437455 | 0.04669585 | 0.051828449999999984 | 180770.125928989 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 4 | ok | 54.311511 | 0.042422 | 0.04602955 | 0.04782789 | 186224.59430972132 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 8 | ok | 53.38252 | 0.042122 | 0.043743699999999996 | 0.045498620000000004 | 189370.71637521725 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 8 | 64 | ok | 53.741577 | 0.037442 | 0.0404117 | 0.044582039999999996 | 210344.31260530362 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 1 | ok | 54.149161 | 0.043327500000000005 | 0.045429599999999994 | 0.047541509999999995 | 366367.9699138623 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 2 | ok | 54.645736 | 0.042995000000000005 | 0.0468993 | 0.050398899999999996 | 367621.37938893976 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 4 | ok | 53.888759 | 0.043453000000000006 | 0.0446969 | 0.04708043 | 366437.10908603715 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 8 | ok | 53.550491 | 0.043352 | 0.04439595 | 0.04559832999999999 | 367757.5889076956 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 16 | 64 | ok | 51.876603 | 0.037558 | 0.03943305 | 0.041699799999999995 | 422690.6691563147 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 1 | ok | 54.095212 | 0.045298500000000005 | 0.0490677 | 0.05427285 | 697132.5196634941 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 2 | ok | 53.738382 | 0.044137499999999996 | 0.045981949999999994 | 0.04927776999999999 | 719801.1909110703 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 4 | ok | 54.280557 | 0.046106999999999995 | 0.04875114999999999 | 0.05593507999999998 | 686499.052202246 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 8 | ok | 54.820401 | 0.042466500000000004 | 0.045849049999999995 | 0.048932119999999996 | 745671.493491453 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 32 | 64 | ok | 52.222513 | 0.039832 | 0.04288 | 0.046460699999999994 | 793614.9707528083 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 1 | ok | 54.413202 | 0.0494425 | 0.05127955 | 0.05620234999999999 | 1285480.2594420535 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 2 | ok | 54.818726 | 0.066751 | 0.0709945 | 0.07818794999999999 | 952170.3980340061 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 4 | ok | 54.290947 | 0.058377 | 0.061373899999999995 | 0.06542497 | 1089611.7100570481 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 8 | ok | 54.777545 | 0.049117499999999994 | 0.05297729999999999 | 0.06008801999999998 | 1283619.80785816 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 64 | 64 | ok | 52.716295 | 0.043692999999999996 | 0.04700769999999999 | 0.05113254999999999 | 1447602.2278598289 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 1 | ok | 54.515664 | 0.057874499999999995 | 0.061895549999999994 | 0.06537269999999999 | 2189041.5213291314 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 2 | ok | 54.91839 | 0.080318 | 0.09140095 | 0.09602208 | 1527427.344577108 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 4 | ok | 55.630287 | 0.08587600000000001 | 0.0894826 | 0.09310336 | 1490859.7485338908 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 8 | ok | 55.88336 | 0.07165250000000001 | 0.07621019999999999 | 0.07803323 | 1784913.9629635932 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `fp32` | 128 | 64 | ok | 67.09672 | 0.096414 | 0.102282 | 0.10678764 | 1314272.008306199 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 1 | ok | 53.514271 | 0.044997999999999996 | 0.0503344 | 0.053264309999999995 | 21832.60245057863 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 2 | ok | 53.146093 | 0.043540499999999996 | 0.04826289999999999 | 0.05868916999999999 | 22494.364037090505 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 4 | ok | 53.498117 | 0.0447485 | 0.050335849999999994 | 0.051399749999999994 | 21974.99860458759 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 8 | ok | 53.100472 | 0.0446655 | 0.0472201 | 0.04810535 | 22300.05436753255 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 1 | 64 | ok | 51.637739 | 0.043675 | 0.050902899999999994 | 0.05741610999999999 | 22202.082466527027 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 1 | ok | 53.79673 | 0.0767645 | 0.08369755 | 0.09338389999999998 | 25567.437311839643 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 2 | ok | 53.543004 | 0.0764235 | 0.07929385 | 0.08350160999999999 | 26012.017031628267 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 4 | ok | 53.187357 | 0.07907700000000001 | 0.08595884999999999 | 0.09282429999999998 | 24931.31422929828 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 8 | ok | 54.223376 | 0.0918545 | 0.1018567 | 0.10898242999999998 | 21803.39832126915 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 2 | 64 | ok | 51.698171 | 0.1754395 | 0.20473785 | 0.8874784299999974 | 9755.43990156371 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 1 | ok | 53.799984 | 0.07429050000000001 | 0.07961205 | 0.08395114 | 53241.16223327508 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 2 | ok | 53.311084 | 0.078707 | 0.0965244 | 0.10026069 | 48749.40708533632 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 4 | ok | 53.864252 | 0.08497450000000001 | 0.09217755 | 0.09425708 | 46562.54350687659 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 8 | ok | 53.58429 | 0.0938605 | 0.10168529999999999 | 0.10514248 | 42602.59236774558 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 4 | 64 | ok | 52.140306 | 0.19306 | 0.20449605 | 0.21939259999999997 | 20568.048359595305 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 1 | ok | 53.688615 | 0.0798905 | 0.08600785 | 0.08685291 | 99213.41127208211 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 2 | ok | 54.191881 | 0.0888655 | 0.1051567 | 0.10774953 | 87117.25307104652 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 4 | ok | 53.855064 | 0.086829 | 0.1136287 | 0.11716697999999999 | 86922.81080554846 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 8 | ok | 53.69378 | 0.092486 | 0.10085309999999999 | 0.10601239 | 86690.51378216624 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 8 | 64 | ok | 52.118642 | 0.189717 | 0.20648895 | 0.24330071999999986 | 42638.90930522775 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 1 | ok | 54.185018 | 0.0848865 | 0.08988185 | 0.09223469 | 187047.52035778447 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 2 | ok | 53.581813 | 0.0899165 | 0.11146715 | 0.11458255999999999 | 171394.76993144423 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 4 | ok | 54.227357 | 0.09268 | 0.12030274999999999 | 0.1276956 | 162736.34657308785 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 8 | ok | 54.451063 | 0.10346749999999999 | 0.1645284 | 0.17520254 | 142097.2271857573 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 16 | 64 | ok | 52.582847 | 0.1910505 | 0.20639285000000002 | 0.21075499 | 83263.08010949094 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 1 | ok | 55.745624 | 0.0896575 | 0.0942887 | 0.09583968 | 355060.0495308769 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 2 | ok | 54.042396 | 0.1051 | 0.10942315 | 0.11082493 | 303125.6613112572 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 4 | ok | 53.735041 | 0.09970899999999999 | 0.12811835 | 0.13409671999999997 | 304698.1793141168 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 8 | ok | 53.577262 | 0.104658 | 0.17813304999999996 | 0.19246346 | 278562.05570440245 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 32 | 64 | ok | 53.143788 | 0.1987175 | 0.2120688 | 0.22202054999999998 | 160029.9576080642 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 1 | ok | 54.093954 | 0.09557299999999999 | 0.1040636 | 0.10681733 | 661780.7942114034 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 2 | ok | 54.318172 | 0.124669 | 0.14934029999999998 | 0.15788964 | 493744.1078584003 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 4 | ok | 55.599075 | 0.12603399999999998 | 0.13293485 | 0.13662863 | 507156.05112449976 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 8 | ok | 53.98316 | 0.120741 | 0.17738749999999998 | 0.20404843999999991 | 496404.172277068 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 64 | 64 | ok | 54.395646 | 0.198565 | 0.21086995 | 0.21674249999999998 | 320466.2784351231 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 1 | ok | 55.140896 | 0.12322 | 0.13363645 | 0.14324016999999997 | 1028110.3027415364 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 2 | ok | 55.61471 | 0.172008 | 0.19478179999999998 | 0.19781641 | 738020.5711702643 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 4 | ok | 55.047317 | 0.16094599999999998 | 0.1925555 | 0.19708422 | 765304.4446849307 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 8 | ok | 55.55376 | 0.163625 | 0.1725904 | 0.17704626999999998 | 780310.8075485317 | - |
| `full_mlp_capacity_search_hd512_depth2` | `jit` | `bf16` | 128 | 64 | ok | 62.195446 | 0.226405 | 0.7530644999999997 | 0.9997086999999996 | 401222.5753398825 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 1 | ok | 1389.933921 | 0.0483815 | 0.053496699999999994 | 0.05612839 | 20371.552674927465 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 2 | ok | 1366.827156 | 0.048936999999999994 | 0.05109175 | 0.05500427999999999 | 20311.233086836208 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 4 | ok | 1359.435816 | 0.044474 | 0.046300999999999995 | 0.05091968999999998 | 22381.218397719622 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 8 | ok | 1380.828772 | 0.0434175 | 0.04683345 | 0.048252819999999995 | 22864.773158585493 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 1 | 64 | ok | 1638.952218 | 0.045409000000000005 | 0.0476604 | 0.048403659999999994 | 21922.66123569272 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 1 | ok | 1351.248517 | 0.0459095 | 0.0474102 | 0.052537039999999986 | 43378.15163960738 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 2 | ok | 1369.454702 | 0.050956 | 0.053333200000000004 | 0.05636535999999999 | 39098.32995393435 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 4 | ok | 1393.395075 | 0.0455595 | 0.04717625 | 0.048452999999999996 | 43817.42680408396 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 8 | ok | 1445.011061 | 0.04708 | 0.04871525 | 0.05659357999999997 | 42045.77944465935 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 2 | 64 | ok | 1530.052146 | 0.045788999999999996 | 0.04768095 | 0.04879631 | 43495.7537270424 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 1 | ok | 1337.565421 | 0.0497895 | 0.05174795 | 0.05340398 | 80340.32160230738 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 2 | ok | 1383.01064 | 0.0498025 | 0.05112895 | 0.05212505 | 80154.37733073901 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 4 | ok | 1397.725196 | 0.0448845 | 0.046852000000000005 | 0.049482469999999994 | 88644.15223392128 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 8 | ok | 1444.095413 | 0.0493865 | 0.0511441 | 0.05191630999999999 | 80843.48862289585 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 4 | 64 | ok | 1658.694457 | 0.0447245 | 0.047057749999999995 | 0.05194906999999999 | 88465.19955979717 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 1 | ok | 1350.953684 | 0.046911499999999995 | 0.0485635 | 0.04901151 | 170341.2104787099 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 2 | ok | 1383.101046 | 0.046563 | 0.04841955 | 0.0488412 | 171432.93072309552 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 4 | ok | 1392.057529 | 0.0460145 | 0.0472843 | 0.04833712 | 173594.15855656457 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 8 | ok | 1444.07209 | 0.046926499999999996 | 0.0508459 | 0.054145809999999996 | 168757.21700785862 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 8 | 64 | ok | 1559.829512 | 0.049533 | 0.0516263 | 0.05784554999999999 | 160359.78320960907 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 1 | ok | 1368.929768 | 0.0516215 | 0.0543197 | 0.05983264999999998 | 308455.73563801177 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 2 | ok | 1397.346079 | 0.0477485 | 0.049341750000000004 | 0.05445306999999998 | 333264.3198470983 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 4 | ok | 1374.753061 | 0.04803 | 0.05293549999999999 | 0.057078449999999996 | 329639.8560957208 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 8 | ok | 1393.577511 | 0.047993499999999994 | 0.050407249999999994 | 0.054416459999999986 | 330990.6093826736 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 16 | 64 | ok | 1512.914323 | 0.0475675 | 0.0497458 | 0.050688859999999995 | 334413.90827444516 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 1 | ok | 1356.438249 | 0.053964 | 0.05636875 | 0.06357421999999997 | 587424.9931711844 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 2 | ok | 1374.49447 | 0.0494505 | 0.051209899999999996 | 0.05236238 | 644790.7956113926 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 4 | ok | 1396.835075 | 0.0524135 | 0.0548207 | 0.05760132999999999 | 606845.8277832225 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 8 | ok | 1461.717199 | 0.0538105 | 0.054993349999999996 | 0.05693466999999999 | 593942.3816495562 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 32 | 64 | ok | 1600.092845 | 0.0506065 | 0.0530937 | 0.05590886 | 628379.9970937425 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 1 | ok | 1371.875403 | 0.053095 | 0.055351649999999995 | 0.05717373 | 1197821.3127597542 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 2 | ok | 1364.098189 | 0.070507 | 0.0726847 | 0.07438778 | 905839.4088265558 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 4 | ok | 1372.619457 | 0.068217 | 0.06968305 | 0.07279390999999999 | 936712.2103949296 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 8 | ok | 1420.26024 | 0.0545935 | 0.05613555 | 0.05746913 | 1169490.621050685 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 64 | 64 | ok | 1533.121032 | 0.0528535 | 0.0546841 | 0.05608733 | 1204007.841101065 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 1 | ok | 1354.402113 | 0.059163 | 0.06136 | 0.06370495999999999 | 2156090.14612901 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 2 | ok | 1361.827508 | 0.10257250000000001 | 0.10501045 | 0.10792199 | 1244331.2448659185 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 4 | ok | 1388.51239 | 0.1113875 | 0.11542735 | 0.11898329999999999 | 1144188.078775919 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 8 | ok | 1388.161162 | 0.076962 | 0.07938719999999999 | 0.08136252 | 1662717.712100428 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `fp32` | 128 | 64 | ok | 1505.601692 | 0.10331850000000001 | 0.10668105 | 0.10792225 | 1239396.6229927826 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 1 | ok | 1368.841728 | 0.053664 | 0.05581565 | 0.05925976999999999 | 18611.451700677233 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 2 | ok | 1383.743316 | 0.0542725 | 0.0565337 | 0.06329027999999998 | 18338.683950020484 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 4 | ok | 1401.646003 | 0.05845 | 0.0608102 | 0.06320174999999999 | 16996.414096553908 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 8 | ok | 1484.771709 | 0.0533465 | 0.054654999999999995 | 0.05535284 | 18720.751166396403 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 1 | 64 | ok | 1575.334006 | 0.0535995 | 0.055436200000000005 | 0.06125924999999998 | 18535.104004175588 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 1 | ok | 1345.456055 | 0.09383 | 0.1010305 | 0.10431955999999999 | 21114.824739564472 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 2 | ok | 1369.46561 | 0.0988165 | 0.1043988 | 0.11082654 | 20119.33987641092 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 4 | ok | 1348.566184 | 0.0959285 | 0.1038508 | 0.10689188 | 20665.772665341163 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 8 | ok | 1358.385336 | 0.109403 | 0.1191724 | 0.12164116999999999 | 18229.26730741326 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 2 | 64 | ok | 1579.369421 | 0.21488200000000002 | 0.23093604999999998 | 0.4357186699999992 | 8918.898405452586 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 1 | ok | 1384.533522 | 0.09700500000000001 | 0.10559634999999999 | 0.10876445 | 40756.838182310225 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 2 | ok | 1322.895788 | 0.09720200000000001 | 0.1133346 | 0.11637586 | 39827.15016826971 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 4 | ok | 1326.275267 | 0.1012835 | 0.11356629999999998 | 0.11950391999999999 | 38979.28075820938 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 8 | ok | 1402.397776 | 0.108448 | 0.11833039999999999 | 0.12253557 | 36798.47389369068 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 4 | 64 | ok | 1574.21774 | 0.21300049999999998 | 0.22556305000000001 | 0.23438869999999998 | 18866.06105981796 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 1 | ok | 1311.813996 | 0.0924725 | 0.0964706 | 0.10032026999999999 | 86036.25180487926 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 2 | ok | 1310.210677 | 0.10452049999999999 | 0.122105 | 0.12841160999999998 | 74385.05871956535 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 4 | ok | 1378.543809 | 0.10328799999999999 | 0.12761125 | 0.13000869999999998 | 74195.02570579409 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 8 | ok | 1496.505761 | 0.10924700000000001 | 0.1179321 | 0.12471402 | 73102.47450048622 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 8 | 64 | ok | 1547.461978 | 0.2130755 | 0.2257848 | 0.23618226999999997 | 38170.56133627501 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 1 | ok | 1364.960747 | 0.1002435 | 0.10452315 | 0.11542433999999999 | 158505.45209222243 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 2 | ok | 1382.422714 | 0.11154349999999999 | 0.13003815 | 0.13525821 | 139732.20671912795 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 4 | ok | 1401.879078 | 0.1075125 | 0.1362612 | 0.14624518999999997 | 142584.0689037513 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 8 | ok | 1346.538996 | 0.1169605 | 0.17571165 | 0.19870948 | 128584.43167208141 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 16 | 64 | ok | 1536.942283 | 0.2084915 | 0.219585 | 0.22500929 | 78033.1890759778 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 1 | ok | 1350.235363 | 0.103078 | 0.1080471 | 0.11560472999999999 | 308484.04385563405 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 2 | ok | 1379.199458 | 0.1188525 | 0.12263365 | 0.12700666 | 268287.9756179888 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 4 | ok | 1403.947705 | 0.117472 | 0.14513245 | 0.14775231 | 261006.94849435694 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 8 | ok | 1356.149471 | 0.12060950000000001 | 0.1721079 | 0.19917653999999996 | 247786.91440402367 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 32 | 64 | ok | 1553.748265 | 0.21388400000000002 | 0.2325449 | 0.23906563 | 150545.86991519562 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 1 | ok | 1371.377338 | 0.11660100000000001 | 0.1259817 | 0.12783415 | 543610.2857861199 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 2 | ok | 1360.659489 | 0.1461285 | 0.1668426 | 0.16802101 | 427409.50438528834 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 4 | ok | 1341.791993 | 0.133438 | 0.1405398 | 0.14535610999999998 | 488393.4069942515 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 8 | ok | 1341.992536 | 0.1383315 | 0.19587539999999995 | 0.21125215999999997 | 434984.26172568044 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 64 | 64 | ok | 1581.210782 | 0.22346349999999998 | 0.23821394999999998 | 0.24560986999999998 | 289182.5919670679 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 1 | ok | 1349.386416 | 0.1411695 | 0.14838615 | 0.15046642999999998 | 903054.5538078 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 2 | ok | 1364.098444 | 0.185582 | 0.20783315 | 0.20936046 | 698068.9340890947 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 4 | ok | 1365.919157 | 0.15632800000000002 | 0.18479484999999998 | 0.19486030999999998 | 796783.2862753955 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 8 | ok | 1345.433652 | 0.168177 | 0.1800529 | 0.18694125 | 760833.8882071105 | - |
| `full_mlp_capacity_search_hd512_depth2` | `compile` | `bf16` | 128 | 64 | ok | 1636.353851 | 0.23444399999999999 | 0.69545825 | 0.8536803099999998 | 407951.6403926687 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.08609 | 0.09284479999999999 | 0.09794396999999999 | 11502.836369391965 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.108386 | 0.11489065 | 0.12018531999999998 | 9178.660422240408 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0820755 | 0.09903994999999999 | 0.09994866 | 11646.408515481222 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.08081250000000001 | 0.1006572 | 0.10455278 | 11918.9113552422 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0855105 | 0.1018638 | 0.10433211999999999 | 10982.161016930539 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.09582750000000001 | 0.1020498 | 0.10705964999999999 | 20701.524998540543 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.094994 | 0.10133465 | 0.10797108999999998 | 20852.261104506735 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.10119349999999999 | 0.10763745 | 0.11857248 | 19557.266512151225 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.1001595 | 0.1071005 | 0.11432565999999998 | 19786.40574992951 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.10117000000000001 | 0.12363465 | 0.12566351 | 18760.32521398965 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.10921249999999999 | 0.11541494999999999 | 0.12673806999999995 | 36290.43525659334 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.108252 | 0.11505485 | 0.12329185999999999 | 36597.17960175681 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.10241600000000001 | 0.10908694999999999 | 0.11278115 | 38688.101938506035 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1011405 | 0.1061582 | 0.12160438 | 39144.113948515704 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.0964605 | 0.10182465 | 0.10223894 | 41263.94773013214 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.1093775 | 0.11690225 | 0.11899420999999999 | 72455.64310813697 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1250755 | 0.1314036 | 0.14008388 | 63700.36368130135 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.11525099999999999 | 0.1204064 | 0.12463007999999999 | 69105.79008045123 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.1109685 | 0.11623254999999999 | 0.12118415999999999 | 71691.73129078811 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.127829 | 0.13235539999999998 | 0.13374146 | 62441.129722383615 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.19593 | 0.2081855 | 0.21456023 | 80894.07365971665 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.28793749999999996 | 0.295168 | 0.30308142 | 59146.35543854435 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.228551 | 0.23959155 | 0.25046474 | 73496.25063470904 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1641145 | 0.17245055 | 0.17601665 | 96947.21696653854 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.143104 | 0.16380495 | 0.18503646999999993 | 107977.19305728245 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.2188725 | 0.22926939999999998 | 0.23643689999999998 | 145110.80434283987 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.288316 | 0.34625975 | 0.35384171999999997 | 104683.8635647355 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.2554765 | 0.26329525000000004 | 0.27041595 | 132561.21946379816 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.19227450000000001 | 0.20026449999999998 | 0.20910378 | 165740.6082825347 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.1838695 | 0.1902512 | 0.19326723999999998 | 173671.97922014768 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.265244 | 0.28073845 | 0.28629263 | 239436.7967666455 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.2898535 | 0.317198 | 0.32258289 | 223681.77688329926 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.27262450000000005 | 0.32627665 | 0.33783184 | 219789.397799029 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.2365825 | 0.2430436 | 0.24717418 | 283874.62606171326 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.188362 | 0.19311625 | 0.19446077 | 340641.5493910553 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.3561195 | 0.36728845 | 0.37186792999999996 | 357660.70673532115 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.46797 | 0.5960113 | 0.61142577 | 295816.15654480073 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.360267 | 0.4343825 | 0.44352463 | 335001.2031254147 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.279997 | 0.3469634 | 0.35582481 | 418966.59140761057 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.2777645 | 0.28716955 | 0.29344137 | 467789.8202795366 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.208202 | 0.2156286 | 0.21853968 | 4786.874925085408 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.214706 | 0.22510305 | 0.22869506 | 4630.797619695931 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.216252 | 0.22754745 | 0.2300997 | 4598.809938359391 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.2240025 | 0.23476315 | 0.24078424999999998 | 4481.022734289601 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.37662399999999996 | 0.44026034999999997 | 2.452926959999992 | 2219.9402223376687 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.21869349999999999 | 0.2273868 | 0.23587064999999996 | 9107.980758844167 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.23811700000000002 | 0.24457769999999998 | 0.2516428 | 8376.153626698622 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2162265 | 0.22920965 | 0.23567561999999997 | 9241.709424205459 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.22554849999999999 | 0.24412649999999997 | 0.25279929 | 8823.766355843614 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.37381600000000004 | 0.5851081999999997 | 0.71549413 | 4553.44003743292 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.236415 | 0.24357905 | 0.24921641 | 16842.73358576661 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.285629 | 0.29222905 | 0.29494873 | 14379.767207386658 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.2338895 | 0.2448564 | 0.25392403999999996 | 17222.548828724593 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.230972 | 0.2498097 | 0.2561501 | 17181.411739904008 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.38746749999999996 | 0.5404702 | 0.57339469 | 9532.256919107838 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.25358250000000004 | 0.26276279999999996 | 0.26689674 | 31424.657263047473 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.30663149999999995 | 0.33963745 | 0.34640033 | 26058.373884506156 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.27360799999999996 | 0.28604835 | 0.29427222 | 29051.656605351694 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.256196 | 0.26914945 | 0.27348485 | 31525.04617631138 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.36904400000000004 | 0.5316535499999999 | 1.4113460099999968 | 19061.005319354754 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.304847 | 0.31671095 | 0.32244729 | 52269.08254366183 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.35785 | 0.42010274999999997 | 0.42300877 | 45703.84173353757 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.32664649999999995 | 0.33774865 | 0.34015650999999997 | 51359.02397310842 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.30821 | 0.3312597 | 0.33516537999999996 | 53448.4756027084 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.430534 | 0.6566736 | 0.673855 | 35223.09648882965 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.382218 | 0.3912324 | 0.39319182 | 83438.65725506424 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.4652885 | 0.5934673 | 0.60087219 | 71544.275129949 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.37088 | 0.4419189 | 0.45831795 | 85973.72062775001 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.3370115 | 0.38265340000000003 | 0.39016569 | 93008.88810373886 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.4239495 | 0.61540615 | 0.6220252 | 71730.93010807161 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.5604025 | 0.57080775 | 0.59741548 | 113677.65252237211 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.560435 | 0.75770425 | 0.76358934 | 117554.66604225546 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.4705965 | 0.6292014 | 0.64983788 | 137952.6787823607 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.401054 | 0.49610129999999997 | 0.5011769 | 156540.60907994234 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.41721600000000003 | 0.5373128 | 3.5755523399999984 | 115792.43858007206 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.9098029999999999 | 0.9195407 | 0.9209595399999999 | 140468.63981648124 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.5884020000000001 | 0.8822035999999993 | 0.98889502 | 195844.70176126203 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.4798095 | 0.7910286 | 0.81514347 | 237127.04870823 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.453446 | 0.57737105 | 0.6536612799999998 | 263295.7460828484 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.4778435 | 0.5955993999999999 | 0.63631785 | 253495.0330618916 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 1 | ok | 58.821057 | 0.046273999999999996 | 0.0504366 | 0.05522146999999999 | 21343.487141616173 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 2 | ok | 58.998861 | 0.0567255 | 0.06112405 | 0.06706179 | 17534.550955931114 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 4 | ok | 58.90477 | 0.0535835 | 0.05725134999999999 | 0.059984939999999994 | 18520.41857627616 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 8 | ok | 58.647718 | 0.052516 | 0.054602 | 0.05636707 | 18959.73256159638 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 64 | ok | 58.970331 | 0.0434905 | 0.0485288 | 0.052345099999999985 | 22620.964476489804 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 1 | ok | 59.192596 | 0.048962 | 0.052077799999999994 | 0.055072239999999995 | 40594.69605939166 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 2 | ok | 58.812904 | 0.048016500000000004 | 0.052653599999999995 | 0.05543707 | 40488.69859200551 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 4 | ok | 59.157712 | 0.0539785 | 0.0564021 | 0.05894081 | 37504.74903884705 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 8 | ok | 59.424646 | 0.0528855 | 0.05804364999999999 | 0.06175106999999999 | 37304.37121430578 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 64 | ok | 58.26257 | 0.0455435 | 0.0495491 | 0.053173109999999996 | 43433.78844704661 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 1 | ok | 59.162593 | 0.049571000000000004 | 0.055977099999999995 | 0.059241459999999996 | 78889.27062530397 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 2 | ok | 59.760741 | 0.0563975 | 0.0632715 | 0.06659328 | 69425.32837312504 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 4 | ok | 59.111233 | 0.0570615 | 0.0583563 | 0.06417853999999999 | 69752.71268299625 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 8 | ok | 60.498069 | 0.055760500000000005 | 0.058398 | 0.06145936999999999 | 71330.46656188222 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 64 | ok | 64.384908 | 0.074514 | 0.07921755 | 0.08396984999999998 | 53230.322014171514 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 1 | ok | 59.551546 | 0.056225 | 0.06340794999999999 | 0.06715797999999999 | 139707.19467603823 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 2 | ok | 59.555171 | 0.0593145 | 0.060909 | 0.06426545 | 134817.72138502292 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 4 | ok | 59.388101 | 0.057638499999999995 | 0.059649549999999996 | 0.062317399999999995 | 137989.4182814591 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 8 | ok | 59.831527 | 0.061913499999999996 | 0.06720775 | 0.06986908 | 128034.70252577259 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 64 | ok | 64.365009 | 0.07802200000000001 | 0.082166 | 0.08556990999999999 | 101607.66182734775 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 1 | ok | 59.69833 | 0.0633405 | 0.06717649999999999 | 0.07422003999999999 | 250079.0093370124 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 2 | ok | 60.495979 | 0.08778949999999999 | 0.09113989999999998 | 0.09407023 | 181693.14390006152 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 4 | ok | 60.496039 | 0.084534 | 0.08762745 | 0.09267185999999998 | 188399.26797464426 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 8 | ok | 59.460277 | 0.072255 | 0.07717874999999999 | 0.08460936 | 219526.55804858453 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 64 | ok | 65.113195 | 0.0862935 | 0.09029459999999999 | 0.09815125999999999 | 184822.68458183753 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 1 | ok | 60.491555 | 0.085863 | 0.0926588 | 0.09365425999999999 | 369239.0629081822 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 2 | ok | 60.787854 | 0.126095 | 0.16238055 | 0.1670073 | 232171.51322282062 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 4 | ok | 60.272876 | 0.130917 | 0.1367573 | 0.1398486 | 243182.78849496145 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 8 | ok | 60.631021 | 0.12826500000000002 | 0.1346846 | 0.1381566 | 249819.3103769164 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 64 | ok | 66.248519 | 7.334543 | 16.99893225 | 31.60054750999994 | 3420.5238253465827 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 1 | ok | 60.479297 | 0.128172 | 0.13504624999999998 | 0.13710318 | 493914.8913641914 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 2 | ok | 61.251551 | 0.1959695 | 0.23705775 | 0.23867702999999998 | 307173.2439817562 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 4 | ok | 60.731272 | 0.1772955 | 0.21196199999999998 | 0.2178269 | 347379.0629645178 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 8 | ok | 60.794893 | 0.152867 | 0.1571189 | 0.18121310999999993 | 416824.98417693283 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 64 | ok | 65.890385 | 4.5601255 | 9.55953415 | 14.079766539999987 | 13107.560196593085 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 1 | ok | 61.417926 | 0.2161555 | 0.22831705 | 0.27323463999999986 | 583966.5416369969 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 2 | ok | 63.12721 | 0.2726065 | 0.35867745 | 0.363006 | 415180.4234645703 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 4 | ok | 62.667125 | 0.259815 | 0.332056 | 0.3355845 | 481147.76596702024 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 8 | ok | 62.782047 | 0.2719485 | 0.3190654 | 0.32366052 | 459256.71740089974 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 64 | ok | 68.426315 | 0.2263165 | 0.23242015 | 0.237681 | 588636.5732963363 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 1 | ok | 58.629737 | 0.107682 | 0.11421185 | 0.12022474999999998 | 9206.68397891522 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 2 | ok | 58.840754 | 0.12277 | 0.13037985 | 0.13503969 | 8073.566336457803 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 4 | ok | 59.11837 | 0.129247 | 0.1385705 | 0.1444049 | 7672.1664847851725 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 8 | ok | 58.229189 | 0.14501799999999998 | 0.15593055 | 0.16367366 | 6913.318189479395 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 64 | ok | 59.34529 | 0.313861 | 0.341761 | 0.8917183099999979 | 3142.754006649942 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 1 | ok | 59.761089 | 0.124749 | 0.13558974999999998 | 0.13885440000000002 | 15860.569102596311 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 2 | ok | 59.084003 | 0.137661 | 0.14355725 | 0.15036918999999999 | 14638.999348271751 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 4 | ok | 59.191657 | 0.1315605 | 0.14404999999999998 | 0.15706257999999998 | 15012.123039960921 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 8 | ok | 59.53802 | 0.1454725 | 0.16375769999999998 | 0.17130292 | 13615.912535734962 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 64 | ok | 59.257047 | 0.2903425 | 0.3488557 | 0.3812491999999999 | 6544.314597244227 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 1 | ok | 59.300818 | 0.1345815 | 0.1406798 | 0.14221278 | 29636.711706071394 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 2 | ok | 59.918431 | 0.17436400000000002 | 0.19198674999999998 | 0.19635562999999998 | 22505.504002322567 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 4 | ok | 59.778635 | 0.1486135 | 0.15872205 | 0.16793075 | 26610.48660734125 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 8 | ok | 59.562078 | 0.158466 | 0.17132514999999998 | 0.17464565999999998 | 25189.012049163914 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 64 | ok | 64.429515 | 0.2922015 | 0.41925585 | 0.42837372 | 12811.581669829526 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 1 | ok | 60.263064 | 0.1625095 | 0.17352145 | 0.17826077999999998 | 48687.26357008367 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 2 | ok | 59.795667 | 0.200735 | 0.2290447 | 0.2345191 | 38108.096855920056 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 4 | ok | 59.993557 | 0.196122 | 0.2085938 | 0.21280517 | 40496.34343329012 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 8 | ok | 59.232924 | 0.17093999999999998 | 0.1880312 | 0.19993815999999995 | 46278.65791429262 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 64 | ok | 60.098585 | 0.30271800000000004 | 0.36605265 | 0.37163839 | 25229.358521483904 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 1 | ok | 60.33411 | 0.212918 | 0.22069095 | 0.22445714 | 74770.21943283611 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 2 | ok | 60.404313 | 0.2578575 | 0.33280794999999996 | 0.33923976 | 56260.15539758173 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 4 | ok | 59.997612 | 0.2256935 | 0.26676995000000003 | 0.27488764 | 67933.28272303772 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 8 | ok | 59.990106 | 0.211959 | 0.2259964 | 0.23109015 | 75020.43603565384 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 64 | ok | 64.779277 | 0.34856149999999997 | 0.3983808 | 0.49570668999999984 | 49164.16615669086 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 1 | ok | 60.270705 | 0.302044 | 0.31023795000000004 | 0.31581052 | 105527.17717241381 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 2 | ok | 61.062446 | 0.3739085 | 0.5033188 | 0.51293533 | 84586.87111313414 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 4 | ok | 60.471302 | 0.3108095 | 0.3462558 | 0.35476747999999997 | 103609.37504249603 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 8 | ok | 60.188674 | 0.258518 | 0.30376064999999997 | 0.31230672 | 119546.90527431981 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 64 | ok | 65.129524 | 0.33286550000000004 | 0.3997245 | 0.5589500399999997 | 95129.97549095094 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 1 | ok | 60.277919 | 0.4827165 | 0.49293175 | 0.50340025 | 132196.18723887898 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 2 | ok | 61.533573 | 0.5101525 | 0.6797133000000001 | 0.68117809 | 135025.64917694905 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 4 | ok | 61.344588 | 0.353648 | 0.4641954 | 0.46680767 | 162397.00414066686 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 8 | ok | 61.217579 | 0.3588255 | 0.37827649999999996 | 0.38552971999999996 | 189923.03962328762 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 64 | ok | 60.326646 | 0.3695045 | 0.4331056 | 0.44919374999999995 | 174262.81926309917 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 1 | ok | 61.42984 | 0.840041 | 0.8526117 | 0.8840405499999998 | 152080.21375824543 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 2 | ok | 63.255137 | 0.5178905 | 0.69776925 | 0.70062858 | 229181.76416458792 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 4 | ok | 62.987333 | 0.440794 | 0.57936625 | 0.6741034299999996 | 264784.1068304577 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 8 | ok | 62.32086 | 0.46667250000000005 | 0.60871115 | 0.6258543099999999 | 287825.80925936525 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 64 | ok | 68.645864 | 0.443962 | 0.5344868 | 0.54357187 | 284636.5045071746 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 1 | ok | 1432.067654 | 0.0642405 | 0.06745075 | 0.07378422999999999 | 15407.902651638413 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 2 | ok | 1362.257816 | 0.081647 | 0.0840726 | 0.08900259999999999 | 12187.38147771513 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 4 | ok | 1358.966588 | 0.0670095 | 0.06876455000000001 | 0.06974709999999999 | 14875.856514628473 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 8 | ok | 1377.252437 | 0.063934 | 0.0664051 | 0.07166196999999999 | 15565.452416100656 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 64 | ok | 1398.40085 | 0.065835 | 0.06835225 | 0.07908657999999998 | 15077.607461123898 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 1 | ok | 1315.637419 | 0.0683035 | 0.07027304999999999 | 0.07103688 | 29164.756375124096 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 2 | ok | 1380.130396 | 0.066993 | 0.07346049999999998 | 0.07889349999999999 | 29366.773082063395 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 4 | ok | 1369.33282 | 0.0751105 | 0.0769122 | 0.0823501 | 26487.905092776535 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 8 | ok | 1375.45675 | 0.06666449999999999 | 0.0698308 | 0.07645246999999998 | 29830.247990783646 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 64 | ok | 1375.7768 | 0.064084 | 0.06732305 | 0.07240422999999999 | 30978.723193336355 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 1 | ok | 1380.339546 | 0.06970000000000001 | 0.0716154 | 0.07750356 | 57014.02174343747 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 2 | ok | 1325.549887 | 0.083837 | 0.08663554999999999 | 0.09162936999999999 | 47432.25488196483 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 4 | ok | 1373.114506 | 0.0756325 | 0.07786889999999999 | 0.08083201999999999 | 52595.786130806766 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 8 | ok | 1391.868401 | 0.073782 | 0.0772575 | 0.08211154999999999 | 53849.04926811065 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 64 | ok | 1428.862635 | 0.097996 | 0.10289605 | 0.10597287 | 40628.61404218347 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 1 | ok | 1303.42697 | 0.083708 | 0.0855062 | 0.09016726999999998 | 95337.65498923398 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 2 | ok | 1320.671461 | 0.096377 | 0.09818265 | 0.10509689999999997 | 82786.85386154105 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 4 | ok | 1416.313796 | 0.0852205 | 0.08749455 | 0.09360759999999997 | 93350.0896744299 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 8 | ok | 1359.731781 | 0.0841715 | 0.0863244 | 0.08915118 | 94731.73157763858 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 64 | ok | 1399.106232 | 0.11591950000000001 | 0.1202958 | 0.12321119999999999 | 68742.0935851425 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 1 | ok | 1369.05459 | 0.14988400000000002 | 0.15854164999999998 | 0.16994891999999998 | 105817.36240700638 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 2 | ok | 1323.309086 | 0.2635425 | 0.27148585 | 0.27558444 | 64487.27617675576 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 4 | ok | 1355.483943 | 0.2066935 | 0.21300525 | 0.21811274 | 79577.16672460879 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 8 | ok | 1359.224709 | 0.133027 | 0.13689825 | 0.14856557999999997 | 119416.88733912306 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 64 | ok | 1433.02031 | 0.1647785 | 0.1701423 | 0.17496764 | 96659.40271255282 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 1 | ok | 1399.898252 | 0.168914 | 0.1762733 | 0.18564962999999998 | 188088.07647379057 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 2 | ok | 1357.954622 | 0.237262 | 0.30156510000000003 | 0.30925417 | 121882.25199399363 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 4 | ok | 1315.250943 | 0.2148725 | 0.23363889999999998 | 0.23652158 | 144965.6159679626 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 8 | ok | 1362.199345 | 0.15931 | 0.16249465 | 0.16647974 | 200721.11569828555 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 64 | ok | 1390.148006 | 0.199141 | 0.207207 | 0.20966215 | 161106.49552189393 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 1 | ok | 1383.768524 | 0.20671699999999998 | 0.2132346 | 0.21910819999999998 | 307964.5898465037 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 2 | ok | 1397.072678 | 0.27313849999999995 | 0.27802055 | 0.28625357 | 252511.88167040714 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 4 | ok | 1303.534405 | 0.2545965 | 0.2805898 | 0.28324058999999996 | 245677.96792996582 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 8 | ok | 1376.684508 | 0.19692900000000002 | 0.20422605 | 0.20805129 | 324165.72405352216 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 64 | ok | 1396.380031 | 0.22265200000000002 | 0.2289712 | 0.23566134 | 291502.11220608617 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 1 | ok | 1361.996097 | 0.27925 | 0.28704155000000003 | 0.29086174000000004 | 457248.41620579833 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 2 | ok | 1366.014043 | 0.401819 | 0.4220595 | 0.42665010999999997 | 339694.9528707753 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 4 | ok | 1333.966554 | 0.325715 | 0.4139846 | 0.41945699 | 386174.6809276953 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 8 | ok | 1408.291352 | 0.2325275 | 0.286973 | 0.29444553 | 503296.7905785986 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 64 | ok | 1409.174935 | 0.29395150000000003 | 0.30384675 | 0.30967743 | 455937.163879917 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 1 | ok | 1375.719137 | 0.16568349999999998 | 0.1716304 | 0.17679734 | 6005.007696017862 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 2 | ok | 1372.504345 | 0.18381799999999998 | 0.19104754999999998 | 0.1942092 | 5441.309222529414 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 4 | ok | 1361.471111 | 0.1688605 | 0.180255 | 0.18177843000000002 | 5894.136359901625 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 8 | ok | 1318.345816 | 0.1904505 | 0.2064347 | 0.21496544999999997 | 5220.778363757941 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 64 | ok | 1396.173016 | 0.34598 | 0.4259968 | 0.42829367 | 2691.949890752597 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 1 | ok | 1316.246081 | 0.1825115 | 0.19120565 | 0.19323476 | 10895.278073645106 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 2 | ok | 1395.698508 | 0.2104855 | 0.21717229999999998 | 0.22136506 | 9643.939950657747 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 4 | ok | 1350.518798 | 0.18462699999999999 | 0.19474814999999998 | 0.21215570999999994 | 10771.76113878197 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 8 | ok | 1350.761839 | 0.2018735 | 0.2145162 | 0.22154656 | 9903.541486628534 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 64 | ok | 1434.437645 | 0.351615 | 0.4147223 | 0.42031237 | 5409.893277412349 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 1 | ok | 1365.084595 | 0.194694 | 0.2015217 | 0.20458399 | 20460.6824967557 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 2 | ok | 1387.362074 | 0.26162799999999997 | 0.2688215 | 0.27277105 | 15468.251028058636 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 4 | ok | 1328.5596 | 0.206752 | 0.21917315 | 0.22174705 | 19369.011827880764 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 8 | ok | 1315.602657 | 0.20829599999999998 | 0.22019555 | 0.22492824 | 19244.74757915509 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 64 | ok | 1415.332102 | 0.35471600000000003 | 0.41476575 | 0.43308108999999995 | 11035.641424862428 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 1 | ok | 1307.312471 | 0.225537 | 0.23348639999999998 | 0.23738162999999998 | 35314.76535942883 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 2 | ok | 1389.176829 | 0.305982 | 0.31752615 | 0.32167318 | 27125.349086289396 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 4 | ok | 1381.510066 | 0.2485575 | 0.26042485 | 0.26486983000000003 | 32485.882244686756 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 8 | ok | 1377.511307 | 0.2260675 | 0.2412882 | 0.24284414 | 35271.67345132002 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 64 | ok | 1390.706584 | 0.3635425 | 0.42478615 | 0.43551234999999994 | 21564.730365932985 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 1 | ok | 1360.364923 | 0.2760185 | 0.28401879999999996 | 0.2861916 | 57809.821440636646 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 2 | ok | 1368.54806 | 0.3223405 | 0.40460425 | 0.40984733 | 45293.89880425806 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 4 | ok | 1298.913341 | 0.2961905 | 0.30549565 | 0.30843971 | 55629.624082119895 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 8 | ok | 1369.602721 | 0.26649100000000003 | 0.29131619999999997 | 0.29718994 | 60272.81131905315 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 64 | ok | 1410.733567 | 0.3822755 | 0.44834135 | 0.45616041 | 39797.09055410086 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 1 | ok | 1299.583457 | 0.36443800000000004 | 0.37578629999999996 | 0.37671731 | 87402.3401758076 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 2 | ok | 1423.126721 | 0.37706649999999997 | 0.47852025 | 0.48330955999999997 | 77438.11484653217 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 4 | ok | 1363.143973 | 0.3351545 | 0.40470045 | 0.41080785 | 94548.49341998623 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 8 | ok | 1363.808819 | 0.311878 | 0.3341254 | 0.34092268 | 104744.91862760065 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 64 | ok | 1448.411309 | 0.3867125 | 0.4577089 | 0.47734447999999996 | 78964.36653995518 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 1 | ok | 1388.328891 | 0.5487139999999999 | 0.5553939999999999 | 0.5591721000000001 | 116530.61838356446 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 2 | ok | 1363.390764 | 0.44541 | 0.7494133 | 0.75674487 | 123342.9786257796 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 4 | ok | 1388.015311 | 0.45536350000000003 | 0.6015457 | 0.6117754999999999 | 139998.57201456546 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 8 | ok | 1323.459127 | 0.3657765 | 0.4467296 | 0.45048042 | 175368.26650934672 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 64 | ok | 1390.409096 | 0.45927799999999996 | 0.5233557999999999 | 0.5281157599999999 | 141840.26215626454 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 1 | ok | 1335.054315 | 0.9425005 | 0.9561786499999999 | 0.97393345 | 135630.91538873027 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 2 | ok | 1364.630165 | 0.634029 | 0.87351005 | 0.88295174 | 195745.6517703206 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 4 | ok | 1379.119865 | 0.552175 | 0.77182195 | 0.7944271999999999 | 236867.08052033625 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 8 | ok | 1373.841087 | 0.4505 | 0.5399023 | 0.6074032899999997 | 265219.45252074517 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 64 | ok | 1401.280392 | 0.5021845 | 0.5869272999999999 | 0.6130633199999999 | 252911.4654172419 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.150964 | 0.15608385 | 0.16108426999999997 | 6600.792517552827 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.16507349999999998 | 0.17825545 | 0.18261619999999998 | 5946.030026024583 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.12373799999999999 | 0.13040415 | 0.14057404999999998 | 8004.651663174504 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.1058705 | 0.1127136 | 0.12001995999999998 | 9345.99703470206 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.1467805 | 0.1572896 | 0.16574717 | 6919.557104060175 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.17659249999999999 | 0.18831534999999996 | 0.1929077 | 11250.523852516882 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.1607835 | 0.1684833 | 0.17665011 | 12350.18911844597 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1349935 | 0.14327185 | 0.15047548 | 14712.931578100213 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.125853 | 0.1344465 | 0.14777204 | 15690.564494731656 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.1533355 | 0.15890564999999998 | 0.15950525999999998 | 12999.664868639686 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.2065165 | 0.2196864 | 0.22181749 | 19230.07214065413 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.21168550000000003 | 0.2192721 | 0.23176929999999998 | 19126.47294163528 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.1642225 | 0.16996124999999998 | 0.18477281999999998 | 24457.274730768204 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.138025 | 0.14616545 | 0.15755757999999997 | 28719.630068189017 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.1519825 | 0.1573468 | 0.1614791 | 26225.143644946173 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.2454305 | 0.2587835 | 0.26727167999999996 | 32356.73428981458 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.233122 | 0.27312549999999997 | 0.28751551999999997 | 32586.83148217048 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.195969 | 0.2069973 | 0.21451045 | 40524.29112123926 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.16005350000000002 | 0.1646589 | 0.17189105999999998 | 49810.92395901707 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.1449745 | 0.15363285 | 0.15827135999999997 | 54986.77842912671 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.353787 | 0.36348319999999995 | 0.37034118 | 45114.51021375593 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.510716 | 0.72238145 | 0.73670791 | 31607.44813711627 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.505377 | 0.56143965 | 0.57334212 | 32816.66637063299 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.29782050000000004 | 0.36107985 | 0.36752362 | 52139.994843354514 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.201104 | 0.205501 | 0.20920821 | 79973.68066169423 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.431448 | 0.44638325 | 0.4525821 | 73703.47865217895 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.5543800000000001 | 0.7676935 | 0.773799 | 59212.90213609804 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.3887185 | 0.5259047 | 0.54257463 | 72981.95653404492 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.34301349999999997 | 0.41965145 | 0.43393820999999994 | 94137.58207620436 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.2203455 | 0.2270805 | 0.23227186 | 144496.16805193337 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.5658719999999999 | 0.57779755 | 0.58744601 | 115349.60428417083 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.514166 | 0.6784053 | 0.69090406 | 128779.83919422455 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.434627 | 0.68578775 | 0.69129372 | 129413.28102987737 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.37861500000000003 | 0.48079875 | 0.48365985 | 158596.30376445674 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.3109875 | 0.35176395 | 0.3583704 | 205393.79510479834 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.8678284999999999 | 0.8807675 | 0.88331035 | 147610.79701019864 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.5870095 | 0.8574707 | 0.87012202 | 198829.26223687702 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.4973265 | 0.6584458 | 0.7140968599999998 | 239256.63859600617 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.4348385 | 0.6828368 | 0.6855078 | 264719.9164014504 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.414793 | 0.4416214 | 0.48254044999999995 | 313901.67520495574 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.354533 | 0.36268649999999997 | 0.36812844 | 2813.03321044504 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.362284 | 0.4128154 | 0.4912203599999999 | 2651.9551751685344 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.3142815 | 0.3432764 | 0.34801248 | 3133.070397710703 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.3244995 | 0.3531093 | 0.37245799999999996 | 3092.61988324494 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.4729395 | 0.5702412 | 0.5809344 | 2089.6410890975317 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.4109645 | 0.4178174 | 0.42246558999999995 | 4856.926363119454 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.40704700000000005 | 0.4641069 | 0.47651177999999994 | 4848.049993091529 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.3495605 | 0.38068009999999997 | 0.38543533 | 5618.242027349377 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.324861 | 0.35282935 | 0.36888266999999997 | 6170.811899275318 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.4771745 | 0.56591535 | 0.6461158399999997 | 4029.6852404799292 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.46337249999999996 | 0.48700285 | 0.52978481 | 8551.084198378827 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.4610915 | 0.6034680999999997 | 0.66219188 | 8134.691620755145 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.3737025 | 0.4278207 | 0.4582704999999999 | 10582.349889102265 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.359893 | 0.41850865 | 0.42590331 | 11025.131622155728 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.4790035 | 0.58543735 | 0.62452438 | 8045.72805632267 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.5492405 | 0.5586998 | 0.56257894 | 14529.78351349054 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.507361 | 0.631269 | 0.76550901 | 14810.458894293346 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.49280349999999995 | 0.61786445 | 0.62421324 | 16616.618795572635 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.3908405 | 0.4531245 | 0.46559193 | 20429.594535369473 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.4876115 | 0.5848258999999999 | 0.59117376 | 15865.141222761336 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.716202 | 0.7281037499999999 | 0.73281793 | 22297.60174242379 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.59618 | 0.7596452 | 0.76373904 | 26295.931720246655 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.5165625 | 0.63974875 | 0.64356989 | 31344.71491276531 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.45249300000000003 | 0.6326185 | 0.6406380199999999 | 33088.86316137864 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.5179325 | 0.6234147 | 0.6261041199999999 | 30922.5961421201 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.0379605 | 1.0491621 | 1.05245626 | 30765.363799657793 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.7139105 | 1.03047645 | 1.0497748999999998 | 42014.42575812012 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.5652645000000001 | 0.7127766 | 0.8500251999999998 | 53460.77133802321 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.5573680000000001 | 0.6962176 | 0.7063938799999999 | 56764.90626038509 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.5550385 | 0.66899045 | 0.67698932 | 58650.970493943074 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.7073494999999999 | 1.71736475 | 1.72590993 | 37462.40486700325 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.042182 | 1.0527412999999999 | 1.2207885399999998 | 60957.24592115066 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.7097575 | 0.8563713 | 1.0178532499999995 | 85425.15740452851 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.6055395 | 0.78492805 | 0.79048863 | 104044.0119175913 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.5956684999999999 | 0.7545123499999999 | 0.76674658 | 104319.82868597734 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 3.0205535 | 3.03071725 | 3.05084164 | 42370.91569778672 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.7169889999999999 | 1.7285523999999999 | 1.73484292 | 74502.20305342623 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.0372875000000001 | 1.04918665 | 1.2011331199999997 | 122612.88518119445 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.7205285 | 0.87598515 | 1.0361596699999995 | 168138.6593285351 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.6914235 | 0.85246905 | 0.97937272 | 177210.39439751554 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 1 | ok | 64.117655 | 0.10742199999999999 | 0.1135799 | 0.11944918999999998 | 9248.674664920518 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 2 | ok | 63.947385 | 0.0824065 | 0.08689894999999999 | 0.08972959 | 12067.168757424326 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 4 | ok | 64.214948 | 0.073996 | 0.0790804 | 0.08141018 | 13440.574783492502 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 8 | ok | 63.512767 | 0.070066 | 0.07593245 | 0.07818391999999999 | 14091.906285441142 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 64 | ok | 68.828714 | 0.0855205 | 0.0888016 | 0.09135594999999999 | 11639.65761714326 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 1 | ok | 64.880212 | 0.110608 | 0.11623855 | 0.11773265 | 18019.219299304637 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 2 | ok | 65.118606 | 0.09360299999999999 | 0.11593144999999999 | 0.11851355 | 20404.86933639896 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 4 | ok | 64.435903 | 0.0756945 | 0.09214465 | 0.10133413 | 25501.233112127138 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 8 | ok | 64.464822 | 0.06774050000000001 | 0.07065104999999999 | 0.07487244 | 29465.825829860787 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 64 | ok | 64.479801 | 0.073367 | 0.07927155 | 0.08405971 | 27021.454764868762 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 1 | ok | 64.342436 | 0.111096 | 0.11710965 | 0.11811392 | 35943.347533351836 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 2 | ok | 64.837638 | 0.101649 | 0.10697994999999999 | 0.10992640000000001 | 39051.31850919248 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 4 | ok | 65.337276 | 0.090325 | 0.09595155 | 0.09958067 | 45459.90765728957 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 8 | ok | 64.967705 | 0.0817375 | 0.0869219 | 0.09273176999999999 | 49844.69638723148 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 64 | ok | 70.491447 | 0.0990065 | 0.10409154999999999 | 0.10680100999999999 | 40145.737054655816 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 1 | ok | 65.553349 | 0.1323705 | 0.1380813 | 0.14130325999999999 | 59992.87884528107 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 2 | ok | 65.047832 | 0.1284025 | 0.13462605 | 0.13989455 | 64764.752560595916 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 4 | ok | 64.851806 | 0.09243799999999999 | 0.10442699999999999 | 0.10683656999999999 | 85210.70263467233 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 8 | ok | 64.726931 | 0.079741 | 0.08913524999999999 | 0.09876272999999997 | 99091.62705478875 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 64 | ok | 70.390957 | 0.1023785 | 0.1080995 | 0.10938648 | 77881.28633405388 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 1 | ok | 65.797144 | 0.15605249999999998 | 0.16272235 | 0.16594836 | 102710.86058220368 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 2 | ok | 67.964588 | 0.1687685 | 0.17393 | 0.17693723 | 94507.94262470056 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 4 | ok | 66.030363 | 0.1213395 | 0.14850585 | 0.15132769 | 123232.6512303394 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 8 | ok | 65.759644 | 0.111958 | 0.1175248 | 0.12252992999999998 | 145770.65574747275 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 64 | ok | 65.13938 | 0.112866 | 0.11803535 | 0.12391880999999999 | 141096.51036583152 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 1 | ok | 65.452228 | 0.208616 | 0.21775275 | 0.2218079 | 151792.70505131021 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 2 | ok | 69.122925 | 0.251048 | 0.3229705 | 0.32729505000000003 | 116101.37391463356 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 4 | ok | 66.450866 | 0.1973045 | 0.25307114999999997 | 0.25740608 | 146802.16261605857 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 8 | ok | 66.475758 | 0.160701 | 0.19046015 | 0.19488038 | 187366.0698549892 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 64 | ok | 64.949374 | 9.837296 | 18.8056581 | 22.865967189999996 | 2922.9371760452777 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 1 | ok | 75.08889 | 0.3377315 | 0.3453898 | 0.35194809 | 188715.7644539468 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 2 | ok | 68.069705 | 0.38723450000000004 | 0.52981055 | 0.53233538 | 172905.623895903 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 4 | ok | 68.741638 | 0.3600795 | 0.3757494 | 0.38004705 | 186253.0792872374 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 8 | ok | 67.416379 | 0.23743 | 0.29801265 | 0.30052517 | 252533.9214217028 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 64 | ok | 66.605827 | 6.265873 | 12.1606053 | 15.432404429999995 | 9789.898888180458 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 1 | ok | 69.261126 | 0.598881 | 0.6080672 | 0.60910643 | 213331.86134349008 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 2 | ok | 71.922772 | 0.4751575 | 0.62662215 | 0.62976055 | 257624.24875360183 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 4 | ok | 70.168855 | 0.41918849999999996 | 0.4992044499999997 | 0.55043554 | 327395.7722770059 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 8 | ok | 69.149166 | 0.43027499999999996 | 0.4956845999999999 | 0.54832368 | 292183.3423080795 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 64 | ok | 74.185837 | 0.348296 | 0.40800754999999994 | 0.4162972 | 346368.4780439187 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 1 | ok | 64.471876 | 0.1740375 | 0.18110355 | 0.18276552000000001 | 5738.83480993553 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 2 | ok | 63.959336 | 0.195287 | 0.2025765 | 0.20751063 | 5194.621696480332 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 4 | ok | 64.241195 | 0.18643300000000002 | 0.19669525000000002 | 0.20331433999999998 | 5388.822977596292 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 8 | ok | 63.268639 | 0.21126450000000002 | 0.23148505 | 0.24481057999999997 | 4700.181088576981 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 64 | ok | 69.105949 | 0.32745 | 0.42320684999999997 | 0.4792894499999999 | 2751.867568684826 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 1 | ok | 64.161049 | 0.223545 | 0.2355266 | 0.24003417999999999 | 8892.585042458537 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 2 | ok | 65.129081 | 0.235071 | 0.2432443 | 0.25083785000000003 | 8551.695899641401 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 4 | ok | 64.544182 | 0.206528 | 0.22628489999999998 | 0.23325262 | 9525.634816881102 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 8 | ok | 63.927162 | 0.2137985 | 0.23093375 | 0.23524048 | 9277.554941216486 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 64 | ok | 69.557603 | 0.35987 | 0.4030848 | 2.718960049999991 | 4528.664521740194 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 1 | ok | 65.224198 | 0.2771395 | 0.2872726 | 0.29744267 | 14356.815388438714 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 2 | ok | 64.928718 | 0.33370350000000004 | 0.4222532 | 0.4343295999999999 | 11437.165641610694 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 4 | ok | 64.966571 | 0.2631185 | 0.2839694 | 0.28953171 | 15539.621917890814 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 8 | ok | 64.075818 | 0.23479899999999998 | 0.2668757 | 0.27692158 | 16650.034669534693 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 64 | ok | 70.180284 | 0.34538 | 0.46530299999999997 | 0.47131905 | 10558.397201517939 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 1 | ok | 64.322564 | 0.35135150000000004 | 0.3599324 | 0.36341450000000003 | 22705.38654075773 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 2 | ok | 65.243308 | 0.411562 | 0.5114119500000001 | 0.51661571 | 19420.21341260923 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 4 | ok | 64.69634 | 0.35902599999999996 | 0.44315145 | 0.45023863000000003 | 22699.805962058635 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 8 | ok | 65.01637 | 0.284809 | 0.3228933 | 0.33261987 | 27362.691449056314 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 64 | ok | 71.12174 | 0.371008 | 0.5411427 | 0.54195217 | 21166.076177237315 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 1 | ok | 65.347161 | 0.517466 | 0.53137875 | 0.5379494899999999 | 30827.005319816308 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 2 | ok | 67.898075 | 0.4541585 | 0.72664585 | 0.73732255 | 32904.75980512485 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 4 | ok | 66.94132 | 0.38909649999999996 | 0.5051894 | 0.50811776 | 38254.72386066673 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 8 | ok | 66.094759 | 0.360867 | 0.42434655 | 0.4564929799999999 | 43084.93753815036 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 64 | ok | 70.673111 | 0.4028435 | 0.58706545 | 0.59774181 | 38976.11499824778 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 1 | ok | 65.716022 | 0.8557305 | 0.86385605 | 0.86566491 | 37338.8359575884 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 2 | ok | 68.807529 | 0.5523905 | 0.86169755 | 0.88583258 | 51665.08496161769 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 4 | ok | 66.635819 | 0.468852 | 0.7660956 | 0.77133102 | 62263.474195233895 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 8 | ok | 66.079882 | 0.42579849999999997 | 0.54109235 | 0.54282858 | 70613.30613308781 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 64 | ok | 70.568296 | 0.374046 | 0.51098355 | 0.5449847899999999 | 74725.57034291564 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 1 | ok | 66.72905 | 1.5288385 | 1.5403580000000001 | 1.54613385 | 41848.62481430163 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 2 | ok | 68.544638 | 0.908276 | 1.0269646999999995 | 1.11005897 | 69462.10090251456 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 4 | ok | 68.458749 | 0.6232465 | 0.82579285 | 0.8304045 | 97436.39367445369 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 8 | ok | 66.644013 | 0.49589 | 0.8011876499999999 | 0.82035967 | 115269.94672583281 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 64 | ok | 70.747839 | 0.4571005 | 0.6027565999999999 | 0.63128217 | 132714.51871121264 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 1 | ok | 69.040905 | 2.8739575 | 2.88363075 | 2.88749856 | 44533.73954041626 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 2 | ok | 72.476268 | 1.567332 | 1.5783912999999998 | 1.58041768 | 81642.97923191407 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 4 | ok | 70.592087 | 0.951335 | 0.96460855 | 1.1644116 | 133174.53379965507 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 8 | ok | 67.938628 | 0.6282730000000001 | 0.98600445 | 0.99633326 | 185439.4722809858 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 64 | ok | 70.438397 | 0.612444 | 0.7568362499999999 | 0.8961009 | 196078.41935794 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 1 | ok | 1385.96747 | 0.13782499999999998 | 0.14232935 | 0.14947384 | 7223.250893407787 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 2 | ok | 1322.107607 | 0.14203749999999998 | 0.1470497 | 0.15533659 | 6996.861627685518 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 4 | ok | 1364.869205 | 0.104006 | 0.11237405 | 0.11335953 | 9497.82167459893 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 8 | ok | 1349.16103 | 0.093373 | 0.09591944999999999 | 0.09833273 | 10679.999863296001 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 64 | ok | 1386.372253 | 0.1031725 | 0.10635085 | 0.11325448 | 9645.3147690603 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 1 | ok | 1364.888601 | 0.13727 | 0.14404675 | 0.15853340999999996 | 14456.9799311096 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 2 | ok | 1340.859369 | 0.12551 | 0.1294686 | 0.13747231999999998 | 15827.012024730659 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 4 | ok | 1303.851778 | 0.10300899999999999 | 0.1059035 | 0.11078734999999999 | 19353.42903087898 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 8 | ok | 1364.062439 | 0.0921315 | 0.0942976 | 0.09847821999999999 | 21686.6534479285 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 64 | ok | 1399.168707 | 0.1296655 | 0.1367589 | 0.14881485999999994 | 15300.612667132415 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 1 | ok | 1389.494051 | 0.1594235 | 0.16581145 | 0.16949075 | 25022.18216448882 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 2 | ok | 1311.716026 | 0.16814400000000002 | 0.17458484999999999 | 0.18293411999999998 | 23630.134518085262 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 4 | ok | 1331.537625 | 0.1562945 | 0.16017945 | 0.1708228 | 25445.062773605987 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 8 | ok | 1368.844396 | 0.105407 | 0.10800255 | 0.14032166999999987 | 37446.444563066456 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 64 | ok | 1394.269772 | 0.13982050000000001 | 0.14486305 | 0.15417297999999996 | 28529.941960112574 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 1 | ok | 1408.435333 | 0.1993165 | 0.21286675 | 0.22021227 | 39755.72097248457 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 2 | ok | 1321.174326 | 0.2231445 | 0.22859975 | 0.23292179999999998 | 35801.48790983753 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 4 | ok | 1353.092105 | 0.1622445 | 0.16677275 | 0.17511227999999998 | 50006.78842152822 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 8 | ok | 1388.207398 | 0.14728249999999998 | 0.15861704999999998 | 0.16365705 | 53814.23165789001 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 64 | ok | 1399.274014 | 0.15487800000000002 | 0.16108235 | 0.16324085 | 51552.55680060708 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 1 | ok | 1298.82656 | 0.2940625 | 0.30259615 | 0.30352297 | 54219.56293339221 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 2 | ok | 1332.112783 | 0.490251 | 0.6753526 | 0.68065271 | 34180.25452839238 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 4 | ok | 1373.843126 | 0.35878849999999995 | 0.48165465 | 0.5139689299999999 | 39404.68194609479 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 8 | ok | 1392.18309 | 0.288944 | 0.31591085 | 0.32009941999999997 | 55266.16426499489 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 64 | ok | 1405.355162 | 0.248925 | 0.25970195 | 0.2972203999999999 | 63963.499868435065 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 1 | ok | 1330.674264 | 0.3516 | 0.35893025 | 0.36175943 | 90710.41448536466 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 2 | ok | 1376.569884 | 0.536159 | 0.7815795 | 0.78514467 | 62148.533051852275 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 4 | ok | 1362.31901 | 0.437587 | 0.61127945 | 0.6240040699999999 | 72063.4763925456 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 8 | ok | 1367.664785 | 0.37379799999999996 | 0.38531815 | 0.39584566 | 92438.45895151794 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 64 | ok | 1333.031142 | 0.291792 | 0.2997192 | 0.30515493 | 112150.67780014484 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 1 | ok | 1400.28788 | 0.47835300000000003 | 0.48716045 | 0.49025437 | 133484.9555572269 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 2 | ok | 1368.42686 | 0.46246 | 0.6206279499999999 | 0.62684838 | 140635.5407985901 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 4 | ok | 1371.656402 | 0.4366905 | 0.5887568999999999 | 0.6014595699999999 | 145212.61463798519 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 8 | ok | 1305.750742 | 0.3822145 | 0.48522525 | 0.48885987000000003 | 169516.23456978478 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 64 | ok | 1390.753778 | 0.344076 | 0.35396075 | 0.3632688 | 198367.91557659884 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 1 | ok | 1363.243658 | 0.727678 | 0.7366276 | 0.73756897 | 175607.41644996987 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 2 | ok | 1413.640696 | 0.5749005 | 0.7724561 | 0.7745463899999999 | 217751.77821545885 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 4 | ok | 1298.356942 | 0.5372445 | 0.7147542499999999 | 0.7185864399999999 | 246966.87059055606 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 8 | ok | 1358.395117 | 0.411509 | 0.5385299499999999 | 0.56398076 | 286836.43080824055 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 64 | ok | 1403.302449 | 0.4139445 | 0.42698709999999995 | 0.43206764 | 331190.01644254936 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 1 | ok | 1367.072833 | 0.327986 | 0.33319865 | 0.33859343999999997 | 3045.694864623432 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 2 | ok | 1377.678361 | 0.3269975 | 0.3814571 | 0.38468821 | 2892.841345877079 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 4 | ok | 1386.618852 | 0.3996635 | 0.4334109 | 0.44340071999999997 | 2533.953455937489 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 8 | ok | 1362.9802 | 0.2708425 | 0.29414365 | 0.3090971799999999 | 3660.2133201604697 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 64 | ok | 1400.991859 | 0.44662999999999997 | 0.53197575 | 0.5926324399999999 | 2133.122091674676 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 1 | ok | 1387.502453 | 0.393353 | 0.40101415 | 0.40485570000000004 | 5073.663248286053 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 2 | ok | 1302.450447 | 0.3957775 | 0.4407954 | 0.44709454 | 5117.521056424201 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 4 | ok | 1315.682296 | 0.3399635 | 0.35935675 | 0.36480470000000004 | 6043.119105163931 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 8 | ok | 1367.911069 | 0.31897949999999997 | 0.336398 | 0.34228429 | 6427.125141782381 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 64 | ok | 1424.457893 | 0.433708 | 0.6658974499999996 | 0.76576515 | 4107.948337784433 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 1 | ok | 1386.532301 | 0.419883 | 0.4330254 | 0.43538361999999997 | 9492.59470821069 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 2 | ok | 1367.959725 | 0.4403995 | 0.5522743499999999 | 0.63457544 | 8375.92841931573 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 4 | ok | 1377.849662 | 0.35329299999999997 | 0.39726104999999995 | 0.40415491 | 11120.141901906782 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 8 | ok | 1365.159161 | 0.332525 | 0.37061685 | 0.37791412 | 11759.909914386091 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 64 | ok | 1377.593113 | 0.45697299999999996 | 0.61337295 | 0.63977619 | 8164.2383500400865 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 1 | ok | 1321.349164 | 0.514026 | 0.524474 | 0.52512171 | 15520.094467711007 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 2 | ok | 1354.195451 | 0.5990005 | 0.60914185 | 0.6115419400000001 | 13325.89881438479 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 4 | ok | 1368.931053 | 0.42657100000000003 | 0.5021609 | 0.52001038 | 17904.973115235243 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 8 | ok | 1377.272159 | 0.41159650000000003 | 0.4699426 | 0.4801268 | 19027.90794716749 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 64 | ok | 1452.075857 | 0.500065 | 0.59589425 | 0.61282775 | 15815.22253125167 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 1 | ok | 1364.8611 | 0.6870315 | 0.69925315 | 0.70473814 | 23221.10447758335 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 2 | ok | 1390.134196 | 0.6001110000000001 | 0.7779049 | 0.78189312 | 26525.80755500093 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 4 | ok | 1402.537819 | 0.48741049999999997 | 0.63281945 | 0.6359394 | 31206.16508997517 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 8 | ok | 1381.636569 | 0.5097645 | 0.5753324 | 0.6069177999999998 | 30751.098265864846 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 64 | ok | 1415.086019 | 0.5020015 | 0.6058450999999999 | 0.6193565799999999 | 30965.637470805694 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 1 | ok | 1315.130271 | 1.0265415 | 1.0371883499999999 | 1.03937816 | 31118.11450209743 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 2 | ok | 1398.89644 | 0.674196 | 0.9705138 | 0.97552688 | 43467.775449927474 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 4 | ok | 1396.988604 | 0.5418835 | 0.82102365 | 0.82406331 | 53814.83076866985 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 8 | ok | 1372.370144 | 0.48314250000000003 | 0.7042527 | 0.71718026 | 60632.403548056674 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 64 | ok | 1435.24217 | 0.5504255 | 0.6635652000000001 | 0.67189055 | 58293.28341873784 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 1 | ok | 1384.750239 | 1.6965919999999999 | 1.70670235 | 1.71395456 | 37681.345301274014 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 2 | ok | 1366.172537 | 1.033786 | 1.09144695 | 1.09733897 | 61461.907187180645 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 4 | ok | 1366.221126 | 0.6685025 | 0.9940299 | 0.99709182 | 87927.04758806904 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 8 | ok | 1398.002242 | 0.5465465 | 0.84749255 | 0.85685068 | 105920.36238533581 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 64 | ok | 1439.75697 | 0.642602 | 0.8201520499999999 | 0.8702568199999999 | 95563.91700011009 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 1 | ok | 1392.233999 | 3.078607 | 3.0889283499999998 | 3.11837754 | 41559.86862717728 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 2 | ok | 1363.541452 | 1.7072895 | 1.7190347499999998 | 1.7308706699999998 | 74888.14373365676 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 4 | ok | 1376.147174 | 1.0210629999999998 | 1.02985475 | 1.1870390899999996 | 124460.39853892049 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 8 | ok | 1373.925035 | 0.670509 | 1.03135295 | 1.06466614 | 175492.4228267272 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 64 | ok | 1398.805561 | 0.7398675 | 0.9674058999999999 | 0.99210463 | 164912.36415247375 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.2039285 | 0.2077724 | 0.22333193999999998 | 4882.579814555713 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2382865 | 0.2502986 | 0.25558667 | 4300.891764101936 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.165908 | 0.17717829999999998 | 0.18420179999999997 | 5967.930964884336 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.1382795 | 0.15451235 | 0.1575783 | 7042.8278587789055 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.178967 | 0.18407844999999998 | 0.18576152999999998 | 5662.379490408496 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.2310005 | 0.24138969999999998 | 0.25014747 | 8607.29788687394 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.2230385 | 0.22962155 | 0.24483614999999997 | 9182.820778886859 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1766635 | 0.18407305 | 0.19175424 | 11247.67159138469 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.15859250000000003 | 0.16488704999999998 | 0.17345583999999997 | 12573.523679717146 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.180122 | 0.18464155 | 0.18983899999999998 | 11078.276330985662 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.28886599999999996 | 0.298527 | 0.30805516 | 13783.970854620668 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.2754765 | 0.31947575 | 0.3587902099999999 | 13716.691073232314 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.222829 | 0.23414495 | 0.24406185999999996 | 17820.366076652168 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.18153750000000002 | 0.18560015 | 0.19215184999999999 | 22289.280761687136 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.172464 | 0.17828395 | 0.18303595 | 23188.556354684737 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.3605095 | 0.3768856 | 0.38177592 | 22070.917499627692 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.33297200000000005 | 0.40453895 | 0.41765406 | 23149.266761443625 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.27516799999999997 | 0.2863962 | 0.29708115 | 29490.578276325836 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.208127 | 0.2158775 | 0.22246115 | 39251.92888885051 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.192868 | 0.2006283 | 0.20372525 | 41237.113402061856 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.4909165 | 0.50437705 | 0.50556126 | 32475.254566416585 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.5694115 | 0.9048530999999995 | 0.9861658799999999 | 25324.246107481355 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.5052455 | 0.70043565 | 0.7067827799999999 | 32014.17587707837 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.41583899999999996 | 0.54534415 | 0.5538586 | 38612.45375193352 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.25636950000000003 | 0.2807689 | 0.28478651 | 60714.87822644555 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.617307 | 0.631795 | 0.63355799 | 51668.989080082094 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.662298 | 0.9135824499999999 | 0.9200279100000001 | 46041.05221645222 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.47093850000000004 | 0.6872946999999996 | 0.79484768 | 61336.76972585723 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.4087375 | 0.52699175 | 0.53453588 | 72862.01114542861 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.31254 | 0.3402513999999999 | 0.35308133999999997 | 103618.25222560673 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.8734785 | 0.8861769 | 0.89176859 | 73420.27502271383 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.615246 | 0.7972581 | 0.80571571 | 100756.68268697921 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.575338 | 0.7642491 | 0.7682236800000001 | 107331.74784217068 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.485753 | 0.6891082 | 0.69831849 | 127887.07571214618 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.40497249999999996 | 0.48416625 | 0.48566548 | 156298.49221775154 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.332578 | 1.3851201499999999 | 1.38797607 | 97738.84093127339 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.839368 | 0.96128255 | 0.96460856 | 151138.43610070014 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.6357685 | 0.8403888999999999 | 0.86415996 | 189328.6727713922 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.5661975 | 0.70398445 | 0.7107309599999999 | 218724.7900455614 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.5116674999999999 | 0.70318015 | 0.70728077 | 256275.9784757009 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.502177 | 0.5124235 | 0.51561662 | 1989.2728066049267 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.4603165 | 0.5260040500000001 | 0.52873989 | 2116.5969777111754 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.4027095 | 0.4539938 | 0.46490003999999996 | 2510.671609676932 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.3830795 | 0.44014555 | 0.48777180999999986 | 2541.946954447599 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.5765745 | 0.68852495 | 0.80417815 | 1673.7519936059994 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.585708 | 0.5958112 | 0.59759224 | 3406.4994920227955 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.5066085 | 0.6064429999999998 | 0.6700332 | 3772.507107497703 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.460152 | 0.53159115 | 0.6098631699999997 | 4370.737110827383 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.405979 | 0.47887945 | 0.49457951999999994 | 4762.407082461317 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.5721805 | 0.7146377999999999 | 0.7703007699999999 | 3385.0846198375643 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.6724905 | 0.68788225 | 0.7450547399999998 | 5910.241982446462 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.5952335 | 0.8089111 | 0.81926972 | 6195.710474201713 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.5028535000000001 | 0.56840925 | 0.6206928599999998 | 7820.6031835720405 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.457416 | 0.5417807 | 0.55610533 | 8698.6199161079 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.5502135 | 0.73775565 | 0.7427957399999999 | 6838.110169271605 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.8243324999999999 | 0.8380536 | 0.84061018 | 9684.09091250235 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.6723595 | 0.8507739 | 0.85572739 | 11451.254728294647 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.60941 | 0.765643 | 0.8422130699999998 | 12784.08555212324 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.5041285 | 0.642293 | 0.66638027 | 15202.811395100049 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.5761324999999999 | 0.7743035 | 0.7807488899999999 | 12978.994665243823 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.0900135 | 1.1016709 | 1.10439789 | 14663.893748798593 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.9703105 | 1.3456474999999999 | 1.36232852 | 15869.749034796907 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.612921 | 0.8522389 | 0.85675927 | 23928.545534466797 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.5606655 | 0.7989773499999999 | 0.8076985 | 26217.686169249653 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.625729 | 0.8272233 | 0.8321972599999999 | 24300.79327509567 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.6628785000000001 | 1.6731526 | 1.67513881 | 19238.43431320972 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.080283 | 1.0913286 | 1.25517975 | 29417.641745045687 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.7551375 | 1.0538521999999997 | 1.0985702899999998 | 40096.18372793116 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.650081 | 0.9083460499999998 | 0.95573318 | 46501.47328292728 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.6630075 | 0.83603355 | 0.89030764 | 46260.58622918339 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 2.825406 | 2.8348934 | 2.91502736 | 22632.686422785795 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.6871935 | 1.69902825 | 1.7474099799999998 | 37867.40255652302 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.0757375 | 1.2170871 | 1.22174205 | 58828.796968199116 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.786597 | 1.0398612499999997 | 1.1109505400000002 | 77196.87344942843 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.723896 | 0.95288685 | 0.9606885199999999 | 83668.81477633104 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 5.128204 | 5.13700515 | 5.14321848 | 24956.934372539563 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.834862 | 2.8425982000000003 | 2.8926522699999997 | 45136.50651296163 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.6580590000000002 | 1.67068645 | 1.71163702 | 77101.01810328291 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.0953455 | 1.14548345 | 1.2353117299999996 | 115792.75073332545 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.9052295 | 1.0382463 | 1.1096133399999997 | 137200.8333149863 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 1 | ok | 69.276787 | 0.14760600000000001 | 0.15484635 | 0.17524777999999994 | 6699.786839581912 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 2 | ok | 69.205465 | 0.125407 | 0.13299009999999997 | 0.15623884999999998 | 7884.617143491516 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 4 | ok | 69.002041 | 0.0873045 | 0.0898942 | 0.09449811 | 11402.063910392548 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 8 | ok | 68.706161 | 0.08868000000000001 | 0.10589825 | 0.11136982 | 11010.116535477458 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 64 | ok | 75.780862 | 0.11057449999999999 | 0.11550745 | 0.11766030999999999 | 9020.547182783888 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 1 | ok | 69.202433 | 0.16001749999999998 | 0.16474704999999998 | 0.16850947 | 12455.172278074675 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 2 | ok | 69.148381 | 0.1506595 | 0.1596465 | 0.17818425999999998 | 13127.742385384307 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 4 | ok | 69.29352 | 0.1099855 | 0.13523225 | 0.13697593 | 17408.128168388128 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 8 | ok | 69.548939 | 0.087231 | 0.1104497 | 0.11126502 | 21687.951583816885 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 64 | ok | 76.274639 | 0.1162945 | 0.1246408 | 0.13241089999999997 | 17003.766334243035 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 1 | ok | 69.545822 | 0.15845900000000002 | 0.16359100000000001 | 0.16594428 | 25135.13275748744 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 2 | ok | 69.636229 | 0.140449 | 0.1680187 | 0.17194441 | 26975.02719419929 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 4 | ok | 69.704346 | 0.109294 | 0.11244105 | 0.11423026 | 36519.53493833037 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 8 | ok | 69.502719 | 0.0911705 | 0.10507405 | 0.11026168999999998 | 42752.637089537275 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 64 | ok | 75.834696 | 0.12247050000000001 | 0.128731 | 0.13197852 | 32610.164881885168 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 1 | ok | 70.420601 | 0.167674 | 0.17252995 | 0.17972339999999998 | 47492.553464445235 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 2 | ok | 70.356674 | 0.177473 | 0.18185085 | 0.18531281 | 48933.2969432715 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 4 | ok | 69.811083 | 0.128291 | 0.13302375 | 0.13703253999999998 | 64698.2336896966 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 8 | ok | 69.67694 | 0.102332 | 0.1071308 | 0.11014849999999998 | 77761.90354918936 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 64 | ok | 76.580712 | 0.1261325 | 0.12988525 | 0.13475974999999998 | 63183.8466704956 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 1 | ok | 71.092965 | 0.218628 | 0.22567345 | 0.22800276 | 72924.63330303921 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 2 | ok | 76.118467 | 0.212056 | 0.2760508 | 0.27988463999999996 | 72824.81826111447 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 4 | ok | 72.22659 | 0.1520215 | 0.18169315 | 0.1843002 | 100385.63140303476 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 8 | ok | 71.912905 | 0.140603 | 0.14671025 | 0.1492344 | 113250.20296559815 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 64 | ok | 76.840997 | 0.137915 | 0.1423294 | 0.14445262 | 115731.46481025682 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 1 | ok | 71.986633 | 0.318892 | 0.32437505 | 0.32615759 | 100098.49065865246 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 2 | ok | 76.938739 | 0.306155 | 0.4069089 | 0.41134713 | 94333.7712778665 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 4 | ok | 73.240972 | 0.2936655 | 0.29922435 | 0.3691644699999999 | 114212.2676396029 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 8 | ok | 72.345907 | 0.210431 | 0.2648686 | 0.26847651 | 139159.54938050517 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 64 | ok | 77.661273 | 7.9964355 | 15.832886299999997 | 19.070177549999993 | 3501.5426341517414 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 1 | ok | 73.111197 | 0.5328280000000001 | 0.5413405499999999 | 0.54576913 | 119797.44049773442 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 2 | ok | 75.258242 | 0.4167345 | 0.4964858 | 0.6154592699999997 | 141698.78062885566 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 4 | ok | 74.733475 | 0.35563500000000003 | 0.4528683 | 0.45584388 | 172116.67854182105 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 8 | ok | 72.888945 | 0.36829449999999997 | 0.42459725 | 0.43054782999999996 | 166822.64584052758 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 64 | ok | 71.022956 | 5.987858 | 10.9966987 | 11.610612429999998 | 10677.547514686032 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 1 | ok | 76.052782 | 0.983447 | 0.99389595 | 1.00025332 | 129882.97057290241 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 2 | ok | 81.45036 | 0.6986939999999999 | 0.7775899999999998 | 0.8344322599999999 | 181056.18730805747 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 4 | ok | 76.99526 | 0.466608 | 0.54057155 | 0.5410832799999999 | 263477.1115580616 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 8 | ok | 75.264981 | 0.44167049999999997 | 0.59932975 | 0.6092005899999999 | 287534.1070486785 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 64 | ok | 81.338778 | 0.503136 | 0.54515885 | 0.57713559 | 251361.41663523993 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 1 | ok | 69.223422 | 0.2323955 | 0.2401341 | 0.24164295 | 4288.61154914513 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 2 | ok | 69.357771 | 0.2537085 | 0.27212385 | 0.27891368 | 3941.728015721503 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 4 | ok | 69.021751 | 0.22582000000000002 | 0.2442569 | 0.25185561 | 4405.234334678339 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 8 | ok | 69.917162 | 0.2606085 | 0.27337435 | 0.28590765999999995 | 3897.110970312822 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 64 | ok | 75.220799 | 0.379694 | 0.5213761499999999 | 0.53166267 | 2426.193495530515 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 1 | ok | 68.930926 | 0.311282 | 0.31994639999999996 | 0.32261963 | 6404.577274523175 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 2 | ok | 69.937942 | 0.308197 | 0.3383990999999999 | 0.35713956 | 6597.117218106501 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 4 | ok | 69.460349 | 0.2898165 | 0.3091967 | 0.31320818 | 7031.221930043701 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 8 | ok | 70.125506 | 0.27029899999999996 | 0.30110085 | 0.30396681 | 7282.898436267028 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 64 | ok | 75.464761 | 0.389882 | 0.5088895999999999 | 1.430117509999997 | 4376.772155045578 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 1 | ok | 69.085056 | 0.3878285 | 0.394757 | 0.39774889 | 10290.784114013655 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 2 | ok | 70.42758 | 0.4120205 | 0.5014614 | 0.50709492 | 9181.239972364468 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 4 | ok | 69.642692 | 0.334115 | 0.3662586 | 0.37051204 | 12072.64351053157 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 8 | ok | 69.682776 | 0.3055195 | 0.35050834999999997 | 0.36026308999999995 | 13037.321810774485 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 64 | ok | 69.64138 | 0.419458 | 0.53945695 | 0.55201749 | 9305.4509144676 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 1 | ok | 70.206339 | 0.5363985 | 0.549379 | 0.5551317299999999 | 14872.490885301011 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 2 | ok | 70.354622 | 0.5308555 | 0.7007255 | 0.70556818 | 15406.598230043777 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 4 | ok | 69.766945 | 0.4199 | 0.5982179499999999 | 0.62189314 | 17123.445271435503 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 8 | ok | 69.86781 | 0.350447 | 0.38591285 | 0.39425585999999996 | 22858.83310457356 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 64 | ok | 75.971549 | 0.4316375 | 0.60999765 | 0.61338028 | 17955.969628195613 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 1 | ok | 71.017397 | 0.8316045000000001 | 0.83948895 | 0.83977463 | 19223.31187963669 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 2 | ok | 75.065154 | 0.577557 | 0.8513362 | 0.85903433 | 25595.836683588393 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 4 | ok | 72.825507 | 0.4809295 | 0.7350039 | 0.74519671 | 30095.690376139322 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 8 | ok | 70.642449 | 0.46197350000000004 | 0.568721 | 0.57277849 | 34033.66388812999 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 64 | ok | 69.77773 | 0.594285 | 1.02286615 | 3.176201459999992 | 20749.388547940413 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 1 | ok | 72.072937 | 1.4197245 | 1.4424167 | 1.45314946 | 22521.118122842283 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 2 | ok | 76.616245 | 0.8608089999999999 | 0.98441615 | 0.98838826 | 36292.79530171618 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 4 | ok | 73.664604 | 0.5832174999999999 | 0.7430612 | 0.8519623599999996 | 50562.63413121275 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 8 | ok | 71.558343 | 0.5247269999999999 | 0.67493645 | 0.67623242 | 58992.03408441746 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 64 | ok | 71.141975 | 0.5162385 | 0.68441405 | 0.68781968 | 60491.557420174206 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 1 | ok | 73.927768 | 2.5818250000000003 | 2.59060865 | 2.59183642 | 24789.76900823517 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 2 | ok | 76.058492 | 1.4725234999999999 | 1.48221805 | 1.48384294 | 43430.246253975005 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 4 | ok | 75.047447 | 0.9261955 | 0.94224995 | 1.0710683099999996 | 68614.71710634585 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 8 | ok | 72.475251 | 0.645374 | 0.7770722999999999 | 0.8618000899999997 | 93869.65447518883 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 64 | ok | 78.328197 | 0.5520430000000001 | 0.6956306999999997 | 0.75562485 | 110294.39642425567 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 1 | ok | 75.901023 | 4.8987865 | 4.9195182 | 6.380452719999996 | 25816.764577553102 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 2 | ok | 80.69725 | 2.6243405 | 2.6371045 | 2.6584752899999997 | 48769.58490324087 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 4 | ok | 76.610958 | 1.513997 | 1.52231495 | 1.5246726499999999 | 84578.95893821056 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 8 | ok | 75.141677 | 0.978189 | 0.9958815 | 1.1133573199999995 | 130034.0683163642 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 64 | ok | 81.862532 | 0.7928945000000001 | 0.91748805 | 1.02041623 | 155762.73188750798 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 1 | ok | 1423.256387 | 0.192406 | 0.1964877 | 0.22530650999999993 | 5165.631852856356 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 2 | ok | 1367.274186 | 0.218839 | 0.2250337 | 0.23086045 | 4776.434219903211 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 4 | ok | 1329.386497 | 0.143671 | 0.15635735 | 0.15887642 | 6877.1127349961 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 8 | ok | 1366.711006 | 0.119921 | 0.12876985 | 0.13520931999999997 | 8207.417864265723 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 64 | ok | 1434.807904 | 0.161061 | 0.16825515 | 0.17200704 | 6254.4922890866355 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 1 | ok | 1360.630673 | 0.18367050000000001 | 0.18859864999999998 | 0.19202793999999998 | 10867.162629479526 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 2 | ok | 1330.866184 | 0.180989 | 0.1835528 | 0.18833924999999999 | 11235.915919843424 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 4 | ok | 1394.337723 | 0.1354225 | 0.13850955 | 0.14209381999999998 | 14726.529814006872 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 8 | ok | 1402.010803 | 0.121284 | 0.12789695 | 0.14883087999999994 | 16266.304123280366 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 64 | ok | 1381.265924 | 0.16485 | 0.17230865 | 0.18456920999999998 | 12068.92077900056 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 1 | ok | 1381.47726 | 0.2458595 | 0.25373924999999997 | 0.2588159 | 16196.307015252225 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 2 | ok | 1407.500055 | 0.233072 | 0.27334655 | 0.27915638 | 16213.18916997875 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 4 | ok | 1390.324542 | 0.21416849999999998 | 0.22232095 | 0.22672137 | 18598.192832200686 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 8 | ok | 1298.66477 | 0.143622 | 0.1474096 | 0.15304873 | 27730.639542034034 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 64 | ok | 1382.639544 | 0.1799505 | 0.1880789 | 0.18850202 | 22241.508081007134 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 1 | ok | 1389.187436 | 0.2895575 | 0.29614975 | 0.30115282 | 27582.661097911274 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 2 | ok | 1368.477927 | 0.3129005 | 0.3720242 | 0.37997726 | 24015.78076954367 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 4 | ok | 1321.559936 | 0.233315 | 0.24264689999999997 | 0.25315213999999997 | 34327.41228380703 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 8 | ok | 1333.386444 | 0.174933 | 0.1776997 | 0.18222056 | 45721.31373336816 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 64 | ok | 1404.609993 | 0.2114305 | 0.217837 | 0.22073398 | 37847.72478875529 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 1 | ok | 1368.481702 | 0.4055015 | 0.41278925 | 0.41837332 | 39350.64161467094 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 2 | ok | 1308.871339 | 0.553683 | 0.79344475 | 0.8472088599999998 | 27072.482273530815 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 4 | ok | 1309.328689 | 0.4791485 | 0.65464135 | 0.65944665 | 33345.671231689055 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 8 | ok | 1362.744487 | 0.38463250000000004 | 0.4935447 | 0.4977048 | 41244.93931039183 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 64 | ok | 1444.060326 | 0.312751 | 0.3788692 | 0.38123555 | 47743.93780301734 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 1 | ok | 1327.240577 | 0.513314 | 0.52021095 | 0.5261877500000001 | 62235.41198053463 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 2 | ok | 1366.471315 | 0.5613625 | 0.8975998999999999 | 0.9014944699999999 | 50988.29659254774 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 4 | ok | 1373.901162 | 0.5036579999999999 | 0.7051682 | 0.72000077 | 60980.38314432082 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 8 | ok | 1368.180488 | 0.39129800000000003 | 0.4991156 | 0.50541684 | 76647.7970968018 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 64 | ok | 1340.755921 | 0.3341655 | 0.41183994999999995 | 0.4568618 | 88655.39520441912 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 1 | ok | 1381.549246 | 0.730182 | 0.73845325 | 0.7425733099999999 | 87507.77554441527 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 2 | ok | 1364.45255 | 0.501738 | 0.93220715 | 0.93687868 | 108988.5106696176 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 4 | ok | 1373.607218 | 0.4732995 | 0.8751325 | 0.88028461 | 112532.87014803982 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 8 | ok | 1363.376708 | 0.4660835 | 0.63381985 | 0.64010343 | 136320.20559813408 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 64 | ok | 1399.673579 | 0.40360050000000003 | 0.45505384999999987 | 0.49795847 | 164863.7527307491 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 1 | ok | 1384.560706 | 1.1574550000000001 | 1.16565035 | 1.166196 | 110463.86573338047 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 2 | ok | 1338.959573 | 0.7342865000000001 | 0.9608185999999995 | 1.15739068 | 163837.9783704173 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 4 | ok | 1383.032141 | 0.5619055 | 0.76157165 | 0.7971481499999999 | 217497.48951823526 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 8 | ok | 1372.003261 | 0.483321 | 0.62445745 | 0.84018026 | 243195.23580533054 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 64 | ok | 1327.908241 | 0.48049200000000003 | 0.59630075 | 0.59882293 | 258354.45650134457 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 1 | ok | 1318.707756 | 0.444283 | 0.45295149999999995 | 0.45834077 | 2249.862533399209 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 2 | ok | 1402.834981 | 0.4413325 | 0.5353641 | 0.5462363699999999 | 2233.774509755742 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 4 | ok | 1371.766752 | 0.430388 | 0.47451885 | 0.47923531999999996 | 2298.045360749689 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 8 | ok | 1364.867474 | 0.343511 | 0.3769363 | 0.38867008999999997 | 2879.99670067578 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 64 | ok | 1401.605966 | 0.5506245 | 0.6706452 | 0.67491148 | 1789.0120738993655 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 1 | ok | 1308.289699 | 0.522842 | 0.5327115 | 0.53776001 | 3818.32078423116 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 2 | ok | 1377.893705 | 0.4749755 | 0.56402995 | 0.6090566499999999 | 3987.947624688666 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 4 | ok | 1403.132053 | 0.4152455 | 0.48943615 | 0.49714776 | 4711.317106646363 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 8 | ok | 1354.877869 | 0.3736565 | 0.4258244 | 0.43853281 | 5383.751955042225 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 64 | ok | 1390.166983 | 0.5541739999999999 | 0.6756038 | 0.6894545299999999 | 3556.4044139673233 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 1 | ok | 1361.773266 | 0.607851 | 0.62017005 | 0.62269071 | 6561.365498896707 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 2 | ok | 1384.636257 | 0.553901 | 0.7132317499999998 | 0.80293977 | 6731.672634182261 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 4 | ok | 1386.902835 | 0.47994250000000005 | 0.5694324 | 0.57499985 | 8177.070537495609 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 8 | ok | 1301.878447 | 0.48541049999999997 | 0.56135575 | 0.5716017499999999 | 8382.692021068722 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 64 | ok | 1410.497453 | 0.590927 | 0.73528315 | 0.7523031499999999 | 6669.197627166176 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 1 | ok | 1368.048446 | 0.758121 | 0.76590585 | 0.76995172 | 10542.938952298076 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 2 | ok | 1393.338113 | 0.6482265 | 0.7899514 | 0.9121402799999996 | 11782.280886984245 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 4 | ok | 1379.802651 | 0.5696635000000001 | 0.7251261999999999 | 0.73351429 | 13978.176432263364 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 8 | ok | 1340.080958 | 0.5015615 | 0.6074817499999999 | 0.6197185 | 15648.260895316811 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 64 | ok | 1424.59042 | 0.597527 | 0.736729 | 0.74523823 | 13103.534465489633 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 1 | ok | 1356.179651 | 1.0670735 | 1.0787446 | 1.08221604 | 14970.798335037633 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 2 | ok | 1367.233534 | 0.7706305 | 1.09221345 | 1.09932901 | 19545.19551608783 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 4 | ok | 1379.980845 | 0.6214465 | 0.8508882999999995 | 0.96551409 | 23956.937405014487 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 8 | ok | 1362.66349 | 0.586019 | 0.7485762499999999 | 0.76486107 | 26901.235577112715 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 64 | ok | 1444.600268 | 0.6600435 | 0.7797935500000001 | 0.7821745600000001 | 24236.67500933415 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 1 | ok | 1391.252696 | 1.659438 | 1.6710266500000002 | 1.6731600899999999 | 19269.77825651936 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 2 | ok | 1394.272851 | 1.047446 | 1.1583164999999995 | 1.23939714 | 30194.168887007454 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 4 | ok | 1305.801044 | 0.708009 | 0.9911806 | 0.9978121999999999 | 42015.68461262471 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 8 | ok | 1377.702128 | 0.6349115 | 0.8377011999999999 | 0.9094714699999997 | 47971.41335506758 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 64 | ok | 1411.083704 | 0.6836065 | 0.8244545999999999 | 0.8277490399999999 | 46135.35386191263 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 1 | ok | 1415.345647 | 2.842031 | 2.8500416 | 2.85487824 | 22520.69366720487 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 2 | ok | 1310.603499 | 1.655061 | 1.6816842 | 1.8367202199999997 | 38450.251094558495 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 4 | ok | 1364.853081 | 1.0132984999999999 | 1.1634181499999994 | 1.2603226 | 62162.89016100926 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 8 | ok | 1365.155971 | 0.702833 | 1.0555424 | 1.06740614 | 83385.79037859832 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 64 | ok | 1413.538008 | 0.743352 | 0.91322755 | 1.684947749999997 | 78415.03800470584 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 1 | ok | 1367.221895 | 5.219998 | 5.22989165 | 5.25645528 | 24517.862178185722 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 2 | ok | 1390.435965 | 2.8502465 | 2.8593798 | 2.86198699 | 44913.232862291065 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 4 | ok | 1306.958143 | 1.621403 | 1.63469565 | 1.63761972 | 78924.39661867167 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 8 | ok | 1388.121167 | 1.039188 | 1.1833077 | 1.18397114 | 120632.73679942348 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 64 | ok | 1402.765246 | 1.0348255 | 1.1595932 | 1.16213945 | 121546.65074405543 | - |
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
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0711135 | 0.07536595 | 0.07705854 | 13955.845936392603 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.07262350000000001 | 0.07602915 | 0.08229789999999998 | 13634.848928600843 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.071268 | 0.07530965 | 0.08073567999999999 | 13889.035495374674 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0686415 | 0.07417515 | 0.07836428999999999 | 14333.321388898845 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.073658 | 0.07868035 | 0.0832573 | 13445.819264524442 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.09075949999999999 | 0.09418454999999999 | 0.10104030999999998 | 21902.062736268505 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.08540600000000001 | 0.0917311 | 0.09788397999999998 | 23164.2827145111 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.08744199999999999 | 0.09111855 | 0.09895681999999997 | 22711.875108307257 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0781895 | 0.10840165 | 0.11041255 | 22707.29546800985 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.0783595 | 0.08134075 | 0.08806100999999998 | 25335.158815976556 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.085921 | 0.09100155 | 0.10222281 | 46055.8994268113 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0889415 | 0.09175789999999999 | 0.09986269999999998 | 44728.75032120834 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.090478 | 0.0939568 | 0.10176574999999997 | 43877.85458352457 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0895285 | 0.09500375 | 0.10094897999999998 | 44299.20212707049 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.0802705 | 0.08263915 | 0.08660475999999999 | 49586.658015447734 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.08608650000000001 | 0.0925236 | 0.09931483999999997 | 91658.0202600888 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0936305 | 0.0978367 | 0.10366174999999998 | 84565.09440424315 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.08918899999999999 | 0.09393615 | 0.09930732999999999 | 89216.29285783428 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0861025 | 0.09029645 | 0.09625294999999998 | 92098.98891427495 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.079934 | 0.08266105 | 0.08607725 | 99640.993500418 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0886095 | 0.09586425 | 0.10077818999999998 | 178381.5796045637 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0933095 | 0.09658135 | 0.10478653999999997 | 170423.6561668652 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.109765 | 0.11529869999999999 | 0.12498224999999999 | 144620.4012746119 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0883095 | 0.09291134999999999 | 0.10107307999999998 | 179446.38994238875 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.077706 | 0.0808185 | 0.0857556 | 204618.91389815146 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0913815 | 0.0959559 | 0.10275150999999999 | 346892.4184710673 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.116225 | 0.12198545 | 0.13467572 | 272934.0047282405 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1088935 | 0.11605524999999998 | 0.12747519 | 290777.8179414639 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.106944 | 0.1098026 | 0.12165168999999998 | 297590.8163474075 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.121113 | 0.1247825 | 0.12556936000000002 | 263357.14520854596 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.10228799999999999 | 0.10745885 | 0.11770741999999998 | 620812.3057415244 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.128391 | 0.13565795 | 0.14323224999999998 | 494939.7055170619 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1209395 | 0.12786655 | 0.13996928 | 524492.7458555648 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.119147 | 0.1263531 | 0.1332866 | 533640.3544172414 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.1114645 | 0.13609354999999998 | 0.14010766 | 541415.7683958698 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.122324 | 0.13189225000000002 | 0.13510569 | 1035385.4295061873 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.1744685 | 0.1821623 | 0.18939618 | 741565.7326182786 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.17119299999999998 | 0.17745105 | 0.18315356 | 743856.7601528021 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1457505 | 0.1520415 | 0.16752908 | 869620.0412634709 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.1460995 | 0.15043125 | 0.15193946 | 875254.3708015142 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.13804850000000002 | 0.14436425 | 0.14927801999999998 | 7213.620700898471 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.13385750000000002 | 0.13763295 | 0.14368712 | 7439.2540273145605 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.14806750000000002 | 0.15679785000000002 | 0.16728785 | 6715.919428845975 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.155358 | 0.16872195 | 0.1728282 | 6373.823471859888 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.25342699999999996 | 0.26496770000000003 | 0.27741055 | 3954.53017927388 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.13680350000000002 | 0.14348429999999998 | 0.14701455 | 14540.883949059795 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.146449 | 0.15296944999999998 | 0.16025426999999998 | 13603.937414813843 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1482415 | 0.1586025 | 0.16266883999999998 | 13428.43530260442 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.15517399999999998 | 0.16999445 | 0.17540201 | 12807.106407203228 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.269423 | 0.2819554 | 0.28962431 | 7476.801540938891 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.173017 | 0.18013479999999998 | 0.1831764 | 23015.092146675182 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.19256450000000003 | 0.2003106 | 0.20504546 | 20651.580067879677 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.17768299999999998 | 0.1850692 | 0.18615802 | 22437.409444017823 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.2009475 | 0.21337979999999998 | 0.22386535999999999 | 19757.861480201682 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.3590665 | 0.40919995 | 0.42359133 | 10861.48108785336 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.170233 | 0.17697865 | 0.18051937999999998 | 46691.06274695195 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.19017 | 0.1971465 | 0.20113031 | 41875.520826790285 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1889395 | 0.20539249999999998 | 0.21535228999999997 | 41726.87942298402 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.20202199999999998 | 0.21975655 | 0.2258233 | 39473.486843092105 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.35696950000000005 | 0.40935255 | 0.41784102 | 22081.41882831796 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1777365 | 0.1867303 | 0.18920257000000001 | 89783.59236273829 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.202135 | 0.20865184999999997 | 0.21315367999999998 | 79389.63659592322 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.20623950000000002 | 0.218217 | 0.22257327 | 77022.68799553269 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.210405 | 0.22702165 | 0.23341354999999997 | 75235.28188638174 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.3644145 | 0.4275156 | 0.46735923999999984 | 42988.18158164912 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1960695 | 0.2065097 | 0.21040846 | 162334.2529230565 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.23071049999999999 | 0.23777275 | 0.24182503 | 138969.2787563917 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.226777 | 0.2387823 | 0.24487873999999998 | 141239.6711869835 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.22529949999999999 | 0.23855664999999998 | 0.2426463 | 142197.63783061507 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.36544699999999997 | 0.42190774999999997 | 0.4484808999999999 | 85403.34585755608 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.2271355 | 0.23513794999999998 | 0.23780427999999998 | 280506.60544843756 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.28450149999999996 | 0.2965048 | 0.30335665 | 229080.1501334034 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.263328 | 0.2783368 | 0.28069532999999997 | 245792.45561564324 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.262681 | 0.27744075 | 0.28284138 | 243502.2008413762 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.3728825 | 0.43352809999999997 | 0.43721561 | 171377.3938543317 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.28631300000000004 | 0.29284525 | 0.29816191999999997 | 445642.3868856915 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.33385600000000004 | 0.4039202 | 0.40697265 | 356207.593555559 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.31682699999999997 | 0.3476078 | 0.36342608 | 398191.2907227589 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.30686199999999997 | 0.32847364999999995 | 0.33460101000000003 | 426868.9741024597 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.41363150000000004 | 0.47669005 | 0.48137239 | 312078.08505766233 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 1 | ok | 57.690865 | 0.04503 | 0.04775715 | 0.05102704999999999 | 22207.52577516479 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 2 | ok | 58.507625 | 0.0427155 | 0.04679199999999999 | 0.056780779999999975 | 23214.783173925152 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 4 | ok | 58.276092 | 0.0413325 | 0.044897299999999994 | 0.04798998999999999 | 23882.935403824613 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 8 | ok | 58.309863 | 0.042652499999999996 | 0.0466751 | 0.050979199999999995 | 23129.70863043444 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 64 | ok | 56.895284 | 0.03921 | 0.04309085 | 0.04659862 | 25115.96038911655 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 1 | ok | 57.834719 | 0.046656500000000004 | 0.050161000000000004 | 0.052723719999999995 | 42435.563718271995 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 2 | ok | 57.771822 | 0.0470305 | 0.050428499999999994 | 0.05616234 | 41986.79851081223 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 4 | ok | 58.133985 | 0.0437045 | 0.04867215 | 0.05127991 | 45158.8371780062 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 8 | ok | 59.244076 | 0.041313 | 0.044317249999999996 | 0.047310849999999995 | 47958.59446788021 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 64 | ok | 59.607848 | 0.041291499999999995 | 0.04384389999999999 | 0.04655089999999999 | 47964.184184385835 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 1 | ok | 59.289779 | 0.0449535 | 0.0474349 | 0.05335723999999998 | 87859.64246397096 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 2 | ok | 58.395405 | 0.046248 | 0.0495192 | 0.052852839999999984 | 85664.18440414991 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 4 | ok | 58.80137 | 0.046386 | 0.049955849999999996 | 0.053225829999999995 | 85321.8296754443 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 8 | ok | 58.547069 | 0.0473095 | 0.0511473 | 0.057078099999999986 | 83622.38805470911 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 64 | ok | 59.266787 | 0.041499 | 0.044662549999999995 | 0.04567182 | 95157.21160822824 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 1 | ok | 58.816673 | 0.046338500000000005 | 0.049187749999999995 | 0.05226437999999999 | 171156.01769582066 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 2 | ok | 58.07628 | 0.045817 | 0.0493138 | 0.05236266999999999 | 172697.1056828573 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 4 | ok | 58.793278 | 0.0494995 | 0.05312255 | 0.056868149999999985 | 160298.79695752883 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 8 | ok | 58.314653 | 0.049631999999999996 | 0.05251685 | 0.05591738999999999 | 159836.90242476578 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 64 | ok | 56.595194 | 0.043764 | 0.045870999999999995 | 0.04692241 | 181762.2486171301 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 1 | ok | 58.577807 | 0.046443 | 0.05029575 | 0.05304167 | 340575.82005335967 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 2 | ok | 59.035391 | 0.0502245 | 0.05354514999999999 | 0.05513147 | 315573.63792500866 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 4 | ok | 58.565544 | 0.0520635 | 0.055761349999999994 | 0.057397229999999994 | 304535.21454886533 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 8 | ok | 58.703497 | 0.050589499999999996 | 0.052091349999999995 | 0.05471642 | 315108.3834979315 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 64 | ok | 56.972628 | 0.044495 | 0.0472924 | 0.04991416 | 355905.76683011645 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 1 | ok | 59.399165 | 0.050183000000000005 | 0.0536846 | 0.05715782999999999 | 631276.2670109227 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 2 | ok | 59.033619 | 0.0609885 | 0.0634931 | 0.06580769 | 525646.4547939121 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 4 | ok | 59.237024 | 0.059407 | 0.06435714999999999 | 0.06795547999999998 | 529175.0755893522 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 8 | ok | 59.841837 | 0.0586035 | 0.062132999999999994 | 0.06781815 | 540521.5492428645 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 64 | ok | 64.289542 | 0.077992 | 0.0818208 | 0.08580977 | 409388.92562017305 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 1 | ok | 59.349788 | 0.058054499999999995 | 0.061239049999999996 | 0.06753693999999999 | 1096735.8741276239 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 2 | ok | 59.680268 | 0.078009 | 0.08295355 | 0.08574674 | 814863.5180000805 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 4 | ok | 59.298065 | 0.06678300000000001 | 0.0701512 | 0.07678999999999998 | 953329.74245797 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 8 | ok | 59.12418 | 0.069241 | 0.0725627 | 0.07707489 | 917612.4526891924 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 64 | ok | 64.694584 | 0.0864485 | 0.09157165 | 0.09691034999999998 | 734181.0371133103 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 1 | ok | 59.7428 | 0.07697999999999999 | 0.08186365 | 0.08444726999999999 | 1658864.818069185 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 2 | ok | 60.638448 | 0.13352999999999998 | 0.1383278 | 0.14212022 | 1020924.3225489162 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 4 | ok | 60.541179 | 0.106409 | 0.12491754999999999 | 0.12893503 | 1154993.4427051968 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 8 | ok | 60.571058 | 0.086566 | 0.0952113 | 0.09883063 | 1458022.839016509 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 64 | ok | 67.307618 | 0.1102355 | 0.11573249999999999 | 0.12112955999999998 | 1162445.1743631435 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 1 | ok | 58.787358 | 0.08347750000000001 | 0.090998 | 0.09446257999999999 | 11845.2115140194 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 2 | ok | 58.136505 | 0.085631 | 0.09029179999999999 | 0.09582682999999999 | 11591.369900550684 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 4 | ok | 58.073123 | 0.087948 | 0.09588474999999999 | 0.09953479 | 11263.76262836747 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 8 | ok | 58.740339 | 0.10198499999999999 | 0.11536774999999999 | 0.12084322 | 9768.784591266238 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 64 | ok | 56.305883 | 0.1930715 | 0.21925309999999998 | 0.22257322 | 5139.50624352358 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 1 | ok | 58.337408 | 0.08873600000000001 | 0.09444685 | 0.09690278 | 22389.591884758083 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 2 | ok | 58.542334 | 0.0923825 | 0.0975543 | 0.10041593999999998 | 21495.31283955877 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 4 | ok | 58.551043 | 0.09360450000000001 | 0.10203055 | 0.10684756999999999 | 21173.601938824384 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 8 | ok | 58.58572 | 0.1093035 | 0.11913834999999999 | 0.12656972999999996 | 18259.79747693422 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 64 | ok | 59.494841 | 0.19337749999999998 | 0.20699865 | 0.2406471599999999 | 10212.36406832807 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 1 | ok | 58.520129 | 0.107422 | 0.1140134 | 0.11666433 | 36901.22004658779 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 2 | ok | 58.593695 | 0.119171 | 0.12482999999999998 | 0.12708824000000002 | 33378.82867681984 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 4 | ok | 58.504903 | 0.1205025 | 0.1276618 | 0.13246882 | 32971.88998005695 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 8 | ok | 59.001507 | 0.14301049999999998 | 0.15438464999999998 | 0.15889838 | 28011.039710970887 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 64 | ok | 56.675175 | 0.286622 | 0.32572589999999996 | 0.33730534 | 14061.528326421507 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 1 | ok | 58.858655 | 0.1120305 | 0.1188033 | 0.12279273 | 70630.14805491635 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 2 | ok | 58.207441 | 0.1249595 | 0.1316288 | 0.13589990999999998 | 63532.13916206826 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 4 | ok | 59.08307 | 0.1326355 | 0.1393506 | 0.14263107 | 60163.79291802969 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 8 | ok | 58.845241 | 0.145732 | 0.16039689999999998 | 0.16805779 | 54548.27617901326 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 64 | ok | 57.396683 | 0.2889075 | 0.3115254499999999 | 0.34223920999999996 | 28321.425236075007 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 1 | ok | 58.391163 | 0.11675450000000001 | 0.12327755 | 0.12655393 | 135910.44453078532 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 2 | ok | 58.528427 | 0.138823 | 0.146925 | 0.15324295 | 114458.92333927614 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 4 | ok | 59.143517 | 0.14170149999999998 | 0.1581927 | 0.16415802 | 110822.48011844707 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 8 | ok | 59.291369 | 0.152423 | 0.1614273 | 0.16468383 | 105509.85659891614 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 64 | ok | 57.456748 | 0.29513 | 0.3575903 | 0.4037404399999998 | 51619.11084016813 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 1 | ok | 59.0339 | 0.135583 | 0.1446977 | 0.14724285 | 233992.53417570645 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 2 | ok | 59.781369 | 0.1749885 | 0.18185215 | 0.18495933 | 185287.62951460545 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 4 | ok | 59.626207 | 0.1607155 | 0.17259775 | 0.17578573 | 197494.14491577246 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 8 | ok | 59.207678 | 0.1605625 | 0.17135945 | 0.17337545 | 200356.5093638493 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 64 | ok | 63.666203 | 0.304627 | 0.39782395 | 0.41010972999999995 | 101582.96738069336 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 1 | ok | 59.064586 | 0.1574955 | 0.16692169999999998 | 0.16987357 | 402212.26802721666 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 2 | ok | 60.277261 | 0.2248135 | 0.23003355 | 0.23147655 | 288369.6293071496 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 4 | ok | 60.071814 | 0.191642 | 0.2155025 | 0.22002596 | 331142.89209025877 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 8 | ok | 59.105589 | 0.18677850000000001 | 0.20064674999999998 | 0.20506860999999998 | 342639.97030168056 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 64 | ok | 66.128885 | 0.2972675 | 0.4503161 | 0.45743052999999995 | 190273.6396564514 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 1 | ok | 59.906705 | 0.2171945 | 0.22578275 | 0.22951346 | 587019.6024189611 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 2 | ok | 60.915979 | 0.2902615 | 0.34751065000000003 | 0.35554006 | 419472.07734960236 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 4 | ok | 60.407473 | 0.23437950000000002 | 0.27198964999999997 | 0.27871980999999996 | 526656.1670079371 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 8 | ok | 60.894492 | 0.2384685 | 0.274175 | 0.2776875 | 518195.00050321594 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 64 | ok | 66.574239 | 0.335909 | 0.40575669999999997 | 0.41006086 | 356944.76474969665 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 1 | ok | 1375.522138 | 0.0577675 | 0.06215239999999999 | 0.07228219999999998 | 17072.036482259082 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 2 | ok | 1361.925472 | 0.0618605 | 0.06868184999999999 | 0.07393043999999999 | 15970.272296336705 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 4 | ok | 1384.700406 | 0.056488 | 0.058526049999999996 | 0.05967728 | 17648.5309186378 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 8 | ok | 1421.796666 | 0.054578 | 0.058515199999999996 | 0.06313643999999999 | 18170.02418430219 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 64 | ok | 1518.437506 | 0.0562345 | 0.0578809 | 0.06036480999999999 | 17746.178537913645 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 1 | ok | 1382.162219 | 0.059756500000000004 | 0.06179735 | 0.06580927 | 33327.52323511601 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 2 | ok | 1387.794636 | 0.0614525 | 0.0634281 | 0.06660896999999999 | 32365.779919817018 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 4 | ok | 1386.460983 | 0.0587235 | 0.0606471 | 0.06618573999999998 | 33877.553647647015 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 8 | ok | 1455.337286 | 0.06103 | 0.06352624999999999 | 0.07138030999999997 | 32594.173530727992 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 64 | ok | 1529.235079 | 0.059556 | 0.06219695 | 0.0643686 | 33416.37302028875 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 1 | ok | 1363.767588 | 0.0607235 | 0.06294965 | 0.07012953999999998 | 65345.16952170603 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 2 | ok | 1395.258723 | 0.0600075 | 0.06294805 | 0.06396111 | 66253.70730900961 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 4 | ok | 1406.004731 | 0.062046500000000004 | 0.06373345 | 0.06886090999999998 | 64091.16327063615 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 8 | ok | 1337.672215 | 0.0597605 | 0.062356699999999994 | 0.08536429999999992 | 65820.99847821852 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 64 | ok | 1570.994588 | 0.058716000000000004 | 0.06017275 | 0.060840160000000004 | 68133.60591317939 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 1 | ok | 1387.315174 | 0.066272 | 0.06869444999999999 | 0.07570278 | 119969.0959608805 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 2 | ok | 1371.951691 | 0.06471450000000001 | 0.06730185 | 0.07112508 | 123245.94066385808 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 4 | ok | 1352.850756 | 0.0609265 | 0.0622514 | 0.06600758999999999 | 130827.80640362864 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 8 | ok | 1399.910229 | 0.06336 | 0.06632635 | 0.06895402 | 125827.82915223185 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 64 | ok | 1581.381911 | 0.065381 | 0.06711790000000001 | 0.07108114999999998 | 122003.10802917705 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 1 | ok | 1376.411093 | 0.062300499999999995 | 0.06401934999999999 | 0.07116930999999997 | 255278.3587133205 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 2 | ok | 1317.18988 | 0.062042 | 0.0648333 | 0.07187518 | 256197.2513877885 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 4 | ok | 1350.147203 | 0.075736 | 0.07786395 | 0.07931268 | 210905.55728234482 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 8 | ok | 1436.30104 | 0.0650415 | 0.0670974 | 0.07091828999999998 | 245203.43762959386 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 64 | ok | 1559.474112 | 0.065928 | 0.06705995 | 0.07234431999999998 | 241934.72782494326 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 1 | ok | 1367.305556 | 0.0719415 | 0.07443665 | 0.07897430999999999 | 442723.6023976804 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 2 | ok | 1318.975336 | 0.08799799999999999 | 0.09101645 | 0.09564211999999998 | 361634.3885372748 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 4 | ok | 1349.430478 | 0.0825965 | 0.08451 | 0.08927605999999999 | 385709.3713636345 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 8 | ok | 1382.187691 | 0.082237 | 0.08434935 | 0.0860626 | 388202.4310691993 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 64 | ok | 1620.021016 | 0.112944 | 0.1175903 | 0.12446514999999997 | 281857.12134037155 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 1 | ok | 1332.388029 | 0.07348750000000001 | 0.07606555 | 0.07861865999999999 | 869017.122624562 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 2 | ok | 1322.793508 | 0.101683 | 0.1035566 | 0.10658377 | 627900.6064734983 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 4 | ok | 1378.499246 | 0.0919625 | 0.0941014 | 0.09739504999999998 | 693308.9404137798 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 8 | ok | 1387.536888 | 0.08563 | 0.08777725 | 0.09041112 | 746305.3802320962 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 64 | ok | 1553.349818 | 0.112156 | 0.11634885 | 0.12201855999999998 | 568055.7404695335 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 1 | ok | 1332.389042 | 0.08666850000000001 | 0.089972 | 0.09439797 | 1469563.0529805033 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 2 | ok | 1325.32471 | 0.150421 | 0.15387779999999998 | 0.16032623999999998 | 848582.1782892371 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 4 | ok | 1381.039089 | 0.1472985 | 0.1511055 | 0.15456841 | 866134.2196773515 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 8 | ok | 1416.102807 | 0.11115749999999999 | 0.1128506 | 0.11737085999999998 | 1149392.4652654494 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 64 | ok | 1547.7821 | 0.13130150000000002 | 0.1368111 | 0.14356098 | 972299.9344760999 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 1 | ok | 1373.222722 | 0.105868 | 0.10979515000000001 | 0.1136827 | 9403.266807728207 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 2 | ok | 1377.996219 | 0.118586 | 0.12180505 | 0.12724338999999998 | 8408.376693930548 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 4 | ok | 1423.495751 | 0.11530950000000001 | 0.1259707 | 0.12865026999999998 | 8584.041717069298 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 8 | ok | 1352.830747 | 0.126864 | 0.1339504 | 0.14090919 | 7902.485853364949 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 64 | ok | 1584.891443 | 0.22617700000000002 | 0.24325514999999998 | 0.24832251 | 4482.694780932948 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 1 | ok | 1354.481547 | 0.111747 | 0.1162119 | 0.12424864 | 17749.564647553107 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 2 | ok | 1359.548691 | 0.11899 | 0.12456294999999999 | 0.12951742 | 16703.910819824603 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 4 | ok | 1325.613823 | 0.1258505 | 0.1395562 | 0.14225989 | 15715.074386518854 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 8 | ok | 1406.750542 | 0.1342835 | 0.14746574999999998 | 0.15569603999999998 | 14790.567049633888 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 64 | ok | 1562.181152 | 0.22492800000000002 | 0.24777995 | 1.058320669999997 | 7728.431570457483 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 1 | ok | 1389.144708 | 0.141878 | 0.14848505 | 0.15260589 | 28075.979777994806 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 2 | ok | 1358.025958 | 0.157864 | 0.16363995 | 0.16911659 | 25254.776499016196 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 4 | ok | 1418.866318 | 0.161904 | 0.17299425000000002 | 0.17618409999999998 | 24592.70182062231 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 8 | ok | 1486.764166 | 0.24584299999999998 | 0.2852631 | 0.2972601899999999 | 16298.656583231128 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 64 | ok | 1617.892122 | 0.3372285 | 0.39176815 | 0.39884991999999997 | 11540.955938534486 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 1 | ok | 1354.908304 | 0.14326149999999999 | 0.1494587 | 0.15359513 | 55530.89829418798 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 2 | ok | 1364.347642 | 0.170678 | 0.1781235 | 0.18470689999999998 | 46702.49993811919 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 4 | ok | 1371.5511 | 0.2432675 | 0.27455835 | 0.27994737 | 32254.04836688077 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 8 | ok | 1399.977305 | 0.1752335 | 0.19019325 | 0.19551879 | 45306.37815263016 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 64 | ok | 1551.684867 | 0.3051985 | 0.36438590000000004 | 1.7459493999999947 | 21361.884203634298 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 1 | ok | 1388.305109 | 0.151405 | 0.15610255 | 0.16220463999999998 | 105374.36944965206 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 2 | ok | 1368.415679 | 0.1792855 | 0.1842387 | 0.18734815 | 88931.17071994563 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 4 | ok | 1328.514145 | 0.2672825 | 0.28014589999999995 | 0.28901734999999995 | 65253.80221670428 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 8 | ok | 1396.939266 | 0.1945115 | 0.2090757 | 0.21199795999999999 | 82085.62330845899 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 64 | ok | 1584.794683 | 0.351003 | 0.38280935 | 0.39854324 | 46087.321763764114 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 1 | ok | 1389.041271 | 0.1765585 | 0.1848115 | 0.18634865 | 180337.04091261365 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 2 | ok | 1314.993368 | 0.21224500000000002 | 0.21974235 | 0.22547224999999999 | 153443.60046201866 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 4 | ok | 1391.665583 | 0.184631 | 0.19409265 | 0.19511952999999999 | 172970.1119538239 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 8 | ok | 1344.783168 | 0.2032215 | 0.21512314999999999 | 0.22334926 | 157568.32679670976 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 64 | ok | 1569.886811 | 0.3516815 | 0.37494425 | 0.39027832999999995 | 93000.24366063839 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 1 | ok | 1357.879652 | 0.198045 | 0.20716265 | 0.20991282 | 321239.3091306953 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 2 | ok | 1373.983959 | 0.2683045 | 0.2760353 | 0.28111375 | 244982.86574525546 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 4 | ok | 1391.329149 | 0.2325975 | 0.24521565 | 0.24841759 | 272739.27480501914 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 8 | ok | 1411.921909 | 0.26081299999999996 | 0.3371552 | 0.35741139 | 233547.13357020364 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 64 | ok | 1521.664799 | 0.3464 | 0.40858764999999997 | 0.41906856 | 178442.3068419186 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 1 | ok | 1324.844841 | 0.25553349999999997 | 0.26322134999999997 | 0.26740671 | 499210.27274899266 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 2 | ok | 1315.780478 | 0.3854605 | 0.39247469999999995 | 0.3971731 | 362865.76869484445 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 4 | ok | 1412.065694 | 0.263828 | 0.3008958 | 0.30747896999999996 | 470772.6674097029 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 8 | ok | 1372.449156 | 0.2618305 | 0.29451629999999995 | 0.29987363 | 480005.8080702776 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 64 | ok | 1610.556726 | 0.38442750000000003 | 0.41375735 | 0.43248174 | 339785.66850945534 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0818085 | 0.08698329999999999 | 0.09398380999999999 | 12129.975109291077 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.088122 | 0.09186145 | 0.09942622999999998 | 11269.827853379538 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0836615 | 0.08909045 | 0.09378397999999999 | 11856.902365286025 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.083093 | 0.08662375 | 0.09303287999999998 | 11948.65614659185 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0801505 | 0.0838365 | 0.08686037 | 12386.45951882073 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1018535 | 0.10870674999999999 | 0.11317086999999999 | 19439.13809971928 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.1022035 | 0.10693875 | 0.11577316 | 19401.50245234991 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0993845 | 0.10629834999999999 | 0.11422075999999999 | 19942.426215515807 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.103255 | 0.11103099999999999 | 0.11463785 | 19333.198000947326 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.090141 | 0.09922375 | 0.10219264 | 21847.417635235513 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.100562 | 0.1053162 | 0.11109514999999998 | 39544.33849637398 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.1035335 | 0.10788465 | 0.11365811999999999 | 38514.28080646593 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.102335 | 0.10988085 | 0.11853691 | 38643.57944656211 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1432715 | 0.15075355 | 0.16113691 | 30136.05827590405 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.088532 | 0.0926728 | 0.09704119 | 44844.32628462738 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.1032135 | 0.10986974999999999 | 0.11746364999999998 | 76807.79138235783 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.107045 | 0.1130051 | 0.11830331999999999 | 74148.59341045158 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.10404 | 0.1111669 | 0.11527309999999999 | 76200.65079165807 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.10144 | 0.10733085 | 0.11917784000000001 | 78059.6250640089 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.1043995 | 0.1096441 | 0.11477603999999998 | 76038.81857727189 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.106077 | 0.1103999 | 0.12619952999999995 | 149457.56245924946 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.108243 | 0.11249849999999999 | 0.12106185999999998 | 146842.66249241924 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1261285 | 0.1326446 | 0.13928635999999997 | 125884.2978535783 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.11143549999999999 | 0.11583589999999999 | 0.11997147999999999 | 142975.50615116372 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.091592 | 0.0973249 | 0.10022838999999999 | 173092.50972455356 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.110001 | 0.11506445 | 0.12181608999999997 | 288820.86355272046 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.13838499999999998 | 0.1455707 | 0.1540624 | 229432.09390770318 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.14023950000000002 | 0.14482955 | 0.15262732 | 227010.18225796244 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.122837 | 0.1314907 | 0.13350355 | 258571.2741212657 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.140496 | 0.1448871 | 0.15281422 | 226522.33984552027 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.129774 | 0.136449 | 0.14287018999999998 | 489207.3968769916 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.160968 | 0.1685264 | 0.17505055 | 395302.08182135434 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1545725 | 0.1613425 | 0.17125564999999998 | 411715.3076137354 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.14429750000000002 | 0.15299749999999998 | 0.15909953000000002 | 439887.0919806659 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.149319 | 0.1730894 | 0.17519327999999998 | 413476.2238250168 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.150975 | 0.16155635 | 0.16522682 | 838156.6630049697 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.2288645 | 0.239584 | 0.25234501 | 563750.1434700072 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.2119255 | 0.221188 | 0.22828529 | 610854.4249912762 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1833665 | 0.1913759 | 0.19535707 | 698373.3248451876 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.1903305 | 0.19761309999999999 | 0.19928617 | 670197.560627485 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.154706 | 0.16229025 | 0.16893376 | 6411.050394189844 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1587975 | 0.16765124999999997 | 0.16934621 | 6253.116396884297 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.16118549999999998 | 0.17092439999999998 | 0.17930558999999996 | 6128.5903273929725 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.1791285 | 0.18510705 | 0.19501661999999997 | 5596.801495107165 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.27553300000000003 | 0.2948871 | 0.29861508999999997 | 3655.041563305137 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.191441 | 0.1975038 | 0.2028823 | 10402.20868016464 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1964135 | 0.20308379999999998 | 0.20633338 | 10161.135283322934 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.198814 | 0.2087494 | 0.21343484 | 9989.273518096268 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.21799849999999998 | 0.2337995 | 0.23850607 | 9124.899340954145 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.3707555 | 0.42969625 | 0.44449224 | 5292.629484180331 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.22956700000000002 | 0.23899265 | 0.24360605999999999 | 17325.36401239491 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.23397600000000002 | 0.24323939999999997 | 0.24666582 | 17095.885951977143 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.23919449999999998 | 0.25726155 | 0.26363269 | 16551.01369993608 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.2609545 | 0.27749475 | 0.27955842999999997 | 15358.088421736793 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.4223625 | 0.50265445 | 0.50999604 | 8956.452786126312 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.22627999999999998 | 0.23547164999999998 | 0.24150944 | 35097.451144128645 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.2545305 | 0.26262775 | 0.26575934 | 31327.67795648304 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.24640299999999998 | 0.2609482 | 0.26588040999999996 | 32543.399267073837 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.2736305 | 0.2952454 | 0.30005377 | 29136.697145594324 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.4792555 | 0.56296525 | 0.5846053099999999 | 17158.862194132023 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.2376995 | 0.2475346 | 0.25121365 | 67060.91780242714 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.2686635 | 0.27854555 | 0.28458326 | 59630.149957901114 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.2591555 | 0.27191355 | 0.27537607 | 61448.69916944402 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.28063499999999997 | 0.3051366 | 0.30915925 | 56823.59781734879 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.4994865 | 0.5931382 | 0.60933283 | 32830.45556607133 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.256021 | 0.2628106 | 0.26786111 | 125052.59047613539 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.31350199999999995 | 0.32369745 | 0.32924014 | 103668.55814270522 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.2804735 | 0.2976985 | 0.30249351 | 114730.7877917837 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.298929 | 0.317625 | 0.32426681999999996 | 106952.00034331593 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.4437795 | 0.5267636 | 0.53407196 | 69854.15543815691 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.2896215 | 0.2994696 | 0.302318 | 220323.70509561908 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.33447899999999997 | 0.37976075 | 0.3819246 | 182463.0435360243 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.3169885 | 0.34690275 | 0.36157081999999996 | 199495.25207534287 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.321721 | 0.35630475 | 0.36002895 | 196386.89839175087 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.4872665 | 0.6002953 | 0.6236501 | 130345.78088018841 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.3562645 | 0.3644947 | 0.36550021 | 358734.96375011216 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.44496 | 0.52272275 | 0.5371503 | 296809.02013742534 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.3841735 | 0.4419705 | 0.44556843 | 327676.2755003821 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.37552050000000003 | 0.39858309999999997 | 0.43490576 | 343860.57098585094 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.5173544999999999 | 0.6118384 | 0.6124497799999999 | 247868.7355340514 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 1 | ok | 62.23468 | 0.0445885 | 0.0502264 | 0.05186763999999999 | 21975.838883697706 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 2 | ok | 62.657953 | 0.045815499999999995 | 0.05135175 | 0.053130739999999996 | 21536.908230372555 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 4 | ok | 62.350391 | 0.0481365 | 0.054511399999999995 | 0.055984559999999996 | 20399.19594529262 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 8 | ok | 62.201527 | 0.0462335 | 0.052201449999999996 | 0.05406968 | 21283.143707617277 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 64 | ok | 61.024319 | 0.041651 | 0.04693605 | 0.04775503 | 23611.30131326058 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 1 | ok | 62.141817 | 0.0495625 | 0.05336394999999999 | 0.05488725 | 40304.29744571515 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 2 | ok | 62.726102 | 0.048484 | 0.054123199999999996 | 0.05518613 | 40385.48755581779 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 4 | ok | 62.251874 | 0.051497 | 0.05502699999999999 | 0.05943178999999999 | 38427.10186640434 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 8 | ok | 62.518174 | 0.0484675 | 0.05182894999999999 | 0.05284982 | 41225.6212288863 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 64 | ok | 60.722442 | 0.0454465 | 0.046978849999999996 | 0.04960161 | 43809.28765660177 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 1 | ok | 62.772478 | 0.047494499999999995 | 0.05292544999999999 | 0.05760195 | 82752.27444626315 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 2 | ok | 63.130062 | 0.0501855 | 0.0530198 | 0.05487492 | 78965.58245605862 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 4 | ok | 63.413865 | 0.0534365 | 0.0567015 | 0.0584322 | 74922.27750237598 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 8 | ok | 63.444516 | 0.048802 | 0.05236244999999999 | 0.055126619999999994 | 81384.08281969804 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 64 | ok | 61.375503 | 0.046088000000000004 | 0.0494313 | 0.053006349999999994 | 85971.58639069788 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 1 | ok | 62.910113 | 0.051172 | 0.05410155 | 0.05586763 | 154618.95125830837 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 2 | ok | 63.422102 | 0.050060499999999994 | 0.0535533 | 0.05645906 | 158519.49135850806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 4 | ok | 63.341673 | 0.054572499999999996 | 0.057954 | 0.06030558 | 147985.52849516846 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 8 | ok | 63.197524 | 0.0522615 | 0.05599165 | 0.057539379999999994 | 152155.81973187102 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 64 | ok | 60.807122 | 0.047217499999999996 | 0.049486949999999995 | 0.05134808 | 168613.946986932 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 1 | ok | 63.447108 | 0.0544805 | 0.058058549999999993 | 0.060165949999999996 | 291330.9381074705 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 2 | ok | 62.785535 | 0.055236 | 0.0577436 | 0.05992331 | 293220.7365704903 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 4 | ok | 63.11113 | 0.060669 | 0.06345069999999998 | 0.06798149999999999 | 262173.6225152085 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 8 | ok | 62.884771 | 0.0507045 | 0.05440785 | 0.05652682999999999 | 312658.52764409455 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 64 | ok | 61.244256 | 0.049579 | 0.052183049999999995 | 0.05790550999999998 | 320017.40894704673 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 1 | ok | 63.712385 | 0.0557435 | 0.059053549999999996 | 0.060211110000000005 | 570771.3689665899 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 2 | ok | 63.899543 | 0.06600500000000001 | 0.07110525 | 0.07284087 | 479616.8820346307 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 4 | ok | 63.553524 | 0.0663195 | 0.07391865 | 0.07510199 | 470525.95979943825 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 8 | ok | 63.981478 | 0.0655275 | 0.06973209999999999 | 0.07335989 | 485934.03202548233 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 64 | ok | 68.795509 | 0.089058 | 0.0941137 | 0.09510369 | 357716.1611690164 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 1 | ok | 63.219552 | 0.0630985 | 0.06502145 | 0.06950692 | 1008787.1667140586 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 2 | ok | 64.164573 | 0.093196 | 0.113203 | 0.11547439999999999 | 647201.6718837188 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 4 | ok | 64.207703 | 0.077135 | 0.08543815 | 0.08584903 | 823110.0492991476 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 8 | ok | 64.084881 | 0.0861745 | 0.0907823 | 0.09392067999999999 | 756986.8705358473 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 64 | ok | 70.132272 | 0.107518 | 0.11403775 | 0.11494654 | 590152.9658045085 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 1 | ok | 63.831491 | 0.0840215 | 0.08885719999999998 | 0.09189401999999999 | 1512666.21601819 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 2 | ok | 65.495679 | 0.12988650000000002 | 0.16093795 | 0.16318289 | 911531.0674000313 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 4 | ok | 64.945238 | 0.119759 | 0.12587505 | 0.13062348 | 1063611.2660364327 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 8 | ok | 66.119667 | 0.1130205 | 0.12149375 | 0.12777207 | 1124536.0410286973 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 64 | ok | 75.45671 | 0.131696 | 0.13809375 | 0.14950328999999996 | 965432.3915238658 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 1 | ok | 61.99922 | 0.0895945 | 0.09290369999999999 | 0.09709859 | 11125.339183778366 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 2 | ok | 62.930865 | 0.1000965 | 0.10848909999999999 | 0.11170561 | 9878.190062698848 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 4 | ok | 62.790253 | 0.1010015 | 0.1111026 | 0.11864654 | 9765.49148742107 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 8 | ok | 62.225056 | 0.108704 | 0.11953565000000001 | 0.12018212 | 9180.266466414454 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 64 | ok | 60.469954 | 0.2053385 | 0.22570255 | 0.23213198999999998 | 4893.607109119217 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 1 | ok | 63.195694 | 0.122477 | 0.12633405 | 0.13272895 | 16264.72994936952 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 2 | ok | 62.537933 | 0.119983 | 0.12486625 | 0.12750749 | 16592.3412739301 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 4 | ok | 62.360655 | 0.1259045 | 0.1352144 | 0.13825696 | 15748.316701798536 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 8 | ok | 65.844107 | 0.152038 | 0.1653125 | 0.16752452 | 13138.18572953915 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 64 | ok | 60.671113 | 0.2932325 | 0.3083771 | 0.31440348 | 6979.946195782744 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 1 | ok | 62.812167 | 0.1371925 | 0.144705 | 0.14855557 | 28971.370057538585 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 2 | ok | 62.986988 | 0.153662 | 0.15988059999999998 | 0.16364643999999998 | 26045.59046200059 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 4 | ok | 62.696221 | 0.1513875 | 0.16340369999999999 | 0.16512781999999998 | 26107.595143256287 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 8 | ok | 63.155305 | 0.181414 | 0.20023844999999998 | 0.20622483 | 21839.869205391304 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 64 | ok | 60.714833 | 0.383207 | 0.4415540499999999 | 0.46672262 | 10887.510082514806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 1 | ok | 63.082898 | 0.1451365 | 0.1525823 | 0.15650395 | 54805.571753646764 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 2 | ok | 63.031848 | 0.165864 | 0.1724447 | 0.17890943999999998 | 47963.873610396644 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 4 | ok | 63.433753 | 0.16801149999999998 | 0.17838915 | 0.18122708 | 47229.95153498524 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 8 | ok | 62.562624 | 0.182376 | 0.199948 | 0.21041290999999998 | 43386.21136308739 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 64 | ok | 61.945778 | 0.3878595 | 0.4106216 | 0.4409165699999999 | 21651.32021090118 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 1 | ok | 63.286211 | 0.154964 | 0.16365955 | 0.16575877 | 102419.10075261396 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 2 | ok | 62.987503 | 0.17216700000000001 | 0.18714804999999998 | 0.19121325 | 91078.548758958 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 4 | ok | 63.000514 | 0.1719 | 0.18157555 | 0.19647947999999996 | 91810.924581916 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 8 | ok | 63.148796 | 0.194816 | 0.2117203 | 0.21465212 | 81719.20036536655 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 64 | ok | 60.709246 | 0.3852725 | 0.46185315 | 0.47275247000000004 | 41538.3175922148 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 1 | ok | 63.053493 | 0.170412 | 0.1801273 | 0.18527238 | 186448.41636541655 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 2 | ok | 64.376574 | 0.2124935 | 0.22384295 | 0.22826633000000002 | 149007.24389403238 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 4 | ok | 63.607886 | 0.2015315 | 0.2118573 | 0.21451066 | 157832.81342218138 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 8 | ok | 63.791864 | 0.211089 | 0.2340749 | 0.23951898 | 150410.941493528 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 64 | ok | 68.834692 | 0.3252735 | 0.47081195000000003 | 0.47222043999999996 | 86942.32745914742 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 1 | ok | 63.802369 | 0.212157 | 0.21773095 | 0.22381256999999996 | 300771.0736427635 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 2 | ok | 64.503218 | 0.2970715 | 0.30683565 | 0.30791483 | 227494.87407838806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 4 | ok | 64.055401 | 0.2570435 | 0.26848045 | 0.27492634 | 254868.60766633623 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 8 | ok | 64.105613 | 0.2398535 | 0.2671869 | 0.27268924 | 265457.3976132891 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 64 | ok | 68.986652 | 0.367095 | 0.44562955000000004 | 0.44903992 | 168475.72743870458 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 1 | ok | 63.89825 | 0.27985899999999997 | 0.28953035 | 0.29051234000000004 | 455418.6097564186 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 2 | ok | 64.791283 | 0.36758599999999997 | 0.44594845 | 0.45160102 | 361316.75115268514 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 4 | ok | 64.643949 | 0.2979505 | 0.3712009 | 0.37305134 | 406136.36510745424 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 8 | ok | 65.077573 | 0.29884849999999996 | 0.34998085 | 0.36226691 | 423625.93618022325 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 64 | ok | 74.24909 | 0.40400349999999996 | 0.48173299999999997 | 0.4907766 | 301763.1407356806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 1 | ok | 1367.459034 | 0.067373 | 0.07201894999999998 | 0.07568517999999999 | 14720.498942479355 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 2 | ok | 1352.328051 | 0.069506 | 0.07487854999999999 | 0.07888979 | 14263.8050811097 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 4 | ok | 1327.827511 | 0.0673405 | 0.07010519999999999 | 0.07198421999999999 | 14804.339921856774 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 8 | ok | 1416.609568 | 0.0691985 | 0.07381135 | 0.07490658 | 14368.976345503761 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 64 | ok | 1593.644581 | 0.066883 | 0.06935285 | 0.07544716 | 14839.542989562458 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 1 | ok | 1298.59582 | 0.070465 | 0.0735885 | 0.07936111 | 28134.43646583087 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 2 | ok | 1323.836726 | 0.069217 | 0.07192865 | 0.07410931999999999 | 28718.137226746927 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 4 | ok | 1410.589067 | 0.0735285 | 0.07579735 | 0.08582096999999998 | 27001.645480275565 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 8 | ok | 1445.09022 | 0.07374249999999999 | 0.07574465 | 0.08062551999999999 | 27023.492062119443 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 64 | ok | 1511.317272 | 0.071098 | 0.0738214 | 0.07917842999999998 | 27976.887733505595 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 1 | ok | 1377.758534 | 0.0727415 | 0.0751526 | 0.08082319999999998 | 54676.0308482166 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 2 | ok | 1376.102478 | 0.074244 | 0.07629135 | 0.07947523999999999 | 53720.32019459648 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 4 | ok | 1403.843488 | 0.07283 | 0.07559065 | 0.07961737999999999 | 54634.13568114425 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 8 | ok | 1349.043544 | 0.0735565 | 0.0753479 | 0.07824858 | 54181.38029775378 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 64 | ok | 1662.204093 | 0.071114 | 0.0731308 | 0.07439607 | 56102.63415893035 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 1 | ok | 1360.255879 | 0.0726055 | 0.07396715 | 0.07454496999999999 | 110184.45980415813 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 2 | ok | 1361.692404 | 0.0733555 | 0.074827 | 0.0791267 | 108927.92321017126 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 4 | ok | 1345.65698 | 0.075732 | 0.0773375 | 0.08291161999999999 | 105324.77288687567 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 8 | ok | 1375.138925 | 0.0727125 | 0.075222 | 0.08018261999999998 | 109292.60362304981 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 64 | ok | 1574.807191 | 0.077152 | 0.07910685 | 0.08264268999999999 | 103359.41349734405 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 1 | ok | 1387.961415 | 0.079257 | 0.08124995 | 0.08652518999999997 | 201352.53531786392 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 2 | ok | 1367.012047 | 0.0750475 | 0.0769165 | 0.08028991999999999 | 212417.5587540329 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 4 | ok | 1417.07431 | 0.091597 | 0.09825949999999999 | 0.10198916999999999 | 173618.5336048438 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 8 | ok | 1396.700306 | 0.0745115 | 0.07809189999999999 | 0.08187633 | 213288.2868471512 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 64 | ok | 1551.675563 | 0.0779865 | 0.0800476 | 0.08461913999999998 | 204141.62529396394 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 1 | ok | 1343.419015 | 0.080018 | 0.08278115 | 0.08957615999999999 | 397552.9621298503 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 2 | ok | 1316.810069 | 0.1075055 | 0.10940555 | 0.11328710999999998 | 296988.94050309184 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 4 | ok | 1371.135871 | 0.10571 | 0.1078224 | 0.11102021 | 302285.0672877113 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 8 | ok | 1404.486787 | 0.0955925 | 0.0990607 | 0.10395623999999999 | 332917.3253921454 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 64 | ok | 1582.135432 | 0.13200250000000002 | 0.13527440000000002 | 0.13647681 | 242093.38146956736 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 1 | ok | 1315.326969 | 0.091782 | 0.0944568 | 0.09842329999999999 | 694246.4757228517 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 2 | ok | 1346.81655 | 0.13095800000000002 | 0.13545000000000001 | 0.13780546999999999 | 486865.2874178795 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 4 | ok | 1403.378641 | 0.124535 | 0.1275781 | 0.13370111 | 512764.3059638975 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 8 | ok | 1412.938112 | 0.1153265 | 0.1191223 | 0.12937352 | 551151.7952994675 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 64 | ok | 1555.497727 | 0.1556995 | 0.1620742 | 0.16699489 | 408984.46855919115 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 1 | ok | 1353.410377 | 0.11374000000000001 | 0.11754479999999999 | 0.12232020999999998 | 1118263.3370375806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 2 | ok | 1370.110743 | 0.184438 | 0.18982829999999998 | 0.19945340999999997 | 702790.6168228526 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 4 | ok | 1400.046223 | 0.17654199999999998 | 0.18318954999999998 | 0.18735237 | 722630.7281328836 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 8 | ok | 1480.974483 | 0.14256649999999998 | 0.14496905 | 0.15116678999999997 | 900491.4713608535 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 64 | ok | 1550.7726 | 0.171329 | 0.17972335 | 0.18353464 | 743785.5266541935 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 1 | ok | 1391.555022 | 0.123999 | 0.12869589999999997 | 0.13096767999999998 | 8039.4124154253805 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 2 | ok | 1363.072638 | 0.130611 | 0.13827895 | 0.14227254 | 7597.860442499392 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 4 | ok | 1410.157115 | 0.1323035 | 0.14073245 | 0.1446609 | 7474.885133440153 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 8 | ok | 1355.375876 | 0.144914 | 0.15606135 | 0.16139521999999998 | 6888.888429629661 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 64 | ok | 1631.876198 | 0.2402725 | 0.26849955 | 0.27833114 | 4084.6155894745666 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 1 | ok | 1297.345951 | 0.158601 | 0.16446444999999998 | 0.16840053 | 12566.789343664237 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 2 | ok | 1403.510976 | 0.1728805 | 0.18126935 | 0.18383757 | 11516.569867977498 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 4 | ok | 1335.099272 | 0.1710305 | 0.18517755 | 0.18997895 | 11526.18703589808 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 8 | ok | 1518.270409 | 0.189917 | 0.21272729999999998 | 0.22068042 | 10470.125381845473 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 64 | ok | 1546.804747 | 0.342951 | 0.39640735 | 0.40898108 | 5675.001730875529 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 1 | ok | 1371.543607 | 0.1942645 | 0.2011747 | 0.20551294999999997 | 20494.938416296674 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 2 | ok | 1410.173908 | 0.2107095 | 0.21543245 | 0.21869414 | 18966.933711609858 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 4 | ok | 1363.662145 | 0.204483 | 0.21880239999999998 | 0.22140334 | 19360.180870553766 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 8 | ok | 1421.495277 | 0.2353765 | 0.2544222 | 0.25780324 | 16940.026462015336 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 64 | ok | 1616.694118 | 0.4372975 | 0.5461622 | 0.5531096799999999 | 8918.951702092743 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 1 | ok | 1333.730755 | 0.19040649999999998 | 0.20105705 | 0.20379919999999999 | 41759.81279911118 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 2 | ok | 1368.782012 | 0.2196515 | 0.22712545 | 0.23031363 | 36380.87428334225 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 4 | ok | 1340.181808 | 0.3597325 | 0.3765536 | 0.38118672 | 24728.961310674516 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 8 | ok | 1383.233043 | 0.231362 | 0.25060095 | 0.25843859999999996 | 34261.85824327461 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 64 | ok | 1624.172161 | 0.5250215 | 0.79444995 | 3.4136901699999904 | 12179.365639567568 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 1 | ok | 1372.937293 | 0.2087445 | 0.21965105000000001 | 0.22227198999999997 | 76253.27984419929 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 2 | ok | 1373.680866 | 0.244203 | 0.25053505 | 0.26202016999999994 | 65603.8407112506 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 4 | ok | 1372.21689 | 0.230416 | 0.2462598 | 0.26001624999999995 | 68686.68482857091 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 8 | ok | 1394.054338 | 0.244981 | 0.2622493 | 0.27013320999999996 | 64610.51248573925 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 64 | ok | 1503.692554 | 0.41062750000000003 | 0.4956526 | 0.50694458 | 37569.98113423397 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 1 | ok | 1365.542584 | 0.2211595 | 0.22987915 | 0.2330669 | 144284.60715900545 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 2 | ok | 1382.357452 | 0.2782825 | 0.28637515 | 0.28794576 | 116403.60301162311 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 4 | ok | 1366.772031 | 0.2940695 | 0.3810032 | 0.38932622 | 109070.70126462025 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 8 | ok | 1396.556986 | 0.2709025 | 0.28306945 | 0.2844146 | 118279.03413932119 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 64 | ok | 1553.346287 | 0.4566345 | 0.5447599 | 2.1689317699999937 | 62369.41160814032 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 1 | ok | 1378.04513 | 0.2674745 | 0.27665179999999995 | 0.28244456 | 238393.49602944462 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 2 | ok | 1404.605549 | 0.3191165 | 0.36379895 | 0.36659097 | 193076.38928568433 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 4 | ok | 1365.872333 | 0.311934 | 0.3234863 | 0.33047583 | 210050.3858363025 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 8 | ok | 1378.45704 | 0.2830385 | 0.31583 | 0.31789753000000004 | 221607.7378219796 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 64 | ok | 1623.950944 | 0.46135550000000003 | 0.5708712 | 0.7414704299999993 | 134277.19008195103 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 1 | ok | 1362.075959 | 0.3414645 | 0.35018065 | 0.36090847 | 373667.09589728125 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 2 | ok | 1337.598444 | 0.42647999999999997 | 0.51333815 | 0.5288938999999999 | 308239.37330311816 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 4 | ok | 1349.74518 | 0.34303 | 0.39210575 | 0.39816177999999997 | 360177.91888748243 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 8 | ok | 1412.83007 | 0.3347285 | 0.3884038 | 0.39678671 | 376493.35500993824 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 64 | ok | 1557.717983 | 0.4643615 | 0.5482478 | 0.5553296799999999 | 279941.76161576784 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0922635 | 0.0971958 | 0.10467238999999998 | 10765.864200680875 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.096876 | 0.1028332 | 0.10919256999999999 | 10230.036697187641 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0924685 | 0.0987194 | 0.10788511 | 10704.884531762998 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.090457 | 0.09470185 | 0.10158753999999998 | 10999.132608402502 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0943435 | 0.09748725 | 0.10089028 | 10559.72230464672 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1176065 | 0.12778655 | 0.12960429 | 16859.085023737593 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.11048250000000001 | 0.11698259999999999 | 0.12382710999999999 | 17942.92606419943 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1162585 | 0.1237873 | 0.12833876 | 17121.358586652394 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.11221049999999999 | 0.1168329 | 0.12483488999999999 | 17732.266536535473 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.100266 | 0.1042863 | 0.10938403999999999 | 19818.458952305096 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1180625 | 0.123432 | 0.12713579 | 33683.18372105204 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.116996 | 0.1214181 | 0.12695236999999998 | 34048.389230290195 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.115885 | 0.1224604 | 0.13674946999999998 | 34211.70752000701 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1164985 | 0.12067365 | 0.12544155 | 34187.128375231085 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.1038125 | 0.12655809999999992 | 0.14171222 | 37551.619394810594 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.117511 | 0.12330385 | 0.12914272 | 67744.848512357 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1178205 | 0.12469559999999999 | 0.13544464 | 67203.27360586388 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.1176735 | 0.12541935 | 0.13409336 | 67481.11878296452 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.117562 | 0.1238388 | 0.13190148 | 67569.57639281168 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.1186365 | 0.1236364 | 0.13036103999999998 | 69775.21043767373 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.1210675 | 0.1275152 | 0.13952019 | 130705.69146496737 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.126031 | 0.13403695 | 0.14308484999999999 | 125777.00711410474 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1390685 | 0.14535320000000002 | 0.14962298999999998 | 114324.07817994924 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1185155 | 0.12275275 | 0.13095789 | 134119.9917918565 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.112024 | 0.1159346 | 0.12046954 | 142149.51748234726 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.1284205 | 0.1337661 | 0.13911843999999998 | 247941.12018248465 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.15408650000000002 | 0.16134265 | 0.17908815999999994 | 205832.2570021561 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1585105 | 0.16586695 | 0.1757849 | 200330.3698211313 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1410895 | 0.1478007 | 0.15366876 | 225261.9444442098 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.1782785 | 0.18279925 | 0.18610592 | 179859.956541338 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.14922950000000001 | 0.15733334999999998 | 0.16764559999999998 | 425740.8789340619 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.19634449999999998 | 0.201217 | 0.20809599999999998 | 332590.9971566588 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.17928699999999997 | 0.18593995 | 0.19169876 | 355139.2928444205 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1745265 | 0.18118215 | 0.19927361999999998 | 363954.3273714581 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.18988349999999998 | 0.21732155 | 0.21968217 | 332966.0648270361 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.18481 | 0.19766505 | 0.20507979999999998 | 686385.5953402999 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.238246 | 0.26809185 | 0.27397755 | 520118.5090022761 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.2451895 | 0.2528285 | 0.26289754 | 544115.2953104998 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.221574 | 0.22847725 | 0.23370283999999997 | 574480.1896789967 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.23560599999999998 | 0.2489781 | 0.25008374 | 541435.8302943838 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.174581 | 0.18483344999999998 | 0.18760481 | 5695.232395666475 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.17532350000000002 | 0.18391405 | 0.18533505 | 5679.908312648054 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.179792 | 0.1901012 | 0.20443054 | 5499.775609155146 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.1998915 | 0.2167199 | 0.22073369999999998 | 4984.596599587953 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.285382 | 0.32154464999999993 | 0.33139696 | 3406.07761415435 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.23934149999999998 | 0.2487593 | 0.25208363 | 8314.53693683147 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.242505 | 0.249855 | 0.25539814 | 8232.785636291832 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2537825 | 0.2700574 | 0.27361088 | 7842.686377324347 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.2778215 | 0.29707185 | 0.29986855 | 7178.394296564737 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.4727345 | 0.5540169500000001 | 0.5809239199999999 | 4160.374100839147 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.2799425 | 0.28806535 | 0.29494972999999997 | 14243.187180903646 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.28943850000000004 | 0.29760654999999997 | 0.30075322 | 13796.993248855264 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.295674 | 0.316202 | 0.32169606 | 13415.966744501633 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.3194415 | 0.33980645 | 0.35123351999999997 | 12482.656129474102 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.489665 | 0.6140732 | 0.6175392 | 7787.819911960254 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.272902 | 0.28291995 | 0.28804522 | 29254.630770357067 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.30657100000000004 | 0.319677 | 0.32120943999999996 | 26048.760544619676 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.3071385 | 0.32206049999999997 | 0.32676478 | 26041.334368389565 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.318243 | 0.3398172 | 0.35068 | 25011.065833439687 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.5216025 | 0.6660469499999998 | 0.7180862499999999 | 14571.800891349774 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.28934099999999996 | 0.30436949999999996 | 0.31316973 | 54937.15018340424 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.3219785 | 0.3313871 | 0.33403372000000003 | 50225.26344563263 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.3251615 | 0.3403973 | 0.34229276 | 49263.468047160415 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.35225399999999996 | 0.37681855 | 0.38405476 | 45255.90289712451 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.531628 | 0.6433783 | 0.69533879 | 29067.31171875923 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.306459 | 0.31626355 | 0.32026148 | 104108.96616998664 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.35751 | 0.38784055 | 0.39012501 | 87701.98383531848 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.34617549999999997 | 0.36108165 | 0.36594427999999996 | 94129.41877319482 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.342083 | 0.3773118 | 0.3831999 | 92711.92163478467 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.5288695 | 0.7023676 | 0.73210181 | 57585.546027946984 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.357665 | 0.36848960000000003 | 0.37653276999999996 | 178362.30076443293 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.4039045 | 0.46116314999999997 | 0.46226339 | 152184.86580481206 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.3831445 | 0.4252372 | 0.43002638 | 167464.670711078 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.3832425 | 0.4107327 | 0.41963976999999997 | 167681.7613962944 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.563824 | 0.7085259999999998 | 0.77269241 | 109696.24287625741 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.43624450000000004 | 0.4440903 | 0.44719216 | 292724.80968084856 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.45955650000000003 | 0.55402955 | 0.64476188 | 262661.39773374924 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.457628 | 0.5403165999999999 | 0.55540212 | 276701.3088058376 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.438463 | 0.52260745 | 0.52679805 | 287435.24005446 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.582778 | 0.6805342999999999 | 0.7126187499999999 | 217210.33508495367 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 1 | ok | 66.928948 | 0.046806 | 0.0501935 | 0.05209796999999999 | 21088.557601708006 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 2 | ok | 67.087394 | 0.0481485 | 0.05178775 | 0.05215706 | 20550.110004738854 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 4 | ok | 66.851906 | 0.048236 | 0.0525447 | 0.05614588999999999 | 20562.947706778945 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 8 | ok | 67.230256 | 0.046882 | 0.0526794 | 0.05493147 | 20976.63873697141 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 64 | ok | 64.303903 | 0.045199 | 0.0495001 | 0.04966689 | 21823.035261223915 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 1 | ok | 67.208942 | 0.050813 | 0.054269 | 0.05860052999999999 | 38873.62882992709 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 2 | ok | 67.200988 | 0.050539 | 0.05239415 | 0.053727809999999994 | 39513.21302086812 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 4 | ok | 67.969438 | 0.0502285 | 0.05273165 | 0.05745392999999999 | 39513.55650853473 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 8 | ok | 66.641026 | 0.0527975 | 0.05452275 | 0.05974633999999998 | 37597.476156620556 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 64 | ok | 66.70495 | 0.050506999999999996 | 0.0520524 | 0.053717709999999995 | 39379.970244494485 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 1 | ok | 67.269008 | 0.051331 | 0.05300385 | 0.05425629 | 77639.93239114687 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 2 | ok | 67.63616 | 0.0509515 | 0.0527708 | 0.05752402999999999 | 77801.90933665703 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 4 | ok | 67.706865 | 0.0556315 | 0.0575337 | 0.06226527999999999 | 71550.84671483231 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 8 | ok | 67.664871 | 0.052706 | 0.05735075 | 0.05885732 | 74125.35784016497 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 64 | ok | 66.668744 | 0.050286 | 0.05311705 | 0.05424022 | 78692.38438711617 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 1 | ok | 67.755368 | 0.0525485 | 0.05407465 | 0.05762831999999999 | 151495.35382686733 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 2 | ok | 67.881201 | 0.052924 | 0.0568138 | 0.058791159999999995 | 149030.9078925651 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 4 | ok | 67.669571 | 0.054475499999999996 | 0.05780535 | 0.060472769999999995 | 145137.72299957584 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 8 | ok | 67.898371 | 0.0523595 | 0.0549895 | 0.05838847 | 151551.25975090277 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 64 | ok | 67.742172 | 0.0514335 | 0.05245995 | 0.05371127 | 155439.77019784375 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 1 | ok | 68.176078 | 0.0558825 | 0.058509200000000004 | 0.062285469999999996 | 284674.20816089783 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 2 | ok | 67.64966 | 0.055191000000000004 | 0.059789049999999996 | 0.0608669 | 285015.6918951869 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 4 | ok | 68.453117 | 0.062144000000000005 | 0.06681844999999999 | 0.06935448 | 255186.83184910292 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 8 | ok | 67.36637 | 0.054114999999999996 | 0.0572243 | 0.060015469999999994 | 293209.8821992647 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 64 | ok | 64.605951 | 0.054455 | 0.05601035 | 0.05756053999999999 | 292947.15086924745 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 1 | ok | 67.628424 | 0.060239 | 0.0617746 | 0.06651618 | 528284.3437651881 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 2 | ok | 67.334142 | 0.0720665 | 0.07604944999999999 | 0.07915025 | 447980.25303044636 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 4 | ok | 68.100001 | 0.077375 | 0.0793569 | 0.08341922999999998 | 419756.4677912992 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 8 | ok | 68.391353 | 0.06682550000000001 | 0.0697917 | 0.07420922999999999 | 477796.6407909924 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 64 | ok | 72.864011 | 0.09870899999999999 | 0.10174415 | 0.10382776999999999 | 323891.294793913 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 1 | ok | 68.169718 | 0.0717545 | 0.07469525 | 0.07921882 | 885254.9174527454 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 2 | ok | 68.99584 | 0.108263 | 0.1104821 | 0.1129587 | 589334.9888735396 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 4 | ok | 68.778366 | 0.094772 | 0.0980051 | 0.10171483 | 671045.7997146377 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 8 | ok | 68.089321 | 0.09781100000000001 | 0.1004582 | 0.10757823999999999 | 651668.913906187 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 64 | ok | 77.291437 | 0.13232149999999998 | 0.1373149 | 0.14196471 | 481225.73005702824 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 1 | ok | 68.830887 | 0.097788 | 0.10215705 | 0.10395746 | 1302336.901125972 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 2 | ok | 68.947132 | 0.160382 | 0.16460825 | 0.16702536 | 843563.5180921967 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 4 | ok | 69.463748 | 0.13263599999999998 | 0.13809675 | 0.14083577 | 962852.9815269134 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 8 | ok | 69.194443 | 0.109931 | 0.11422009999999999 | 0.11714342999999999 | 1161468.6408003971 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 64 | ok | 76.82409 | 0.15195399999999998 | 0.16211535 | 0.16787373 | 835624.445827577 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 1 | ok | 66.907915 | 0.104439 | 0.10706175 | 0.11072958999999999 | 9542.037541810823 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 2 | ok | 67.186712 | 0.10769 | 0.11242635 | 0.11462175 | 9252.03068195423 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 4 | ok | 66.740269 | 0.110232 | 0.1186469 | 0.12143658 | 8969.580028117838 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 8 | ok | 69.756849 | 0.118282 | 0.1283668 | 0.13126805 | 8472.469641870402 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 64 | ok | 65.436988 | 0.23531400000000002 | 0.2448969 | 0.24766239999999998 | 4353.046335827599 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 1 | ok | 66.602428 | 0.1476225 | 0.15626725 | 0.16177431999999997 | 13455.277617360967 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 2 | ok | 66.891405 | 0.1566125 | 0.16268265 | 0.16628429 | 12706.67728267198 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 4 | ok | 65.335984 | 0.1688265 | 0.18058969999999996 | 0.18517339 | 11756.696114447204 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 8 | ok | 66.760162 | 0.18925799999999998 | 0.20592555 | 0.20728641 | 10646.02530113081 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 64 | ok | 66.748788 | 0.37302199999999996 | 0.44949619999999996 | 0.4860836399999999 | 5197.644863949827 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 1 | ok | 66.876245 | 0.16494799999999998 | 0.1726787 | 0.17389635 | 24137.576461807836 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 2 | ok | 67.178587 | 0.18352049999999998 | 0.19073345 | 0.19213839 | 21689.656799138054 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 4 | ok | 67.008608 | 0.1840945 | 0.19613374999999997 | 0.20168738 | 21656.095784911657 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 8 | ok | 67.342682 | 0.219255 | 0.23569215 | 0.24410315 | 18201.205775278995 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 64 | ok | 65.47475 | 0.41412150000000003 | 0.50253375 | 0.5840547999999999 | 9050.554724337526 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 1 | ok | 67.420498 | 0.1781395 | 0.1856426 | 0.19774949 | 44500.2428044498 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 2 | ok | 67.374553 | 0.19177 | 0.203866 | 0.20657249 | 41025.245500197234 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 4 | ok | 67.52369 | 0.196913 | 0.2075445 | 0.21069353999999998 | 40429.24129778269 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 8 | ok | 67.517659 | 0.2352795 | 0.24723375 | 0.2515717 | 34427.20320544804 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 64 | ok | 64.996759 | 0.4391025 | 0.54465935 | 0.5612474900000001 | 18391.32293705565 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 1 | ok | 67.77674 | 0.188015 | 0.1957155 | 0.19658777 | 84445.72989611064 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 2 | ok | 67.065887 | 0.2067785 | 0.22670559999999998 | 0.22979163 | 75178.43837280406 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 4 | ok | 68.263947 | 0.2184685 | 0.22634885 | 0.23131462 | 74060.59915950478 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 8 | ok | 67.139521 | 0.242913 | 0.2642228 | 0.26722382 | 65840.09908934913 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 64 | ok | 65.228723 | 0.440048 | 0.5269962 | 0.7638043099999992 | 35578.59783678567 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 1 | ok | 67.720527 | 0.20823999999999998 | 0.21512409999999998 | 0.21765794 | 153199.95936371078 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 2 | ok | 67.858034 | 0.25839500000000004 | 0.26536709999999997 | 0.26865562 | 125939.27082421586 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 4 | ok | 68.866243 | 0.23929099999999998 | 0.26888185 | 0.27324367 | 131056.6929137728 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 8 | ok | 68.657295 | 0.25487550000000003 | 0.28387484999999996 | 0.2868928 | 124402.75984412647 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 64 | ok | 69.155233 | 0.41538699999999995 | 0.5707883 | 0.59104174 | 72979.68956994345 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 1 | ok | 67.957278 | 0.2502625 | 0.2577221 | 0.25953503 | 255175.21247124195 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 2 | ok | 68.573903 | 0.308125 | 0.37356675 | 0.38150504 | 192485.93845105724 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 4 | ok | 68.890904 | 0.28135750000000004 | 0.31658415 | 0.31983852 | 226792.30771520498 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 8 | ok | 68.651975 | 0.2906245 | 0.3305863 | 0.33344983 | 215365.84128156674 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 64 | ok | 73.972837 | 0.4081375 | 0.51419405 | 0.5190750000000001 | 153112.50204249684 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 1 | ok | 68.342004 | 0.346496 | 0.3525598 | 0.35625074 | 369149.6344005469 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 2 | ok | 69.737869 | 0.43678300000000003 | 0.56081865 | 0.5625184900000001 | 306418.74007221236 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 4 | ok | 69.277668 | 0.3711795 | 0.4340867499999999 | 0.44843438 | 357138.9509355813 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 8 | ok | 69.537135 | 0.35301400000000005 | 0.37333215000000003 | 0.37650325 | 375508.0065934512 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 64 | ok | 75.70333 | 0.5328135 | 0.6583327 | 0.6752616499999999 | 245792.05914985904 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 1 | ok | 1383.731296 | 0.0736685 | 0.079079 | 0.08292793999999999 | 13426.001385026304 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 2 | ok | 1376.275121 | 0.078729 | 0.08703315 | 0.08874051 | 12562.971896631865 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 4 | ok | 1397.671307 | 0.074583 | 0.08131115 | 0.08359177 | 13298.661064206732 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 8 | ok | 1395.000343 | 0.079475 | 0.0828858 | 0.08686545999999999 | 12536.35857395411 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 64 | ok | 1647.39199 | 0.081109 | 0.084026 | 0.087418 | 12316.359993408285 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 1 | ok | 1364.029547 | 0.0826965 | 0.08526575 | 0.08992717999999998 | 24071.050036973134 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 2 | ok | 1373.68136 | 0.081277 | 0.08287725 | 0.08863901999999999 | 24569.59596036531 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 4 | ok | 1316.302994 | 0.07758699999999999 | 0.07980575 | 0.08709481 | 25630.91785854706 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 8 | ok | 1447.574002 | 0.0779565 | 0.0798324 | 0.08416379999999998 | 25527.448133330883 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 64 | ok | 1638.906768 | 0.0786245 | 0.0810116 | 0.08597249999999998 | 25306.787862560857 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 1 | ok | 1306.328208 | 0.085698 | 0.08949304999999999 | 0.09418585 | 46342.134245284746 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 2 | ok | 1314.491398 | 0.081626 | 0.08374045 | 0.08498652999999999 | 48926.98235008038 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 4 | ok | 1398.543774 | 0.080987 | 0.0826284 | 0.08651824999999999 | 49152.71773978109 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 8 | ok | 1447.105322 | 0.079542 | 0.08217945 | 0.0854465 | 50058.76899479989 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 64 | ok | 1594.180492 | 0.08206150000000001 | 0.08392445 | 0.08821184999999998 | 48646.389878437534 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 1 | ok | 1381.429663 | 0.0853785 | 0.0878153 | 0.09380730999999998 | 93233.26917330526 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 2 | ok | 1367.203948 | 0.0875225 | 0.08901305 | 0.09403231 | 91140.65030221101 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 4 | ok | 1404.223772 | 0.0826475 | 0.0841393 | 0.09049336999999998 | 96407.28607705256 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 8 | ok | 1341.833388 | 0.0815195 | 0.08446215 | 0.09020124999999998 | 97474.98365466368 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 64 | ok | 1578.28852 | 0.0862515 | 0.088338 | 0.09183463999999998 | 92429.36490165861 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 1 | ok | 1372.699452 | 0.085181 | 0.08742605 | 0.09169775999999999 | 187228.3579815566 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 2 | ok | 1347.279103 | 0.09057899999999999 | 0.0970185 | 0.1006463 | 175256.7675948484 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 4 | ok | 1395.62248 | 0.10155549999999999 | 0.10371514999999999 | 0.10983093999999997 | 156845.09233666642 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 8 | ok | 1408.1696 | 0.08561099999999999 | 0.08759065 | 0.09231524999999999 | 186047.37696454403 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 64 | ok | 1631.995272 | 0.08847150000000001 | 0.08983175 | 0.09775421999999999 | 180216.71059448985 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 1 | ok | 1384.171695 | 0.0906045 | 0.09340675 | 0.10065664999999997 | 350927.13853353256 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 2 | ok | 1373.341122 | 0.116686 | 0.12215564999999999 | 0.12602873999999997 | 272690.3975485133 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 4 | ok | 1396.603268 | 0.1310545 | 0.13498585 | 0.14324392 | 243058.18235622425 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 8 | ok | 1419.53782 | 0.1051645 | 0.1082715 | 0.11424620999999999 | 302813.2868414 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 64 | ok | 1625.935144 | 0.1466365 | 0.15264495 | 0.16120948999999998 | 217499.31521699973 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 1 | ok | 1375.485617 | 0.1117805 | 0.11641625 | 0.12326524999999999 | 569005.2188447417 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 2 | ok | 1326.793648 | 0.158937 | 0.1627845 | 0.16791037 | 401685.4722415255 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 4 | ok | 1376.057279 | 0.150439 | 0.15425305 | 0.16120126999999998 | 423936.74015963334 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 8 | ok | 1424.985993 | 0.14347100000000002 | 0.1483998 | 0.15382348 | 444259.6447727647 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 64 | ok | 1542.006768 | 0.196521 | 0.20985944999999998 | 0.23244813999999994 | 325295.84895049897 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 1 | ok | 1355.643253 | 0.1312755 | 0.1347409 | 0.13921308 | 971902.894752028 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 2 | ok | 1364.613577 | 0.2198695 | 0.22641804999999998 | 0.22943513 | 580015.7039251837 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 4 | ok | 1327.541791 | 0.215422 | 0.21952895 | 0.22183809 | 594209.1716557025 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 8 | ok | 1328.635974 | 0.18024800000000002 | 0.184958 | 0.18979009 | 708046.6067253585 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 64 | ok | 1570.310843 | 0.205564 | 0.2184185 | 0.2188688 | 621735.0410063403 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 1 | ok | 1381.572801 | 0.14284550000000001 | 0.14672215 | 0.1543033 | 6975.351062443621 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 2 | ok | 1367.94099 | 0.15073150000000002 | 0.15701849999999998 | 0.15848236 | 6601.501049308592 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 4 | ok | 1413.563562 | 0.15288049999999997 | 0.16239355 | 0.16487965999999998 | 6517.1230238616645 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 8 | ok | 1376.390669 | 0.164312 | 0.1747329 | 0.17954816999999998 | 6048.1029822591 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 64 | ok | 1575.856016 | 0.27302550000000003 | 0.2857151 | 0.28820878 | 3673.608145182757 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 1 | ok | 1372.519814 | 0.196908 | 0.20334249999999998 | 0.21069196 | 10106.766874586443 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 2 | ok | 1370.639924 | 0.22160600000000003 | 0.23162675 | 0.23487963 | 9006.92443343518 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 4 | ok | 1384.099211 | 0.2245605 | 0.2427761 | 0.25061594 | 8778.36960570285 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 8 | ok | 1334.177382 | 0.2822855 | 0.40515645 | 0.40956006 | 6253.600119418747 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 64 | ok | 1567.728737 | 0.454311 | 0.55475685 | 0.7414094699999993 | 4334.275258032409 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 1 | ok | 1312.245617 | 0.2245325 | 0.23477465 | 0.23696527 | 17727.30930999285 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 2 | ok | 1329.36696 | 0.262042 | 0.26752755 | 0.27226817000000003 | 15299.785137467421 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 4 | ok | 1431.473203 | 0.246048 | 0.2588977 | 0.26212735 | 16141.995321726916 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 8 | ok | 1355.727561 | 0.29115 | 0.3214225 | 0.32571107 | 13701.068176378236 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 64 | ok | 1546.318132 | 0.495225 | 0.6036069 | 0.6591257399999998 | 7775.339344966536 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 1 | ok | 1314.198462 | 0.22992649999999998 | 0.2352246 | 0.23834751 | 34748.5309624179 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 2 | ok | 1394.14504 | 0.26474600000000004 | 0.2825918 | 0.2883869 | 29690.99762271605 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 4 | ok | 1383.499143 | 0.2764675 | 0.29023875 | 0.30131177 | 28791.28084850784 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 8 | ok | 1419.654766 | 0.334701 | 0.4801416 | 0.49340487999999993 | 21290.28947767894 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 64 | ok | 1644.018384 | 0.49051049999999996 | 0.5801726 | 0.58458391 | 15751.401923994366 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 1 | ok | 1362.407946 | 0.2543515 | 0.26687415 | 0.27539525 | 62449.298975394355 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 2 | ok | 1375.75912 | 0.28880300000000003 | 0.29594415 | 0.30229813 | 56077.772018902135 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 4 | ok | 1326.473203 | 0.277208 | 0.28588355 | 0.29132705 | 57985.85551026538 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 8 | ok | 1385.468336 | 0.30023 | 0.32175985 | 0.32418133 | 53110.33366434349 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 64 | ok | 1507.415482 | 0.5168980000000001 | 0.6240162 | 0.6367590399999999 | 30844.53414034973 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 1 | ok | 1393.307601 | 0.26409550000000004 | 0.27471275 | 0.27615335 | 120617.44067883496 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 2 | ok | 1381.46381 | 0.3324825 | 0.340976 | 0.34450415 | 99076.21338638535 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 4 | ok | 1320.281699 | 0.31465350000000003 | 0.429023 | 0.43408038 | 90006.11985361179 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 8 | ok | 1409.061793 | 0.31514050000000005 | 0.33571305 | 0.33941166 | 101502.50993503394 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 64 | ok | 1551.258615 | 0.5017755 | 0.60805095 | 0.7068261499999999 | 60707.40057360905 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 1 | ok | 1364.13759 | 0.3313825 | 0.3413857 | 0.3438835 | 192623.59964147932 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 2 | ok | 1370.857394 | 0.37269399999999997 | 0.4277394 | 0.43486219 | 171674.75916445468 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 4 | ok | 1314.684585 | 0.330213 | 0.4226102 | 0.4986361499999999 | 178306.088813037 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 8 | ok | 1396.872101 | 0.35276549999999995 | 0.3790324 | 0.38155458 | 180766.40890129952 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 64 | ok | 1625.273319 | 0.5142279999999999 | 0.6248745 | 0.62951914 | 122005.45707533475 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 1 | ok | 1369.31681 | 0.41784299999999996 | 0.42936625 | 0.43064051 | 305554.8486947293 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 2 | ok | 1326.398812 | 0.44162500000000005 | 0.531875 | 0.53905934 | 275954.025886816 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 4 | ok | 1385.148703 | 0.4069715 | 0.44888659999999997 | 0.45126248 | 309020.4717371325 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 8 | ok | 1421.974559 | 0.4055145 | 0.45709425 | 0.4678447 | 309297.36771991925 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 64 | ok | 1553.024005 | 0.6021000000000001 | 0.9123467 | 3.53819662999999 | 175721.649000509 | - |
