# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

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

## Run References

### full_mlp_capacity_search_hd64_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,960`
- MACs / sample: `10,752`
- FLOPs / sample estimate: `21,784`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
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
