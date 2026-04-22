# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd512_depth4

- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`268939.113` samples/s, p50=`0.470` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.118` ms, throughput=`8533.868` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `349,328`
- MACs / sample: `348,160`
- FLOPs / sample estimate: `697,560`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 1 | 1 | ok | 0.152935 | 0.1590956 | 0.16949313 | 6489.921865234696 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 1 | 2 | ok | 0.1673095 | 0.17534160000000001 | 0.18617094999999997 | 5937.827619638914 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 1 | 4 | ok | 0.12239749999999999 | 0.129403 | 0.13683565 | 8098.548319016719 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 1 | 8 | ok | 0.117783 | 0.1290694 | 0.13255026 | 8533.86784930145 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 1 | 64 | ok | 0.12944 | 0.1499644 | 0.15206464 | 7405.886731998437 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 2 | 1 | ok | 0.1726435 | 0.17760375 | 0.1884506 | 11518.55304858088 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 2 | 2 | ok | 0.16545549999999998 | 0.17108085 | 0.17926535 | 12013.5185721789 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 2 | 4 | ok | 0.1350325 | 0.14582889999999998 | 0.15497966 | 14751.053114559481 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 2 | 8 | ok | 0.1335995 | 0.13853505 | 0.14255709 | 14905.17773082978 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 2 | 64 | ok | 0.146768 | 0.15239685 | 0.15550943999999997 | 13603.404442409375 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 4 | 1 | ok | 0.2038565 | 0.21620419999999999 | 0.22135294 | 19478.137835484144 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 4 | 2 | ok | 0.1940395 | 0.21405159999999998 | 0.23026137 | 19928.961224818016 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 4 | 4 | ok | 0.1649815 | 0.1752606 | 0.18538772999999997 | 24049.945965783903 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 4 | 8 | ok | 0.1379485 | 0.14489529999999998 | 0.15422879999999997 | 28791.299499621615 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 4 | 64 | ok | 0.1468905 | 0.15286415 | 0.15926223 | 27093.32879610387 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 8 | 1 | ok | 0.242301 | 0.25570745 | 0.26256913 | 32759.92007234701 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 8 | 2 | ok | 0.262841 | 0.27423375 | 0.27817297999999996 | 30776.380359129584 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 8 | 4 | ok | 0.19633299999999998 | 0.20370685 | 0.22199011 | 40471.488797340135 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 8 | 8 | ok | 0.158469 | 0.1639163 | 0.17203056 | 50195.71938434448 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 8 | 64 | ok | 0.138164 | 0.1434086 | 0.15418625 | 57718.20041000123 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 16 | 1 | ok | 0.35290200000000005 | 0.3669577 | 0.36805438999999995 | 45133.181813719784 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 16 | 2 | ok | 0.531312 | 0.7168161 | 0.7216088700000001 | 31889.88422975466 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 16 | 4 | ok | 0.4145225 | 0.56723825 | 0.57444784 | 38352.9259255424 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 16 | 8 | ok | 0.289613 | 0.35692395 | 0.36752969 | 52538.49939144 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 16 | 64 | ok | 0.19738899999999998 | 0.2033197 | 0.20742576 | 80569.40817841934 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 32 | 1 | ok | 0.422275 | 0.43347225 | 0.43968064999999995 | 75556.0535958755 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 32 | 2 | ok | 0.45708899999999997 | 0.6629179499999998 | 0.8300311 | 61216.9782228252 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 32 | 4 | ok | 0.451392 | 0.61457095 | 0.6279498699999999 | 70982.83093903807 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 32 | 8 | ok | 0.343613 | 0.42161709999999997 | 0.42916318 | 89697.66784381776 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 32 | 64 | ok | 0.296464 | 0.3027854 | 0.3133565 | 114844.18515179529 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 64 | 1 | ok | 0.5770390000000001 | 0.58637645 | 0.5954856 | 110635.94509303998 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 64 | 2 | ok | 0.518578 | 0.6801018999999999 | 0.6850596899999999 | 128461.1549926388 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 64 | 4 | ok | 0.49112 | 0.6396621 | 0.64969488 | 131196.0243668736 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 64 | 8 | ok | 0.4125885 | 0.49551219999999996 | 0.49887153 | 151369.17916325672 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 64 | 64 | ok | 0.2963955 | 0.3052655 | 0.33393613999999994 | 221004.38065214388 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 128 | 1 | ok | 0.8744890000000001 | 0.8891531500000001 | 0.894221 | 146352.7522092405 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 128 | 2 | ok | 0.6075865 | 0.85490605 | 0.8625001999999999 | 200463.50294122245 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 128 | 4 | ok | 0.49628150000000004 | 0.6270884 | 0.7362703499999996 | 240027.65718679933 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 128 | 8 | ok | 0.4698725 | 0.6011910500000001 | 0.60797153 | 268939.1125210991 | - |
| `full_mlp_capacity_search_hd512_depth4` | `fp32` | 128 | 64 | ok | 0.485433 | 0.6702301499999999 | 0.69977282 | 250225.25159855816 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 1 | 1 | ok | 0.364396 | 0.4803613999999998 | 0.5106494199999999 | 2679.9920157677866 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 1 | 2 | ok | 0.367494 | 0.4123433 | 0.42171672 | 2674.885054839424 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 1 | 4 | ok | 0.31872100000000003 | 0.33438585 | 0.34516391999999996 | 3150.9899559674363 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 1 | 8 | ok | 0.3063015 | 0.33581700000000003 | 0.34823277999999996 | 3226.662037482197 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 1 | 64 | ok | 0.4902455 | 0.5934396 | 0.63629212 | 1978.5686978639726 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 2 | 1 | ok | 0.4243575 | 0.4371879 | 0.4391184 | 4695.949921638683 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 2 | 2 | ok | 0.4042775 | 0.4646894 | 0.47082283999999996 | 4850.839357712002 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 2 | 4 | ok | 0.3426715 | 0.37099985 | 0.37965598 | 5869.060899195299 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 2 | 8 | ok | 0.32671649999999997 | 0.36096839999999997 | 0.37502136999999997 | 6100.821194936123 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 2 | 64 | ok | 0.473514 | 0.64329985 | 0.64623286 | 3920.040272925748 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 4 | 1 | ok | 0.4504085 | 0.46108055000000003 | 0.4632395 | 8854.090346252484 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 4 | 2 | ok | 0.5154365000000001 | 0.65230785 | 0.65769332 | 8031.61307157878 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 4 | 4 | ok | 0.3676025 | 0.43316455 | 0.43728137 | 10416.701931542999 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 4 | 8 | ok | 0.3476865 | 0.39427134999999996 | 0.41088180999999996 | 11465.289838227627 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 4 | 64 | ok | 0.5264139999999999 | 0.62359835 | 0.6247231999999999 | 7752.52393107231 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 8 | 1 | ok | 0.5401545 | 0.5521639 | 0.5886506199999999 | 14741.95570174989 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 8 | 2 | ok | 0.550552 | 0.67807805 | 0.68430376 | 14934.436888656495 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 8 | 4 | ok | 0.430812 | 0.5217531 | 0.61725451 | 17288.16092477139 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 8 | 8 | ok | 0.39078650000000004 | 0.42825965 | 0.44695501 | 20399.674502793634 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 8 | 64 | ok | 0.49113300000000004 | 0.58817115 | 0.59233138 | 15653.765016167603 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 16 | 1 | ok | 0.71892 | 0.73386765 | 0.73804567 | 22187.448327512917 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 16 | 2 | ok | 0.5544015 | 0.8278636500000001 | 0.8366514799999999 | 25622.79567889487 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 16 | 4 | ok | 0.51857 | 0.6394309499999999 | 0.64731459 | 30955.42886444382 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 16 | 8 | ok | 0.516259 | 0.64596765 | 0.66807889 | 31878.794273867217 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 16 | 64 | ok | 0.509835 | 0.61272655 | 0.7026927799999998 | 30523.30799860615 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 32 | 1 | ok | 1.032408 | 1.0437779 | 1.04568914 | 30935.79324449282 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 32 | 2 | ok | 0.706791 | 1.0723836 | 1.07592535 | 42141.077528915695 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 32 | 4 | ok | 0.5733090000000001 | 0.8532888 | 0.86212468 | 51329.83573971346 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 32 | 8 | ok | 0.4971135 | 0.72266255 | 0.73016832 | 59407.29343341482 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 32 | 64 | ok | 0.542395 | 0.6644718 | 0.6691308399999999 | 58310.610417336335 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 64 | 1 | ok | 1.6914105 | 1.7007276 | 1.70415033 | 37810.00750079655 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 64 | 2 | ok | 1.0391815 | 1.0506688 | 1.1752069599999995 | 61166.95930596604 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 64 | 4 | ok | 0.7104905 | 1.0283449 | 1.03261115 | 83939.72791776592 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 64 | 8 | ok | 0.5856129999999999 | 0.8774724 | 0.9500700599999999 | 99657.19482125159 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 64 | 64 | ok | 0.602676 | 0.7448355 | 0.74978575 | 106451.08547672871 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 128 | 1 | ok | 3.0179405 | 3.0256519 | 3.03873368 | 42412.927651206264 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 128 | 2 | ok | 1.7065614999999998 | 1.71587455 | 1.7948481999999997 | 74878.4392413003 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 128 | 4 | ok | 1.042754 | 1.2033459499999999 | 1.20788394 | 121147.2477787179 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 128 | 8 | ok | 0.74064 | 0.9775375 | 0.98418796 | 164343.0232290903 | - |
| `full_mlp_capacity_search_hd512_depth4` | `bf16` | 128 | 64 | ok | 0.7838624999999999 | 0.9251365 | 3.1247150099999916 | 142884.29725238407 | - |
