# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

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

## Run References

### full_mlp_capacity_search_hd256_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `43,408`
- MACs / sample: `43,008`
- FLOPs / sample estimate: `86,488`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
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
