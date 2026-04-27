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

### full_mlp_capacity_search_hd128_depth3::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1634337.847` samples/s, p50=`0.077` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.035` ms, throughput=`28198.817` samples/s

### full_mlp_capacity_search_hd128_depth3::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2258390.981` samples/s, p50=`0.056` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.021` ms, throughput=`48542.605` samples/s

### full_mlp_capacity_search_hd128_depth3::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1672481.152` samples/s, p50=`0.076` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.034` ms, throughput=`28755.546` samples/s

### full_mlp_capacity_search_hd128_depth3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2580109.372` samples/s, p50=`0.049` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.011` ms, throughput=`88449.707` samples/s

### full_mlp_capacity_search_hd128_depth3::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`504164.676` samples/s, p50=`0.240` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.207` ms, throughput=`4130.100` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `21,776`
- MACs / sample: `21,504`
- FLOPs / sample estimate: `43,352`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.035143 | 0.03716445 | 0.04104622999999999 | 28198.817454391232 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0382865 | 0.040004649999999996 | 0.04414289 | 25967.146366417208 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.039181999999999995 | 0.0411478 | 0.04671148 | 25329.011190863723 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.046516 | 0.050919599999999995 | 0.05790236999999998 | 21162.83873239675 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.036482 | 0.04145779999999999 | 0.044485409999999996 | 54015.57160898344 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.037299 | 0.03933005 | 0.04698776999999999 | 52588.95421605647 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0370205 | 0.0410933 | 0.04325917 | 52826.92736403141 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037311 | 0.04051175 | 0.046590379999999994 | 52733.82706257818 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.038715 | 0.042372349999999996 | 0.09610950999999981 | 97121.65409831528 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.038382 | 0.04312839999999999 | 0.04588722 | 102719.49872884619 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.038322999999999996 | 0.041390899999999994 | 0.04721879 | 103079.50006442469 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.038847 | 0.043409199999999995 | 0.047191409999999996 | 101144.09138975522 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0407 | 0.042426900000000003 | 0.04850948 | 195005.32852060182 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.040899500000000005 | 0.04479245 | 0.0968639899999998 | 182983.2025994594 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.04816 | 0.05454929999999999 | 0.05765407 | 163451.30703837672 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.040966 | 0.0443996 | 0.04834545999999999 | 192646.21245505926 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0420485 | 0.046805849999999996 | 0.10846815999999977 | 355968.3205993083 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.053781 | 0.05699624999999999 | 0.058525890000000004 | 295851.861056132 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0550175 | 0.0605437 | 0.06342263999999999 | 286746.77893758885 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.064995 | 0.0721719 | 0.07394983 | 243876.34128177137 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.048359 | 0.05312735 | 0.057529119999999996 | 652342.5825998216 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0633155 | 0.0701377 | 0.0730486 | 500383.8882642777 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.065841 | 0.07424774999999999 | 0.1035127899999999 | 473619.1190565981 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0846755 | 0.08934819999999999 | 0.09328683999999998 | 377122.43326936394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.057986 | 0.0647846 | 0.08742157999999993 | 1072939.068460889 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.101942 | 0.10772654999999999 | 0.12566199999999997 | 620617.2542845379 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.094728 | 0.10157395 | 0.10268002 | 674721.1503995615 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0784445 | 0.08472575 | 0.08921316 | 812367.0700915385 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0766735 | 0.08463665 | 0.08833779 | 1634337.8467445648 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.12364649999999999 | 0.13132465 | 0.13316753 | 1068702.9019291424 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.129971 | 0.15127875 | 0.15789826999999998 | 952867.0280017843 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1673075 | 0.17893155 | 0.18587221999999998 | 758777.5145768274 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0485435 | 0.053984599999999994 | 0.057115990000000005 | 20256.742051456986 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0508525 | 0.054586949999999995 | 0.057770459999999996 | 19557.721681885832 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.053774 | 0.06039484999999999 | 0.06759583999999998 | 18252.34524384038 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0616315 | 0.06938589999999999 | 0.08195935 | 15840.053476020536 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0690505 | 0.07470265 | 0.07835547 | 28813.58618215661 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0756155 | 0.0866286 | 0.08775349 | 26046.353653743423 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0801105 | 0.08840585 | 0.09336148999999999 | 24652.79618177493 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.09011949999999999 | 0.0995358 | 0.10422471 | 21946.48656501932 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0621835 | 0.06880435 | 0.1278399299999998 | 61047.17267650646 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.069122 | 0.07424019999999999 | 0.07657086999999999 | 57375.01283765912 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.08852099999999999 | 0.09538935 | 0.10303798999999998 | 44650.74078927775 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.08570050000000001 | 0.0963443 | 0.11607095999999996 | 45688.011191735226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.06659699999999999 | 0.07537964999999999 | 0.12377239999999981 | 115442.48092818064 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.075512 | 0.0837949 | 0.08525608 | 105050.29282769126 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.082708 | 0.09086744999999999 | 0.09471356999999998 | 95329.27087645493 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.086719 | 0.09239715 | 0.09297194 | 91830.77974892547 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.06615399999999999 | 0.07327325 | 0.07477423999999999 | 238550.96984387733 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.080962 | 0.08686775 | 0.08838528 | 197105.2142706146 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.106177 | 0.1152881 | 0.11838338999999999 | 149172.21540691663 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.0923935 | 0.0965851 | 0.10006632 | 173113.97196375945 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.072403 | 0.07855595 | 0.1321723499999998 | 426201.0278370537 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0886915 | 0.0961165 | 0.09782077 | 358542.7746011884 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0986535 | 0.10574795 | 0.11112650999999998 | 324297.3944933897 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.106855 | 0.11769334999999999 | 0.12370241 | 298266.90152699605 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.080772 | 0.08741334999999999 | 0.10866489999999994 | 774252.8037024769 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1143635 | 0.1219476 | 0.12505461 | 559516.8571938132 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.13135 | 0.1440628 | 0.14822644000000001 | 484830.1200712094 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.157195 | 0.17512824999999999 | 0.17708393 | 407579.39742188196 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1067535 | 0.1141922 | 0.11544708 | 1187085.8415158791 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.141521 | 0.15702254999999998 | 0.16235195 | 898999.7644339681 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.15680650000000002 | 0.1742816 | 0.17603533000000002 | 813800.6346119073 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.236246 | 0.2807388 | 0.29295192 | 524134.1242844853 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 45.131704 | 0.0205235 | 0.0210992 | 0.02272902 | 48542.6053592978 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 47.14542 | 0.033229999999999996 | 0.0373391 | 0.03887868 | 29781.88344204713 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 46.881519 | 0.0248195 | 0.02720885 | 0.03222658 | 39706.237373416516 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 47.409858 | 0.0295265 | 0.03212195 | 0.03506859 | 33270.717842354024 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 46.218147 | 0.022001 | 0.0225028 | 0.024121209999999997 | 90864.56727112805 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 44.948026 | 0.0218165 | 0.0224955 | 0.030389409999999995 | 90450.99771973034 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 44.958326 | 0.022322 | 0.0227101 | 0.027747879999999996 | 88858.47954255655 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 46.520595 | 0.0221695 | 0.022742099999999998 | 0.02494799999999999 | 91288.43585481122 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 46.547405 | 0.022437 | 0.022812950000000002 | 0.025081009999999994 | 179291.154491602 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 45.183429 | 0.022598 | 0.024047199999999994 | 0.032304259999999994 | 175202.53412945365 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 45.369094 | 0.022543 | 0.0232992 | 0.029508699999999992 | 177237.71476780088 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 46.489483 | 0.022373 | 0.02759845 | 0.027940049999999998 | 168660.79951965404 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 46.346769 | 0.023345 | 0.028890449999999998 | 0.02945722 | 323496.0869104587 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 46.473295 | 0.0240175 | 0.0246589 | 0.02730997999999999 | 333451.98666525504 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 47.484293 | 0.0298575 | 0.03598554999999999 | 0.03839063 | 264241.8076782063 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 45.148545 | 0.023365999999999998 | 0.029850399999999996 | 0.03509177 | 323795.9243806998 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 45.295458 | 0.0260875 | 0.0267351 | 0.03336012999999999 | 609847.6676757105 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 45.672309 | 0.032098 | 0.03640685 | 0.03867751 | 495909.9824199911 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 46.07693 | 0.0326985 | 0.034429249999999995 | 0.03764638999999999 | 487499.89335939835 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 47.255249 | 0.047263 | 0.05059324999999999 | 0.05364417999999999 | 336504.75039549824 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 46.216701 | 0.030947500000000003 | 0.0378188 | 0.038462130000000004 | 1008872.4022323823 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 46.121587 | 0.0397555 | 0.0431536 | 0.04441225 | 800280.8985954069 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 46.010183 | 0.0422235 | 0.04362725 | 0.04862196999999999 | 754928.859750732 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 47.973274 | 0.064985 | 0.06853545 | 0.07088414999999999 | 496868.94926191674 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 46.066353 | 0.038041000000000005 | 0.03955865 | 0.04141432999999999 | 1670556.801802531 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 47.797237 | 0.06536600000000001 | 0.0697526 | 0.07192238 | 976074.5818587998 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 46.026912 | 0.0610615 | 0.06501345 | 0.06558664 | 1046908.7075977437 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 46.767356 | 0.0951525 | 0.10181309999999999 | 0.10341662 | 681450.0148960715 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 46.716233 | 0.056370500000000004 | 0.059343099999999996 | 0.060886289999999996 | 2258390.981115617 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 48.615532 | 0.10038050000000001 | 0.10497825 | 0.10766887 | 1290466.457248157 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 48.841573 | 0.11672199999999999 | 0.12339175 | 0.12650024999999998 | 1102243.0301212019 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 50.024681 | 0.1425885 | 0.1603546 | 0.16491838 | 879692.0747853722 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 45.40981 | 0.0310905 | 0.03436754999999999 | 0.03735176 | 31969.718282842492 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 46.172877 | 0.033119 | 0.0357385 | 0.039477649999999996 | 29847.948580342018 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 48.133147 | 0.0381885 | 0.04275334999999999 | 0.04504936 | 25949.77473000557 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 47.879196 | 0.0384235 | 0.04230724999999999 | 0.04540907999999999 | 25804.95991973625 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 47.391603 | 0.043915499999999996 | 0.04744605 | 0.05350845999999999 | 45072.06798310158 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 49.103068 | 0.048303 | 0.052222399999999995 | 0.06634794999999996 | 40608.708293841366 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 48.860594 | 0.0566295 | 0.060408899999999995 | 0.06599875 | 35047.66657893067 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 48.074227 | 0.060826 | 0.06824714999999999 | 0.06937114 | 32287.30014564801 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 47.708589 | 0.047942 | 0.05219964999999999 | 0.05752076 | 83046.61489540694 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 47.75757 | 0.0546315 | 0.0594872 | 0.06179489 | 73591.58583244226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 46.932946 | 0.0555795 | 0.06291859999999999 | 0.06566235 | 70705.55660828273 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 48.800361 | 0.06346750000000001 | 0.07135354999999999 | 0.07550149999999999 | 62449.76696869456 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 47.59461 | 0.050144999999999995 | 0.0548275 | 0.05579744 | 159054.57957898252 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 48.060266 | 0.058529 | 0.063998 | 0.06768537999999999 | 137925.7552555747 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 48.284464 | 0.060135 | 0.0646934 | 0.06667488999999999 | 132883.1694496278 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 49.399878 | 0.065796 | 0.07303005 | 0.07871080999999999 | 119956.57571958953 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 46.439182 | 0.052712499999999995 | 0.05746169999999999 | 0.060578219999999995 | 304656.29058116995 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 46.968126 | 0.058887999999999996 | 0.0679222 | 0.06921049 | 265550.73396563175 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 47.612236 | 0.080314 | 0.0885374 | 0.08943112 | 197639.63919895666 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 49.275477 | 0.0744615 | 0.08445395 | 0.10593284999999991 | 210669.5710978202 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 46.861422 | 0.056957999999999995 | 0.062214099999999994 | 0.06419321 | 562829.93706506 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 49.01308 | 0.07091800000000001 | 0.07923865 | 0.08140839 | 449590.3810137982 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 47.709589 | 0.0766105 | 0.08498399999999999 | 0.08972994999999999 | 412880.6372399755 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 50.128367 | 0.079036 | 0.0872006 | 0.09049176 | 401733.1773604272 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 47.637062 | 0.065872 | 0.06850985 | 0.07172723 | 969020.1238226028 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 49.703443 | 0.08919099999999999 | 0.0956876 | 0.09717648999999999 | 716228.8046633657 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 49.966366 | 0.10025500000000001 | 0.10888935 | 0.11070461999999999 | 633311.0948781163 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 49.589243 | 0.144584 | 0.1549708 | 0.16314478999999998 | 442942.99074553675 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 48.330636 | 0.08705550000000001 | 0.091642 | 0.09257957 | 1461385.402495224 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 50.087446 | 0.1207975 | 0.13137379999999999 | 0.13354781999999998 | 1056521.942970266 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 49.910023 | 0.21941850000000002 | 0.2331558 | 0.23439352 | 594634.8881394219 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 51.435055 | 0.205023 | 0.23318405 | 0.24219955999999998 | 618720.5304136426 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 492.102978 | 0.033949999999999994 | 0.039452949999999994 | 0.043487889999999994 | 28755.546225978334 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 502.202653 | 0.0383025 | 0.0410684 | 0.045665889999999994 | 25863.478079150518 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 504.582667 | 0.0364965 | 0.0390623 | 0.045747069999999994 | 27101.137651556335 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 530.628801 | 0.039865 | 0.04247065 | 0.048957449999999986 | 24763.128296281913 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.273006 | 0.0342595 | 0.0392377 | 0.03947995 | 57141.68165683449 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 510.130431 | 0.035366499999999995 | 0.04100205 | 0.04128618 | 55093.95999407189 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 510.030566 | 0.035570000000000004 | 0.042222499999999996 | 0.04261502 | 54072.49715984208 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 539.158717 | 0.0346945 | 0.038322699999999994 | 0.04031847 | 57007.57003522498 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 491.097448 | 0.0359095 | 0.04179694999999999 | 0.04418208999999999 | 109385.97278039454 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 509.505362 | 0.0367045 | 0.04227095 | 0.04351671 | 105171.60324643705 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 513.026003 | 0.035689 | 0.04149645 | 0.043866449999999994 | 108914.96192060644 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 535.463848 | 0.035696 | 0.04167985 | 0.0422183 | 109020.5809052633 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 498.546353 | 0.037152 | 0.0426814 | 0.04290232 | 212264.31364599516 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 501.37226 | 0.037373 | 0.043179249999999995 | 0.04573049 | 210043.32668721242 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 514.986223 | 0.046942 | 0.05107295 | 0.06960369999999995 | 166051.239261155 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 534.654537 | 0.0405785 | 0.04604485 | 0.04943397 | 193730.11844659442 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.473761 | 0.0403515 | 0.046863 | 0.04815507 | 382853.3389357538 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 502.71529 | 0.05447 | 0.058851099999999996 | 0.06282027 | 289978.73368461843 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 500.531868 | 0.050351 | 0.0547479 | 0.05840446999999999 | 313387.64602395304 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 549.231271 | 0.046773999999999996 | 0.05140315 | 0.05723077999999999 | 337946.5858523731 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 492.195964 | 0.044552 | 0.05014009999999999 | 0.05371159 | 711192.0289597394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 500.780475 | 0.06386800000000001 | 0.0704702 | 0.07391336 | 493996.55252155906 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 508.727317 | 0.058415499999999995 | 0.0609622 | 0.06382474 | 547718.9900335683 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 541.266205 | 0.055391499999999996 | 0.059689400000000004 | 0.06249493999999999 | 572497.9157497755 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 494.880871 | 0.058341500000000004 | 0.06086484999999999 | 0.06465872999999998 | 1099544.8914972537 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 509.923372 | 0.118566 | 0.1240863 | 0.12752328 | 540476.1764012901 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 508.32215 | 0.08519299999999999 | 0.09211164999999999 | 0.09409744 | 751246.0119401164 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 548.254488 | 0.067514 | 0.0714265 | 0.08222720999999998 | 948634.6923414646 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 491.348856 | 0.07631299999999999 | 0.0801662 | 0.08079201999999999 | 1672481.1519213931 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 500.856494 | 0.163646 | 0.17109755 | 0.17531909999999998 | 844145.4293745726 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 506.048584 | 0.150957 | 0.15769715 | 0.16388399 | 889790.1721841304 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 543.858817 | 0.10062 | 0.10829585 | 0.11151232999999999 | 1255560.5145836298 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.709992 | 0.049889 | 0.057024349999999994 | 0.059162629999999994 | 19764.408253616886 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 502.932206 | 0.0561565 | 0.06034575 | 0.06414501 | 17734.27599286231 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 515.883174 | 0.053677 | 0.06197359999999999 | 0.06652955 | 18283.447556124698 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 513.603008 | 0.0616595 | 0.06854284999999999 | 0.07150142999999999 | 16042.217984224724 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.855195 | 0.072229 | 0.07676495 | 0.08003495 | 27536.299727115267 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 515.052027 | 0.07122049999999999 | 0.07890055 | 0.08240390999999998 | 27510.376914172022 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 536.434745 | 0.0814755 | 0.09119354999999998 | 0.09454962 | 24291.46650498267 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 506.625211 | 0.069571 | 0.08087849999999999 | 0.08307335 | 28066.771973195115 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 531.406041 | 0.0616085 | 0.073388 | 0.0754031 | 63449.84416718272 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 510.085544 | 0.0813605 | 0.08940665 | 0.09639736999999998 | 48399.758775602255 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 540.342267 | 0.080885 | 0.09002355 | 0.09229521 | 48895.27243834612 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 503.978418 | 0.072052 | 0.07972045 | 0.08617155999999998 | 54918.95335458696 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 493.064525 | 0.06642100000000001 | 0.07075675 | 0.07502073 | 121160.86649405281 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 503.282344 | 0.078223 | 0.08599744999999999 | 0.08724066 | 101086.9883861159 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 505.363033 | 0.07468749999999999 | 0.08399495 | 0.08767248 | 104822.87423797046 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 502.684226 | 0.082167 | 0.09025045 | 0.09369009 | 96397.17951492457 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 494.932467 | 0.07257050000000001 | 0.08360999999999999 | 0.08449546999999999 | 214888.78430968052 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 505.276525 | 0.0778615 | 0.0901739 | 0.09180485 | 201268.89977865454 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 503.924753 | 0.07777200000000001 | 0.08889949999999999 | 0.09241244 | 201579.62836275765 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 506.630719 | 0.08384449999999999 | 0.09361059999999999 | 0.09633111999999999 | 187710.27950764532 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 495.510755 | 0.07678950000000001 | 0.0851667 | 0.08714289 | 412936.7935716065 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 505.255754 | 0.08164550000000001 | 0.09055115 | 0.09227045 | 385261.1601128526 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 507.431416 | 0.098748 | 0.19446584999999997 | 0.19753932999999999 | 247880.1214550625 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 511.208595 | 0.086685 | 0.09703255 | 0.10014844 | 364184.4612505457 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 490.701625 | 0.0693255 | 0.0792134 | 0.08521811999999998 | 911996.3428946651 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 503.667871 | 0.110026 | 0.11751160000000001 | 0.12212229999999999 | 581598.7604676418 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 501.345557 | 0.1020625 | 0.11008174999999999 | 0.11359129999999999 | 624273.5504270616 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 510.408184 | 0.10756399999999999 | 0.1160836 | 0.12053963 | 595906.93870315 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 494.209968 | 0.083963 | 0.09095905 | 0.09429799999999999 | 1504050.949725922 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 500.784558 | 0.1264305 | 0.13768655 | 0.14870872999999996 | 1007530.6619917243 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 536.281472 | 0.13751950000000002 | 0.14551865 | 0.15522347999999997 | 929808.3330875636 | - |
| `full_mlp_capacity_search_hd128_depth3` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 508.42714 | 0.137532 | 0.14963775 | 0.15782564999999998 | 926121.8156965493 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.181787 | 0.0110545 | 0.0114027 | 0.011580509999999999 | 90228.92883824841 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.487172 | 0.011074 | 0.0113876 | 0.016456379999999982 | 88748.62661500314 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.678055 | 0.0112055 | 0.0116313 | 0.012390429999999997 | 88713.3547309945 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.226118 | 0.010963500000000001 | 0.012585749999999993 | 0.01962346 | 88449.70661232318 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.186722 | 0.011665 | 0.013754549999999992 | 0.017953629999999995 | 167323.41105505778 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.43378 | 0.0120205 | 0.013097549999999998 | 0.015180629999999994 | 165011.60031550218 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.715 | 0.012036 | 0.01247045 | 0.013342499999999998 | 165700.6319822104 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.199939 | 0.011759499999999999 | 0.01201755 | 0.01480442999999999 | 169381.2840456381 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.354955 | 0.0123285 | 0.01259065 | 0.016326109999999987 | 322047.1895746884 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.544591 | 0.018115 | 0.02211459999999999 | 0.0257922 | 216277.94311024985 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.831775 | 0.017729500000000002 | 0.018863799999999997 | 0.02226992 | 225730.06747071713 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.733536 | 0.017407 | 0.01871465 | 0.023103599999999985 | 229136.54475838694 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.333136 | 0.013954500000000002 | 0.0157216 | 0.017937219999999993 | 565942.9670974908 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.468372 | 0.021163500000000002 | 0.0220704 | 0.026229149999999993 | 376896.61443193676 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.705386 | 0.0210615 | 0.0231431 | 0.02650931 | 380849.3702655663 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.482233 | 0.021965 | 0.028336749999999994 | 0.03100176 | 358525.0994458995 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.349729 | 0.016857999999999998 | 0.0183324 | 0.04270717999999993 | 896531.3203216754 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.495418 | 0.027295 | 0.030351699999999995 | 0.03497627 | 578717.9372177846 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.712467 | 0.028173 | 0.039833549999999995 | 0.04080822 | 539072.2970072727 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.36109 | 0.0281795 | 0.03351619999999999 | 0.03859832 | 560898.1100538181 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.243841 | 0.022100500000000002 | 0.023294599999999995 | 0.029090149999999995 | 1430333.8041515439 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.473204 | 0.0489355 | 0.05268779999999999 | 0.055489859999999995 | 649542.5596523648 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.551328 | 0.053538 | 0.07358179999999999 | 0.07672795999999998 | 567509.7141701855 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.301026 | 0.077352 | 0.0841978 | 0.08757832 | 414659.7781311025 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.349592 | 0.0312275 | 0.03606099999999999 | 0.040354629999999996 | 2019012.791076973 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.467067 | 0.0775 | 0.08797545 | 0.11072475999999994 | 822239.7347865734 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.706568 | 0.085029 | 0.11482824999999999 | 0.12114223999999998 | 756648.9342836213 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.605174 | 0.11200299999999999 | 0.12019125 | 0.12286485999999999 | 571767.6500653333 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.331707 | 0.0486795 | 0.05262529999999999 | 0.05496642 | 2580109.3724488663 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.569772 | 0.131435 | 0.13897945 | 0.1408834 | 1061801.8445488918 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.712294 | 0.164634 | 0.19095945 | 0.19515679 | 767853.9848862342 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.305815 | 0.1533825 | 0.1702929 | 0.17472507 | 831044.0666310355 | - |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 216.869489 | 0.584549 | 0.8038166999999999 | 0.90569791 | 1799.3950865573813 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 223.572 | 0.207226 | 0.3979165999999999 | 0.6100858499999997 | 4130.099795601361 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 216.385743 | 0.210988 | 0.30897059999999993 | 1.4603249199999957 | 3801.1976661558615 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 212.316839 | 0.3158265 | 0.42432624999999996 | 0.4613190899999999 | 3109.8526850343687 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 223.872886 | 0.261965 | 0.5601942499999997 | 0.8052306399999997 | 6922.220750865744 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 211.174668 | 0.585688 | 0.9212930499999997 | 1.7708584399999971 | 3256.8289025587405 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 212.742212 | 0.1632915 | 0.23468229999999998 | 0.4127391399999999 | 11440.512708865152 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 220.799821 | 0.229986 | 0.30065035 | 0.32760672999999996 | 8614.785556450535 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 212.590067 | 0.20223950000000002 | 0.2825854 | 0.29275056 | 19064.44841641636 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 214.346615 | 0.185891 | 0.2550794 | 0.26610452999999995 | 20718.28426915579 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 210.34344 | 0.205125 | 0.29775640000000003 | 0.3088843 | 19239.93595025322 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 205.811865 | 0.244425 | 0.8657836499999998 | 1.0124182899999996 | 11327.040369515242 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 213.977777 | 0.214557 | 0.27917374999999994 | 0.29165501 | 37005.35634030348 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 210.17711 | 0.22549750000000002 | 0.52860545 | 0.7942996 | 29109.581564319804 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 214.961706 | 0.224711 | 0.2878424499999999 | 0.32978559999999996 | 35472.66303657169 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 214.197917 | 0.249788 | 0.3185119 | 4.759089369999982 | 18980.619269276525 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 211.283148 | 0.2447585 | 0.42021269999999966 | 1.269836939999998 | 54872.11196928808 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 213.414285 | 0.24036249999999998 | 0.34563204999999997 | 0.4189348899999999 | 65908.98249630436 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 225.214451 | 0.257324 | 0.34654874999999996 | 0.37543641999999994 | 60016.1698565636 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 217.533331 | 0.294337 | 0.7791584 | 0.9968496099999996 | 41031.168660516465 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 221.063432 | 0.26312250000000004 | 0.3653545499999999 | 0.47103295999999967 | 115945.15446358408 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 220.265965 | 0.2740165 | 0.4016205999999999 | 0.4816320199999998 | 111928.01461667941 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 215.336743 | 0.2786775 | 0.5992627999999997 | 1.0518436099999995 | 99404.93101888004 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 210.988621 | 0.29019700000000004 | 2.497438949999999 | 3.945612109999999 | 55188.8200971744 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 212.461767 | 0.49250499999999997 | 1.8964732999999994 | 2.1321925699999995 | 81575.46491450726 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 209.567689 | 0.2788235 | 0.3692637 | 0.38514798999999994 | 221201.28611957785 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 212.962014 | 0.27495 | 0.34045044999999996 | 0.34880658 | 229902.16800618207 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 220.784842 | 0.257536 | 0.3682127499999999 | 0.40415497 | 244183.62242130644 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 208.828522 | 0.28610349999999996 | 0.34477854999999996 | 0.41443233999999995 | 437512.305033579 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 213.8295 | 0.240467 | 0.3480876 | 0.4360996999999997 | 504164.67593830766 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 210.071527 | 0.30833449999999996 | 0.41284399999999993 | 0.4957303899999999 | 396973.96669537283 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 213.676817 | 0.24538949999999998 | 0.35481209999999996 | 0.4627243799999996 | 495489.9191414646 | - |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd128_depth3` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
