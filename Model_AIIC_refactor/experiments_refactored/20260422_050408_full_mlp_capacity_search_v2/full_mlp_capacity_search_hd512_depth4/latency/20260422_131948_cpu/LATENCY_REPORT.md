# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

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

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
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
