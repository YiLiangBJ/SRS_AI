# Deliver Folder

This folder is the minimal handoff set for deployment.

Main API:

```matlab
state = init_model();
outputData = predict_model(state, x);
ports = split_ports(outputData);
```

Short model-specific aliases:

```matlab
state = init_sep3_hd64_d3_s2();
outputData = predict_sep3_hd64_d3_s2(state, x);
```

Input / output contract:

- input: `N x 24`
- output: `N x 6 x 24`
- `ports{k}`: `N x 24`

Demo:

- run `demo_deliver_two_call.m`
- first call initializes and caches state
- second call reuses cached state without repeating load / parse
