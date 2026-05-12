# Deliver Folder

This folder is the minimal handoff set for deployment.

Main API:

```matlab
state = init_model();
outputData = predict_model(state, x);
```

Short model-specific aliases:

```matlab
state = init_sep3_hd64_d3_s2();
outputData = predict_sep3_hd64_d3_s2(state, x);
```

Input / output contract:

- input: `N x 24`
- output: `N x 6 x 24`
- keep the output as a 3D tensor

Demo:

- run `demo_deliver_two_call.m`
- first call initializes and caches state
- second call reuses cached state without repeating load / parse
- the final reference check uses Python-generated `sample_input` and Python-generated `reference_output`
- treat that check as the required parity-validation gate before deployment handoff
