# Demo Guide

Recommended order:

1. Run `demo_quick_start.m` to confirm the basic API.
2. Run `demo_step_by_step.m` section by section to inspect initialization, sample tensors, and reference-output matching.
3. Use `demo_sim_platform_loop.m` as the template for slot-based platform integration.

Deployment-first API:

```matlab
state = init_model(componentDir);      % once
outputData = predict_model(state, x);  % every slot
```

Short model-specific aliases:

```matlab
state = init_sep3_hd64_d3_s2(componentDir);
outputData = predict_sep3_hd64_d3_s2(state, x);
```
