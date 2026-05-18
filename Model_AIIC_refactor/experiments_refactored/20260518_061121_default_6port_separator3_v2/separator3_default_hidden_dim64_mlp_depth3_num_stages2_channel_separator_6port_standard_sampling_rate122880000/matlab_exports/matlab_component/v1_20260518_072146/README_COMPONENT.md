# SRS AI Matlab Component

    This folder is a copyable off-the-shelf Matlab component package for run:

    - `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000`

Short deployment tag:

- `sep3_hd64_d3_s2`

Copy this entire folder to your Matlab project.

Recommended deployment pattern:

    ```matlab
state = init_model();
outputData = predict_model(state, randn(8, 24, 'single'));
    ```

Short model-specific aliases:

    ```matlab
state = init_sep3_hd64_d3_s2();
outputData = predict_sep3_hd64_d3_s2(state, randn(8, 24, 'single'));
    ```

If you want a quick smoke test:

    ```matlab
    [inputData, outputData] = demo_srs_ai_matlab_component(8);
    ```

Demo scripts are under `demo/`.

    Interface contract:

    - input: `N x 24` real-stacked float32
    - output: `N x 6 x 24` real-stacked float32
    - keep the output as a 3D tensor; slice the port dimension directly if needed

    Deployment-first workflow:

    1. `init_model(...)` is the one-time load/parse step.
    2. `predict_model(state, inputData)` is the per-slot fast path.
    3. Keep `state` in a persistent variable in your simulation platform.
    4. Use `demo/demo_sim_platform_loop.m` as the integration template.

    Required colocated files in this folder:

    - `matlab_model_bundle.mat`
    - `matlab_model_bundle_manifest.json`
    - runtime `*.m` helpers

    This package is versioned so future migrated models can coexist without overwriting each other.
