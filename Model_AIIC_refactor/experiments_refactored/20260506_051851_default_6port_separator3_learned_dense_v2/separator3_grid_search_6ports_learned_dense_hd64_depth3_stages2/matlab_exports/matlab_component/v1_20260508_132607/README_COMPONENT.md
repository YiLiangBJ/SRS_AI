# SRS AI Matlab Component

This folder is a copyable off-the-shelf Matlab component package for run:

- `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2`

Copy this entire folder to your Matlab project, then either:

```matlab
component = load_srs_ai_matlab_component();
outputData = predict_srs_ai_matlab_component(randn(8, 24, 'single'));
```

Model-specific wrapper:

```matlab
[outputData, ports] = predict_separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2_component(randn(8, 24, 'single'));
```

or:

```matlab
[inputData, outputData, ports] = demo_srs_ai_matlab_component(8);
```

Step-by-step debug walkthrough:

```matlab
debug_separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2_step_by_step
```

Interface contract:

- input: `N x 24` real-stacked float32
- output: `N x 6 x 24` real-stacked float32
- `ports{k}`: `N x 24` output for port `k`

Recommended first debug order:

1. `load_srs_ai_matlab_component`
2. inspect `component.manifest` and `component.io_spec`
3. `prepare_refactor_input`
4. `predict_srs_ai_matlab_component`
5. `split_srs_ai_matlab_ports`
6. compare against `reference_output`

Required colocated files in this folder:

- `matlab_model_bundle.mat`
- `matlab_model_bundle_manifest.json`
- runtime `*.m` helpers

This package is versioned so future migrated models can coexist without overwriting each other.
