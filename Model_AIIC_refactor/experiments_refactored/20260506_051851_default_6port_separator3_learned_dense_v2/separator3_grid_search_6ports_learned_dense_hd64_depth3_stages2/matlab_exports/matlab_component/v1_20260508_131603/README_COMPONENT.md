# SRS AI Matlab Component

This folder is a copyable off-the-shelf Matlab component package for run:

- `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2`

Copy this entire folder to your Matlab project, then either:

```matlab
component = load_srs_ai_matlab_component();
outputData = predict_srs_ai_matlab_component(randn(8, 24, 'single'));
```

or:

```matlab
[inputData, outputData, ports] = demo_srs_ai_matlab_component(8);
```

Interface contract:

- input: `N x 24` real-stacked float32
- output: `N x 6 x 24` real-stacked float32
- `ports{k}`: `N x 24` output for port `k`

Required colocated files in this folder:

- `matlab_model_bundle.mat`
- `matlab_model_bundle_manifest.json`
- runtime `*.m` helpers

This package is versioned so future migrated models can coexist without overwriting each other.
