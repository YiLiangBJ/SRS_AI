clear;
clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
repoRoot = fileparts(fileparts(thisDir));

% Point this to a separator1 Matlab bundle export directory.
exportDir = fullfile(repoRoot, "Model_AIIC_refactor", "experiments_refactored", ...
    "20260409_033734_default_6port_separator1", ...
    "separator1_grid_search_6ports_hd16_stages2_depth3_share0", ...
    "matlab_exports");

bundle = import_refactor_matlab_bundle(exportDir);
useReferenceSample = false;
requestedBatchSize = 4;

if useReferenceSample && isfield(bundle.weights, "sample_input")
    inputData = single(bundle.weights.sample_input);
else
    inputData = prepare_refactor_input(bundle, requestedBatchSize, "bundle");
end

[outputData, debug] = predict_refactor_separator1_bundle_explicit(bundle, inputData);

disp("separator1 explicit Matlab demo finished.");
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));
if ~useReferenceSample
    disp("  Generated input at runtime from model metadata.");
    disp("  Requested batch size: " + string(requestedBatchSize));
end

if useReferenceSample && isfield(bundle.weights, "reference_output")
    maxAbsDiff = max(abs(single(bundle.weights.reference_output(:)) - outputData(:)));
    disp("  Max abs diff vs reference_output: " + string(maxAbsDiff));
end

disp("  See debug.stage_port_layer_traces for per-layer Wx+b traces.");
disp("  debug.port_layer_outputs is kept as a backward-compatible alias.");