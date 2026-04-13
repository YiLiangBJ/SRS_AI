clear;
clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% Point this to a separator1 Matlab bundle export directory.
exportDir = "./Model_AIIC_refactor/experiments_refactored/20260413_031523_quick_separator1/separator1_small_hd32_stages2_depth3_share0/matlab_exports/separator1_small_hd32_stages2_depth3_share0";

bundle = import_refactor_matlab_bundle(exportDir);
[outputData, debug] = predict_refactor_separator1_bundle_explicit(bundle, bundle.weights.sample_input);

disp("separator1 explicit Matlab demo finished.");
disp("  Input size: " + mat2str(size(bundle.weights.sample_input)));
disp("  Output size: " + mat2str(size(outputData)));

if isfield(bundle.weights, "reference_output")
    maxAbsDiff = max(abs(single(bundle.weights.reference_output(:)) - outputData(:)));
    disp("  Max abs diff vs reference_output: " + string(maxAbsDiff));
end

disp("  See debug.port_layer_outputs for per-layer branch outputs.");