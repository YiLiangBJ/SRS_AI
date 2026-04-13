clear;
clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
repoRoot = fileparts(fileparts(thisDir));

% Edit this path to your explicit Matlab bundle export directory.
exportDir = fullfile(repoRoot, "Model_AIIC_refactor", "experiments_refactored", ...
    "20260409_033734_default_6port_separator1", ...
    "separator1_grid_search_6ports_hd16_stages2_depth3_share0", ...
    "matlab_exports", ...
    "separator1_grid_search_6ports_hd16_stages2_depth3_share0");

bundle = import_refactor_matlab_bundle(exportDir);
inputData = single(bundle.weights.sample_input);
[outputData, debug] = predict_refactor_matlab_bundle(bundle, inputData);

disp("Matlab bundle demo finished.");
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));

if isfield(bundle.weights, "reference_output")
    referenceOutput = single(bundle.weights.reference_output);
    maxAbsDiff = max(abs(outputData(:) - referenceOutput(:)));
    disp("  Max abs diff vs exported reference_output: " + string(maxAbsDiff));
end

disp("  Saved stage outputs in variable: debug.stage_outputs");