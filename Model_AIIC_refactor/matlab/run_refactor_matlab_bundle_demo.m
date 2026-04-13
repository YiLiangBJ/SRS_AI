clear;
clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% Edit this path to your explicit Matlab bundle export directory.
exportDir = "matlab_exports/my_run";

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