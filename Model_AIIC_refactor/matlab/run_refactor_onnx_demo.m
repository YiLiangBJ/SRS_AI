clear;
clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% Edit this path to your exported ONNX run directory.
exportDir = "onnx_exports/my_run";
batchSize = 2;

[net, inputData, outputData, manifest] = demo_refactor_onnx_inference(exportDir, batchSize);

disp("ONNX demo finished.");
disp("  Network class: " + string(class(net)));
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));
disp("  Run name: " + string(manifest.run_name));