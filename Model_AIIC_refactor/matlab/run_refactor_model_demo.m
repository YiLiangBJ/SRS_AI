clear;
clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% Set exportDir to either an ONNX export directory or a Matlab bundle directory.
exportDir = "./Model_AIIC_refactor/experiments_refactored/example_run/onnx_exports/example_run";

% Use "auto", "onnx", or "bundle".
mode = "auto";
batchSize = 2;

[modelHandle, inputData, outputData, info] = demo_refactor_model_inference(exportDir, mode, batchSize);

disp("run_refactor_model_demo.m completed.");
disp("  Selected mode: " + string(modelHandle.mode));
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));