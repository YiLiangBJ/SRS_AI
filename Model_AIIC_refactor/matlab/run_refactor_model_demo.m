clear;
clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% Set exportDir to either an ONNX export directory or a Matlab bundle directory.
exportDir = "./Model_AIIC_refactor/experiments_refactored/20260409_033734_default_6port_separator1/separator1_grid_search_6ports_hd16_stages2_depth3_share0/matlab_exports/separator1_grid_search_6ports_hd16_stages2_depth3_share0";

% Use "auto", "onnx", or "bundle".
mode = "bundle";
batchSize = 2;

[modelHandle, inputData, outputData, info] = demo_refactor_model_inference(exportDir, mode, batchSize);

disp("run_refactor_model_demo.m completed.");
disp("  Selected mode: " + string(modelHandle.mode));
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));