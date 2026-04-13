clear;
clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% Edit this path to your exported ONNX run directory.
exportDir = "./Model_AIIC_refactor/experiments_refactored/20260409_033734_default_6port_separator1/separator1_grid_search_6ports_hd16_stages2_depth3_share0/onnx_exports/separator1_grid_search_6ports_hd16_stages2_depth3_share0";
batchSize = 2;

[net, inputData, outputData, manifest] = demo_refactor_onnx_inference(exportDir, batchSize);

disp("ONNX demo finished.");
disp("  Network class: " + string(class(net)));
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));
disp("  Run name: " + string(manifest.run_name));