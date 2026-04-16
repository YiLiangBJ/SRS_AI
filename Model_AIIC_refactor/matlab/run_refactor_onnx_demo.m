clear;
clc;

% ADVANCED ONNX-ONLY DEMO.
% Most users should start from run_refactor_model_demo.m instead.

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
repoRoot = fileparts(fileparts(thisDir));

% Edit this path to your exported ONNX directory, .onnx file, or manifest file.
exportPath = fullfile(repoRoot, "Model_AIIC_refactor", "experiments_refactored", ...
	"20260409_033734_default_6port_separator1", ...
	"separator1_grid_search_6ports_hd16_stages2_depth3_share0", ...
	"onnx_exports");
batchSize = 2;

modelHandle = import_refactor_model(exportPath, "onnx");
ioSpec = describe_refactor_model_io(modelHandle, [], true);
[inputData, preparedIoSpec] = prepare_refactor_input(modelHandle, batchSize, "onnx");
[outputData, debug, modelHandle] = predict_refactor_model(modelHandle, inputData, "onnx");

disp("ONNX demo finished.");
disp("  Network class: " + string(class(modelHandle.model)));
disp("  Input shape spec: " + string(preparedIoSpec.input.shape_string));
disp("  Output shape spec: " + string(preparedIoSpec.output.shape_string));
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));
disp("  Batch execution: " + string(debug.io_spec.input.batch_execution));
disp("  Run name: " + string(modelHandle.manifest.run_name));