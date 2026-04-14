clear;
clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
repoRoot = fileparts(fileparts(thisDir));

% Edit this path to your exported ONNX run directory.
exportDir = fullfile(repoRoot, "Model_AIIC_refactor", "experiments_refactored", ...
	"20260409_033734_default_6port_separator1", ...
	"separator1_grid_search_6ports_hd16_stages2_depth3_share0", ...
	"onnx_exports", ...
	"separator1_grid_search_6ports_hd16_stages2_depth3_share0");
batchSize = 2;

modelHandle = import_refactor_model(exportDir, "onnx");
ioSpec = describe_refactor_model_io(modelHandle, [], true);
inputData = randn(batchSize, double(ioSpec.input.shape(end)), "single");
[inputData, preparedIoSpec] = prepare_refactor_input(modelHandle, inputData, "onnx");
[outputData, debug, modelHandle] = predict_refactor_model(modelHandle, inputData, "onnx");

disp("ONNX demo finished.");
disp("  Network class: " + string(class(modelHandle.model)));
disp("  Input shape spec: " + string(preparedIoSpec.input.shape_string));
disp("  Output shape spec: " + string(preparedIoSpec.output.shape_string));
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));
disp("  Batch execution: " + string(debug.io_spec.input.batch_execution));
disp("  Run name: " + string(modelHandle.manifest.run_name));