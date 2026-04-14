clear;
clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
repoRoot = fileparts(fileparts(thisDir));

% Set exportDir to either an ONNX export directory or a Matlab bundle directory.
exportDir = fullfile(repoRoot, "Model_AIIC_refactor", "experiments_refactored", ...
	"20260409_033734_default_6port_separator1", ...
	"separator1_grid_search_6ports_hd16_stages2_depth3_share0", ...
	"matlab_exports", ...
	"separator1_grid_search_6ports_hd16_stages2_depth3_share0");

% Use "auto", "onnx", or "bundle".
mode = "bundle";
batchSize = 1;

modelHandle = import_refactor_model(exportDir, mode);
ioSpec = describe_refactor_model_io(modelHandle, [], true);
inputData = randn(batchSize, double(ioSpec.input.shape(end)), "single");
[inputData, preparedIoSpec] = prepare_refactor_input(modelHandle, inputData, modelHandle.mode);
[outputData, debug, modelHandle] = predict_refactor_model(modelHandle, inputData, modelHandle.mode);

disp("run_refactor_model_demo.m completed.");
disp("  Selected mode: " + string(modelHandle.mode));
disp("  Input shape spec: " + string(preparedIoSpec.input.shape_string));
disp("  Output shape spec: " + string(preparedIoSpec.output.shape_string));
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));
if isfield(debug, "bundle") && isfield(debug.bundle, "stage_port_layer_traces")
	disp("  Bundle layer traces available: yes");
end