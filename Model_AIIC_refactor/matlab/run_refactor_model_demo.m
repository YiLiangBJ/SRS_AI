clear;
clc;

% RECOMMENDED QUICK START.
% Use this script first.
% Set exportPath to exactly one artifact you want to test:
%   - a checkpoint-adjacent .onnx file
%   - a checkpoint-adjacent .export_manifest.json file
%   - an onnx_exports directory
%   - a matlab_model_bundle.mat file
%   - a matlab_model_bundle_manifest.json file
%   - a matlab_exports directory

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
repoRoot = fileparts(fileparts(thisDir));

% Edit this path to the exact artifact you want to use.
exportPath = fullfile(repoRoot, "Model_AIIC_refactor", "experiments_refactored", ...
	"20260414_071319_default_6port_separator1", ...
	"separator1_grid_search_6ports_hd32_stages2_depth3_share0", ...
	"checkpoint_batch_100000.onnx");

% Use "auto" unless you explicitly want to force a backend.
mode = "auto";
requestedBatchSize = 4;

modelHandle = import_refactor_model(exportPath, mode);
ioSpec = describe_refactor_model_io(modelHandle, [], true);
[inputData, preparedIoSpec] = prepare_refactor_input(modelHandle, requestedBatchSize, modelHandle.mode);
[outputData, debug, modelHandle] = predict_refactor_model(modelHandle, inputData, modelHandle.mode);

disp("run_refactor_model_demo.m completed.");
disp("  Selected mode: " + string(modelHandle.mode));
disp("  Artifact path: " + string(exportPath));
disp("  Requested batch size: " + string(requestedBatchSize));
disp("  Input shape spec: " + string(preparedIoSpec.input.shape_string));
disp("  Output shape spec: " + string(preparedIoSpec.output.shape_string));
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));
if isfield(debug, "bundle") && isfield(debug.bundle, "stage_port_layer_traces")
	disp("  Bundle layer traces available: yes");
end