function [modelHandle, inputData, outputData, info] = demo_refactor_model_inference(exportPath, mode, batchSize)
%DEMO_REFACTOR_MODEL_INFERENCE Unified inference demo for ONNX or Matlab bundle.
%
% Recommended main Matlab entrypoint for this repo.
%
% Usage:
%   [modelHandle, inputData, outputData, info] = demo_refactor_model_inference("path/to/artifact")
%   [modelHandle, inputData, outputData, info] = demo_refactor_model_inference("path/to/checkpoint_batch_100000.onnx", "onnx", 2)
%   [modelHandle, inputData, outputData, info] = demo_refactor_model_inference("path/to/matlab_model_bundle.mat", "bundle", 8)
%
% This helper is a thin demo wrapper around:
%   - import_refactor_model
%   - describe_refactor_model_io
%   - prepare_refactor_input
%   - predict_refactor_model

if nargin < 2 || isempty(mode)
    mode = "auto";
end
if nargin < 3
    batchSize = [];
end

modelHandle = import_refactor_model(exportPath, mode);
info = struct();
info.mode = modelHandle.mode;
info.manifest = modelHandle.manifest;
info.io_spec = describe_refactor_model_io(modelHandle, [], false);
info.requested_batch_size = [];

if nargin >= 3 && ~isempty(batchSize)
    info.requested_batch_size = double(batchSize);
    inputData = batchSize;
else
    inputData = [];
end

[inputData, preparedIoSpec] = prepare_refactor_input(modelHandle, inputData, modelHandle.mode);
[outputData, debug, modelHandle] = predict_refactor_model(modelHandle, inputData, modelHandle.mode);

info.io_spec = preparedIoSpec;
info.debug = debug;
info.manifest = modelHandle.manifest;

disp("Unified Matlab demo finished.");
disp("  Mode: " + string(modelHandle.mode));
disp("  Input shape spec: " + string(info.io_spec.input.shape_string));
disp("  Output shape spec: " + string(info.io_spec.output.shape_string));
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));
disp("  Run name: " + string(modelHandle.manifest.run_name));
if ~isempty(info.requested_batch_size)
    disp("  Requested batch size: " + string(info.requested_batch_size));
end
end