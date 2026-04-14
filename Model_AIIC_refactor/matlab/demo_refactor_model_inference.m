function [modelHandle, inputData, outputData, info] = demo_refactor_model_inference(exportDir, mode, batchSize)
%DEMO_REFACTOR_MODEL_INFERENCE Unified inference demo for ONNX or Matlab bundle.
%
% Usage:
%   [modelHandle, inputData, outputData, info] = demo_refactor_model_inference("path/to/export")
%   [modelHandle, inputData, outputData, info] = demo_refactor_model_inference("path/to/export", "onnx", 2)
%   [modelHandle, inputData, outputData, info] = demo_refactor_model_inference("path/to/export", "bundle", 2)
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

modelHandle = import_refactor_model(exportDir, mode);
info = struct();
info.mode = modelHandle.mode;
info.manifest = modelHandle.manifest;
info.io_spec = describe_refactor_model_io(modelHandle, [], false);

if nargin >= 3 && ~isempty(batchSize)
    featureDim = double(info.io_spec.input.shape(end));
    inputData = randn(batchSize, featureDim, "single");
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
end