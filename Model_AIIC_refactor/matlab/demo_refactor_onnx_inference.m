function [net, inputData, outputData, manifest] = demo_refactor_onnx_inference(exportDir, batchSize)
%DEMO_REFACTOR_ONNX_INFERENCE End-to-end Matlab example for refactor ONNX inference.
%
% Usage:
%   [net, inputData, outputData, manifest] = demo_refactor_onnx_inference("onnx_exports/my_run")
%   [net, inputData, outputData, manifest] = demo_refactor_onnx_inference("onnx_exports/my_run", 4)
%
% This helper is a thin demo wrapper around:
%   - import_refactor_model
%   - describe_refactor_model_io
%   - prepare_refactor_input
%   - predict_refactor_model

if nargin < 1
    error("demo_refactor_onnx_inference:MissingExportDir", ...
        "Provide the run export directory that contains export_manifest.json.");
end

if nargin < 2 || isempty(batchSize)
    batchSize = [];
end

modelHandle = import_refactor_model(exportDir, "onnx");
manifest = modelHandle.manifest;
net = modelHandle.model;
ioSpec = describe_refactor_model_io(modelHandle, [], false);

if isempty(batchSize)
    if isfield(manifest, "dummy_input_shape") && ~isempty(manifest.dummy_input_shape)
        batchSize = double(manifest.dummy_input_shape(1));
    else
        batchSize = 1;
    end
end

validateattributes(batchSize, {"numeric"}, {"scalar", "integer", "positive"}, ...
    mfilename, "batchSize");

featureDim = double(ioSpec.input.shape(end));
inputData = randn(batchSize, featureDim, "single");
[inputData, ioSpec] = prepare_refactor_input(modelHandle, inputData, "onnx");
[outputData, debug] = predict_refactor_model(modelHandle, inputData, "onnx");

disp("Prepared Matlab inference example:");
disp("  Input tensor name: " + string(ioSpec.input.name));
disp("  Output tensor name: " + string(ioSpec.output.name));
disp("  Input shape spec: " + string(ioSpec.input.shape_string));
disp("  Output shape spec: " + string(ioSpec.output.shape_string));
disp("  Actual input size: " + mat2str(size(inputData)));
disp("  Actual output size: " + mat2str(size(outputData)));
disp("  Batch execution: " + string(debug.io_spec.input.batch_execution));
end