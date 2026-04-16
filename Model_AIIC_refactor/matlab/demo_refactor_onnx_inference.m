function [net, inputData, outputData, manifest] = demo_refactor_onnx_inference(exportDir, batchSize)
%DEMO_REFACTOR_ONNX_INFERENCE End-to-end Matlab example for refactor ONNX inference.
%
% Advanced ONNX-only helper. Most users should start from
% demo_refactor_model_inference or run_refactor_model_demo.
%
% Usage:
%   [net, inputData, outputData, manifest] = demo_refactor_onnx_inference(".../<run_name>/onnx_exports")
%   [net, inputData, outputData, manifest] = demo_refactor_onnx_inference(".../<run_name>/onnx_exports", 4)
%   [net, inputData, outputData, manifest] = demo_refactor_onnx_inference(".../<run_name>/checkpoint_batch_87000.onnx", 4)
%
% This helper is a thin demo wrapper around:
%   - import_refactor_model
%   - describe_refactor_model_io
%   - prepare_refactor_input
%   - predict_refactor_model

if nargin < 1
    error("demo_refactor_onnx_inference:MissingExportDir", ...
    "Provide an ONNX export directory, .onnx file, or matching manifest file.");
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

validateattributes(batchSize, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
    mfilename, "batchSize");

[inputData, ioSpec] = prepare_refactor_input(modelHandle, batchSize, "onnx");
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