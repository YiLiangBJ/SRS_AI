function [net, inputData, outputData, manifest] = demo_refactor_onnx_inference(exportDir, batchSize)
%DEMO_REFACTOR_ONNX_INFERENCE End-to-end Matlab example for refactor ONNX inference.
%
% Usage:
%   [net, inputData, outputData, manifest] = demo_refactor_onnx_inference("onnx_exports/my_run")
%   [net, inputData, outputData, manifest] = demo_refactor_onnx_inference("onnx_exports/my_run", 4)
%
% This helper:
%   1. imports the ONNX network via import_refactor_onnx
%   2. reads dummy_input_shape and dummy_output_shape from export_manifest.json
%   3. creates a random single-precision input of size N x (2*seq_len)
%   4. runs predict(net, inputData)
%   5. prints the actual and expected tensor shapes

if nargin < 1
    error("demo_refactor_onnx_inference:MissingExportDir", ...
        "Provide the run export directory that contains export_manifest.json.");
end

if nargin < 2 || isempty(batchSize)
    batchSize = [];
end

[net, manifest] = import_refactor_onnx(exportDir);

if ~isfield(manifest, "dummy_input_shape") || numel(manifest.dummy_input_shape) < 2
    error("demo_refactor_onnx_inference:MissingInputShape", ...
        "dummy_input_shape is missing or malformed in export_manifest.json.");
end
if ~isfield(manifest, "dummy_output_shape") || numel(manifest.dummy_output_shape) < 3
    error("demo_refactor_onnx_inference:MissingOutputShape", ...
        "dummy_output_shape is missing or malformed in export_manifest.json.");
end

inputShape = double(reshape(manifest.dummy_input_shape, 1, []));
outputShape = double(reshape(manifest.dummy_output_shape, 1, []));

if isempty(batchSize)
    batchSize = inputShape(1);
end

validateattributes(batchSize, {"numeric"}, {"scalar", "integer", "positive"}, ...
    mfilename, "batchSize");

featureDim = inputShape(end);
inputData = randn(batchSize, featureDim, "single");
outputData = predict(net, inputData);

expectedInputShape = [batchSize, inputShape(2:end)];
expectedOutputShape = [batchSize, outputShape(2:end)];

disp("Prepared Matlab inference example:");
disp("  Input tensor name: " + string(manifest.matlab_notes.input_name));
disp("  Output tensor name: " + string(manifest.matlab_notes.output_name));
disp("  Actual input size: " + local_format_size(size(inputData)));
disp("  Expected input size: " + local_format_size(expectedInputShape));
disp("  Actual output size: " + local_format_size(size(outputData)));
disp("  Expected output size: " + local_format_size(expectedOutputShape));
end

function text = local_format_size(shape)
shape = double(reshape(shape, 1, []));
if isempty(shape)
    text = "[]";
    return;
end
text = "[" + join(string(shape), " x ") + "]";
end