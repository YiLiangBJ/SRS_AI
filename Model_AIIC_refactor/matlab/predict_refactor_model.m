function [outputData, debug, modelHandle] = predict_refactor_model(modelOrDir, inputData, mode)
%PREDICT_REFACTOR_MODEL Unified API entry for ONNX or Matlab bundle inference.
%
% Usage:
%   outputData = predict_refactor_model(".../<run_name>/onnx_exports")
%   outputData = predict_refactor_model(".../<run_name>/matlab_exports", [], "bundle")
%   [outputData, debug, modelHandle] = predict_refactor_model(modelHandle, inputData)

if nargin < 2
    inputData = [];
end
if nargin < 3 || isempty(mode)
    mode = "auto";
end

if (ischar(modelOrDir) || isstring(modelOrDir))
    modelHandle = import_refactor_model(modelOrDir, mode);
else
    modelHandle = modelOrDir;
    if ~isfield(modelHandle, "mode")
        error("predict_refactor_model:MissingMode", ...
            "Imported model handle must contain a mode field.");
    end
end

[inputData, ioSpec] = prepare_refactor_input(modelHandle, inputData, modelHandle.mode);
debug = struct();
debug.io_spec = ioSpec;

switch string(modelHandle.mode)
    case "onnx"
        outputData = local_predict_onnx(modelHandle.model, inputData, modelHandle.manifest);
    case "bundle"
        [outputData, bundleDebug] = predict_refactor_matlab_bundle(modelHandle.model, inputData);
        debug.bundle = bundleDebug;
    otherwise
        error("predict_refactor_model:UnsupportedMode", ...
            "Unsupported mode: %s", string(modelHandle.mode));
end
end

function outputData = local_predict_onnx(net, inputData, manifest)
inputShape = double(reshape(manifest.dummy_input_shape, 1, []));
outputShape = double(reshape(manifest.dummy_output_shape, 1, []));
fixedBatchSize = inputShape(1);
requestedBatchSize = size(inputData, 1);
supportsDynamicBatch = isfield(manifest, "dynamic_batch") && logical(manifest.dynamic_batch);

if supportsDynamicBatch || requestedBatchSize == fixedBatchSize
    outputData = local_normalize_output_shape(predict(net, inputData), requestedBatchSize, outputShape);
    return;
end

numChunks = ceil(requestedBatchSize / fixedBatchSize);
chunkOutputs = cell(numChunks, 1);
for chunkIdx = 1:numChunks
    startIdx = (chunkIdx - 1) * fixedBatchSize + 1;
    endIdx = min(chunkIdx * fixedBatchSize, requestedBatchSize);
    validCount = endIdx - startIdx + 1;

    chunkInput = zeros(fixedBatchSize, size(inputData, 2), "single");
    chunkInput(1:validCount, :) = inputData(startIdx:endIdx, :);
    chunkOutput = predict(net, chunkInput);
    chunkOutput = local_normalize_output_shape(chunkOutput, fixedBatchSize, outputShape);
    chunkOutputs{chunkIdx} = chunkOutput(1:validCount, :, :);
end

outputData = cat(1, chunkOutputs{:});
end

function outputData = local_normalize_output_shape(rawOutput, batchSize, outputShape)
expectedShape = [batchSize, outputShape(2:end)];
if isequal(size(rawOutput), expectedShape)
    outputData = rawOutput;
    return;
end
if batchSize == 1 && isequal(size(rawOutput), outputShape(2:end))
    outputData = reshape(rawOutput, expectedShape);
    return;
end
if numel(rawOutput) == prod(expectedShape)
    outputData = reshape(rawOutput, expectedShape);
    return;
end
error("predict_refactor_model:CouldNotNormalizeOutput", ...
    "Could not normalize ONNX output to expected size [%s].", ...
    strjoin(string(expectedShape), ", "));
end