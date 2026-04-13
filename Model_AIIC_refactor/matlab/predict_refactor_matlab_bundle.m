function [outputData, debug] = predict_refactor_matlab_bundle(bundleOrDir, inputData)
%PREDICT_REFACTOR_MATLAB_BUNDLE Explicit Matlab forward pass from exported weights.
%
% Usage:
%   outputData = predict_refactor_matlab_bundle(bundle, inputData)
%   outputData = predict_refactor_matlab_bundle("matlab_exports/my_run", inputData)
%   outputData = predict_refactor_matlab_bundle("matlab_exports/my_run")

if ischar(bundleOrDir) || isstring(bundleOrDir)
    bundle = import_refactor_matlab_bundle(bundleOrDir);
else
    bundle = bundleOrDir;
end

manifest = bundle.manifest;
weights = bundle.weights;
modelSpec = manifest.model_spec;

if nargin < 2 || isempty(inputData)
    if isfield(weights, "sample_input")
        inputData = single(weights.sample_input);
    else
        inputData = randn(1, modelSpec.seq_len * 2, "single");
    end
end

inputData = single(inputData);
if ndims(inputData) ~= 2
    error("predict_refactor_matlab_bundle:BadInputRank", ...
        "Expected inputData with shape N x (2*seq_len).");
end

expectedWidth = double(modelSpec.seq_len) * 2;
if size(inputData, 2) ~= expectedWidth
    error("predict_refactor_matlab_bundle:BadInputWidth", ...
        "Expected input width %d, got %d.", expectedWidth, size(inputData, 2));
end

switch string(modelSpec.model_type)
    case "separator2"
        [outputData, debug] = local_forward_separator2(weights, modelSpec, inputData);
    case "separator1"
        [outputData, debug] = local_forward_separator1(weights, modelSpec, inputData);
    otherwise
        error("predict_refactor_matlab_bundle:UnsupportedModel", ...
            "Unsupported model_type: %s", string(modelSpec.model_type));
end
end

function [outputData, debug] = local_forward_separator2(weights, modelSpec, inputData)
numPorts = double(modelSpec.num_ports);
numStages = double(modelSpec.num_stages);
numLayers = double(modelSpec.mlp_depth);
seqLen = double(modelSpec.seq_len);
activationType = string(modelSpec.activation_type);

batchSize = size(inputData, 1);
features = repmat(reshape(inputData, [batchSize, 1, size(inputData, 2)]), [1, numPorts, 1]);
stageOutputs = cell(numStages, 1);

for stageIdx = 1:numStages
    newFeatures = zeros(size(features), "single");
    for portIdx = 1:numPorts
        x = reshape(features(:, portIdx, :), batchSize, []);
        for layerIdx = 1:numLayers
            prefix = local_separator2_prefix(portIdx, stageIdx, layerIdx);
            weightReal = single(weights.([prefix "_weight_real"]));
            weightImag = single(weights.([prefix "_weight_imag"]));
            biasReal = single(weights.([prefix "_bias_real"]));
            biasImag = single(weights.([prefix "_bias_imag"]));

            inFeatures = size(weightReal, 2);
            xReal = x(:, 1:inFeatures);
            xImag = x(:, inFeatures + 1:end);

            yReal = xReal * weightReal.' - xImag * weightImag.' + biasReal;
            yImag = xReal * weightImag.' + xImag * weightReal.' + biasImag;
            x = [yReal, yImag];

            if layerIdx < numLayers
                x = local_apply_complex_activation(x, size(yReal, 2), activationType);
            end
        end
        newFeatures(:, portIdx, :) = reshape(x, [batchSize, 1, seqLen * 2]);
    end

    yRecon = squeeze(sum(newFeatures, 2));
    residual = inputData - yRecon;
    features = newFeatures + repmat(reshape(residual, [batchSize, 1, size(residual, 2)]), [1, numPorts, 1]);
    stageOutputs{stageIdx} = features;
end

outputData = features;
debug = struct();
debug.stage_outputs = stageOutputs;
end

function [outputData, debug] = local_forward_separator1(weights, modelSpec, inputData)
numPorts = double(modelSpec.num_ports);
numStages = double(modelSpec.num_stages);
numLayers = double(modelSpec.mlp_depth);
seqLen = double(modelSpec.seq_len);

batchSize = size(inputData, 1);
features = repmat(reshape(inputData, [batchSize, 1, size(inputData, 2)]), [1, numPorts, 1]);
stageOutputs = cell(numStages, 1);

for stageIdx = 1:numStages
    newFeatures = zeros(size(features), "single");
    for portIdx = 1:numPorts
        x = reshape(features(:, portIdx, :), batchSize, []);
        realBranch = x;
        imagBranch = x;

        for layerIdx = 1:numLayers
            realPrefix = local_separator1_prefix(portIdx, stageIdx, "real", layerIdx);
            imagPrefix = local_separator1_prefix(portIdx, stageIdx, "imag", layerIdx);

            realWeight = single(weights.([realPrefix "_weight"]));
            realBias = single(weights.([realPrefix "_bias"]));
            imagWeight = single(weights.([imagPrefix "_weight"]));
            imagBias = single(weights.([imagPrefix "_bias"]));

            realBranch = realBranch * realWeight.' + realBias;
            imagBranch = imagBranch * imagWeight.' + imagBias;

            if layerIdx < numLayers
                realBranch = max(realBranch, 0);
                imagBranch = max(imagBranch, 0);
            end
        end

        x = [realBranch, imagBranch];
        newFeatures(:, portIdx, :) = reshape(x, [batchSize, 1, seqLen * 2]);
    end

    yRecon = squeeze(sum(newFeatures, 2));
    residual = inputData - yRecon;
    features = newFeatures + repmat(reshape(residual, [batchSize, 1, size(residual, 2)]), [1, numPorts, 1]);
    stageOutputs{stageIdx} = features;
end

outputData = features;
debug = struct();
debug.stage_outputs = stageOutputs;
end

function x = local_apply_complex_activation(x, hiddenSize, activationType)
xReal = x(:, 1:hiddenSize);
xImag = x(:, hiddenSize + 1:end);

switch activationType
    case "relu"
        x = max(x, 0);
    case "split_relu"
        x = [max(xReal, 0), max(xImag, 0)];
    case "mod_relu"
        magnitude = sqrt(xReal .^ 2 + xImag .^ 2 + 1e-8);
        scale = max(magnitude + 0.5, 0) ./ (magnitude + 1e-8);
        x = [xReal .* scale, xImag .* scale];
    case "z_relu"
        theta = atan2(xImag, xReal);
        gate = single(theta >= 0 & theta <= pi / 2);
        x = [max(xReal, 0) .* gate, max(xImag, 0) .* gate];
    case "cardioid"
        theta = atan2(xImag, xReal);
        scale = 0.5 .* (1 + cos(theta));
        x = [xReal .* scale, xImag .* scale];
    otherwise
        error("predict_refactor_matlab_bundle:UnknownActivation", ...
            "Unknown activation type: %s", activationType);
end
end

function prefix = local_separator2_prefix(portIdx, stageIdx, layerIdx)
prefix = sprintf('p%02d_s%02d_l%02d', portIdx, stageIdx, layerIdx);
end

function prefix = local_separator1_prefix(portIdx, stageIdx, branchName, layerIdx)
prefix = sprintf('p%02d_s%02d_%s_l%02d', portIdx, stageIdx, branchName, layerIdx);
end