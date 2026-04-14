function [outputData, debug] = predict_refactor_matlab_bundle(bundleOrDir, inputData)
%PREDICT_REFACTOR_MATLAB_BUNDLE Explicit Matlab forward pass from exported weights.
%
% Usage:
%   outputData = predict_refactor_matlab_bundle(bundle, inputData)
%   outputData = predict_refactor_matlab_bundle(".../<run_name>/matlab_exports", inputData)
%   outputData = predict_refactor_matlab_bundle(".../<run_name>/matlab_exports")

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

[normalizedInput, inputRms, normalizationEnabled] = local_normalize_real_stacked_input(inputData, modelSpec);

collectDetailedDebug = nargout > 1;

switch string(modelSpec.model_type)
    case "separator2"
        [outputData, debug] = local_forward_separator2(weights, modelSpec, normalizedInput, collectDetailedDebug);
    case "separator1"
        [outputData, debug] = local_forward_separator1(weights, modelSpec, normalizedInput, collectDetailedDebug);
    otherwise
        error("predict_refactor_matlab_bundle:UnsupportedModel", ...
            "Unsupported model_type: %s", string(modelSpec.model_type));
end

outputData = local_restore_real_stacked_output(outputData, inputRms, normalizationEnabled);
debug.input_rms = inputRms;
debug.normalization_enabled = normalizationEnabled;
debug.normalized_input = normalizedInput;
end

function [outputData, debug] = local_forward_separator2(weights, modelSpec, inputData, collectDetailedDebug)
numPorts = double(modelSpec.num_ports);
numStages = double(modelSpec.num_stages);
numLayers = double(modelSpec.mlp_depth);
seqLen = double(modelSpec.seq_len);
activationType = string(modelSpec.activation_type);

batchSize = size(inputData, 1);
features = repmat(reshape(inputData, [batchSize, 1, size(inputData, 2)]), [1, numPorts, 1]);
stageOutputs = cell(numStages, 1);
layerTraces = cell(numStages, numPorts);

for stageIdx = 1:numStages
    newFeatures = zeros(size(features), "single");
    for portIdx = 1:numPorts
        x = reshape(features(:, portIdx, :), batchSize, []);
        if collectDetailedDebug
            portLayerTrace = cell(numLayers, 1);
        end
        for layerIdx = 1:numLayers
            prefix = local_separator2_prefix(portIdx, stageIdx, layerIdx);
            weightReal = single(weights.([prefix '_weight_real']));
            weightImag = single(weights.([prefix '_weight_imag']));
            biasReal = single(weights.([prefix '_bias_real']));
            biasImag = single(weights.([prefix '_bias_imag']));

            inFeatures = size(weightReal, 2);
            xReal = x(:, 1:inFeatures);
            xImag = x(:, inFeatures + 1:end);

            affineReal = xReal * weightReal.' - xImag * weightImag.' + biasReal;
            affineImag = xReal * weightImag.' + xImag * weightReal.' + biasImag;
            x = [affineReal, affineImag];

            if layerIdx < numLayers
                x = local_apply_complex_activation(x, size(affineReal, 2), activationType);
            end

            if collectDetailedDebug
                portLayerTrace{layerIdx} = struct( ...
                    'layer_index', layerIdx, ...
                    'prefix', prefix, ...
                    'x_real', xReal, ...
                    'x_imag', xImag, ...
                    'weight_real', weightReal, ...
                    'weight_imag', weightImag, ...
                    'bias_real', biasReal, ...
                    'bias_imag', biasImag, ...
                    'affine_real', affineReal, ...
                    'affine_imag', affineImag, ...
                        'post_activation', x ...
                    );
            end
        end
        newFeatures(:, portIdx, :) = reshape(x, [batchSize, 1, seqLen * 2]);
        if collectDetailedDebug
            layerTraces{stageIdx, portIdx} = portLayerTrace;
        end
    end

    yRecon = reshape(sum(newFeatures, 2), [batchSize, size(inputData, 2)]);
    residual = inputData - yRecon;
    features = newFeatures + repmat(reshape(residual, [batchSize, 1, size(residual, 2)]), [1, numPorts, 1]);
    stageOutputs{stageIdx} = features;
end

outputData = features;
debug = struct();
debug.stage_outputs = stageOutputs;
debug.model_type = "separator2";
if collectDetailedDebug
    debug.stage_port_layer_traces = layerTraces;
end
end

function [outputData, debug] = local_forward_separator1(weights, modelSpec, inputData, collectDetailedDebug)
numPorts = double(modelSpec.num_ports);
numStages = double(modelSpec.num_stages);
numLayers = double(modelSpec.mlp_depth);
seqLen = double(modelSpec.seq_len);

batchSize = size(inputData, 1);
features = repmat(reshape(inputData, [batchSize, 1, size(inputData, 2)]), [1, numPorts, 1]);
stageOutputs = cell(numStages, 1);
layerTraces = cell(numStages, numPorts);

for stageIdx = 1:numStages
    newFeatures = zeros(size(features), "single");
    for portIdx = 1:numPorts
        x = reshape(features(:, portIdx, :), batchSize, []);
        realBranch = x;
        imagBranch = x;
        if collectDetailedDebug
            portLayerTrace = cell(numLayers, 1);
        end

        for layerIdx = 1:numLayers
            realPrefix = local_separator1_prefix(portIdx, stageIdx, "real", layerIdx);
            imagPrefix = local_separator1_prefix(portIdx, stageIdx, "imag", layerIdx);

            realWeight = single(weights.([realPrefix '_weight']));
            realBias = single(weights.([realPrefix '_bias']));
            imagWeight = single(weights.([imagPrefix '_weight']));
            imagBias = single(weights.([imagPrefix '_bias']));

            realInput = realBranch;
            imagInput = imagBranch;
            realAffine = realInput * realWeight.' + realBias;
            imagAffine = imagInput * imagWeight.' + imagBias;

            realBranch = realAffine;
            imagBranch = imagAffine;

            if layerIdx < numLayers
                realBranch = max(realBranch, 0);
                imagBranch = max(imagBranch, 0);
            end

            if collectDetailedDebug
                portLayerTrace{layerIdx} = struct( ...
                    'layer_index', layerIdx, ...
                    'real_prefix', realPrefix, ...
                    'imag_prefix', imagPrefix, ...
                    'real_input', realInput, ...
                    'imag_input', imagInput, ...
                    'real_weight', realWeight, ...
                    'imag_weight', imagWeight, ...
                    'real_bias', realBias, ...
                    'imag_bias', imagBias, ...
                    'real_affine', realAffine, ...
                    'imag_affine', imagAffine, ...
                    'real_post_activation', realBranch, ...
                    'imag_post_activation', imagBranch ...
                    );
            end
        end

        x = [realBranch, imagBranch];
        newFeatures(:, portIdx, :) = reshape(x, [batchSize, 1, seqLen * 2]);
        if collectDetailedDebug
            layerTraces{stageIdx, portIdx} = portLayerTrace;
        end
    end

    yRecon = reshape(sum(newFeatures, 2), [batchSize, size(inputData, 2)]);
    residual = inputData - yRecon;
    features = newFeatures + repmat(reshape(residual, [batchSize, 1, size(residual, 2)]), [1, numPorts, 1]);
    stageOutputs{stageIdx} = features;
end

outputData = features;
debug = struct();
debug.stage_outputs = stageOutputs;
debug.model_type = "separator1";
if collectDetailedDebug
    debug.stage_port_layer_traces = layerTraces;
end
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

function [normalizedInput, inputRms, enabled] = local_normalize_real_stacked_input(inputData, modelSpec)
enabled = local_manifest_bool(modelSpec, "normalize_energy", false);
batchSize = size(inputData, 1);
if ~enabled
    inputRms = ones(batchSize, 1, "single");
    normalizedInput = inputData;
    return;
end

seqLen = double(modelSpec.seq_len);
realPart = inputData(:, 1:seqLen);
imagPart = inputData(:, seqLen + 1:end);
inputRms = sqrt(mean(realPart.^2 + imagPart.^2, 2));
normalizedInput = inputData ./ (inputRms + 1e-8);
end

function outputData = local_restore_real_stacked_output(outputData, inputRms, enabled)
if ~enabled
    return;
end

scale = reshape(inputRms, [size(inputRms, 1), 1, 1]);
outputData = outputData .* scale;
end

function value = local_manifest_bool(structValue, fieldName, defaultValue)
if isstruct(structValue) && isfield(structValue, fieldName)
    value = logical(structValue.(fieldName));
else
    value = logical(defaultValue);
end
end