function [outputData, debug] = predict_refactor_separator1_bundle_explicit(bundleOrDir, inputData)
%PREDICT_REFACTOR_SEPARATOR1_BUNDLE_EXPLICIT Clear Matlab reference for separator1.
%
% Usage:
%   [outputData, debug] = predict_refactor_separator1_bundle_explicit(bundle, inputData)
%   [outputData, debug] = predict_refactor_separator1_bundle_explicit(".../matlab_exports/my_run", inputData)
%   [outputData, debug] = predict_refactor_separator1_bundle_explicit(".../matlab_exports/my_run")
%
% This function is intentionally verbose and separator1-specific.
% It exposes the real-branch / imag-branch MLP structure clearly:
%   stage -> port -> layer -> Wx + b -> ReLU -> residual refinement

if ischar(bundleOrDir) || isstring(bundleOrDir)
    bundle = import_refactor_matlab_bundle(bundleOrDir);
else
    bundle = bundleOrDir;
end

manifest = bundle.manifest;
weights = bundle.weights;
modelSpec = manifest.model_spec;

if string(modelSpec.model_type) ~= "separator1"
    error("predict_refactor_separator1_bundle_explicit:WrongModelType", ...
        "This helper only supports separator1 bundles, got %s.", string(modelSpec.model_type));
end

if nargin < 2 || isempty(inputData)
    if isfield(weights, "sample_input")
        inputData = single(weights.sample_input);
    else
        inputData = randn(1, double(modelSpec.seq_len) * 2, "single");
    end
end

inputData = single(inputData);
seqLen = double(modelSpec.seq_len);
numPorts = double(modelSpec.num_ports);
numStages = double(modelSpec.num_stages);
numLayers = double(modelSpec.mlp_depth);
batchSize = size(inputData, 1);

if size(inputData, 2) ~= seqLen * 2
    error("predict_refactor_separator1_bundle_explicit:BadInputWidth", ...
        "Expected input width %d, got %d.", seqLen * 2, size(inputData, 2));
end

features = repmat(reshape(inputData, [batchSize, 1, seqLen * 2]), [1, numPorts, 1]);
stageOutputs = cell(numStages, 1);
layerTraces = cell(numStages, numPorts);

for stageIdx = 1:numStages
    stageTensor = zeros(size(features), "single");

    for portIdx = 1:numPorts
        portInput = reshape(features(:, portIdx, :), batchSize, seqLen * 2);

        % separator1 uses two real MLP branches fed by the same real-stacked input.
        realBranch = portInput;
        imagBranch = portInput;
        layerTrace = cell(numLayers, 1);

        for layerIdx = 1:numLayers
            realPrefix = sprintf('p%02d_s%02d_real_l%02d', portIdx, stageIdx, layerIdx);
            imagPrefix = sprintf('p%02d_s%02d_imag_l%02d', portIdx, stageIdx, layerIdx);

            realWeight = single(weights.([realPrefix '_weight']));
            realBias = single(weights.([realPrefix '_bias']));
            imagWeight = single(weights.([imagPrefix '_weight']));
            imagBias = single(weights.([imagPrefix '_bias']));

            % Dense layer: y = xW^T + b
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

            layerTrace{layerIdx} = struct( ...
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

        portOutput = [realBranch, imagBranch];
        stageTensor(:, portIdx, :) = reshape(portOutput, [batchSize, 1, seqLen * 2]);
        layerTraces{stageIdx, portIdx} = layerTrace;
    end

    yRecon = reshape(sum(stageTensor, 2), [batchSize, seqLen * 2]);
    residual = inputData - yRecon;
    features = stageTensor + repmat(reshape(residual, [batchSize, 1, seqLen * 2]), [1, numPorts, 1]);
    stageOutputs{stageIdx} = features;
end

outputData = features;
debug = struct();
debug.model_type = "separator1";
debug.stage_outputs = stageOutputs;
debug.stage_port_layer_traces = layerTraces;
debug.port_layer_outputs = layerTraces;
end