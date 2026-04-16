function [inputData, ioSpec] = prepare_refactor_input(modelOrBundleOrManifest, inputData, mode)
%PREPARE_REFACTOR_INPUT Normalize and validate API inputs for ONNX or bundle inference.
%
% Usage:
%   inputData = prepare_refactor_input(modelHandle)
%   inputData = prepare_refactor_input(bundle, [], "bundle")
%   inputData = prepare_refactor_input(bundle, 8, "bundle")
%   inputData = prepare_refactor_input(manifest, randn(2, 24), "onnx")

if nargin < 2
    inputData = [];
end
if nargin < 3
    mode = [];
end

[manifest, weights, ioSpec] = local_extract_context(modelOrBundleOrManifest, mode);
featureDim = local_feature_dim(ioSpec);

if nargin >= 2 && isnumeric(inputData) && isscalar(inputData) && ~isempty(inputData)
    validateattributes(inputData, {'numeric'}, {'integer', 'positive'}, ...
        mfilename, "inputData");
    inputData = randn(double(inputData), featureDim, "single");
end

if nargin < 2 || isempty(inputData)
    if ~isempty(weights) && isfield(weights, "sample_input")
        inputData = single(weights.sample_input);
    else
        inputData = randn(1, featureDim, "single");
    end
end

inputData = single(inputData);
if ndims(inputData) ~= 2
    error("prepare_refactor_input:BadInputRank", ...
        "Expected inputData with shape N x %d.", featureDim);
end

if size(inputData, 2) ~= featureDim
    error("prepare_refactor_input:BadInputWidth", ...
        "Expected input width %d, got %d.", featureDim, size(inputData, 2));
end

if ~isempty(manifest) && isfield(manifest, "dynamic_batch") && ~logical(manifest.dynamic_batch)
    exportedBatchSize = double(manifest.dummy_input_shape(1));
    if size(inputData, 1) ~= exportedBatchSize
        ioSpec.input.batch_execution = "chunk_or_pad_required";
    else
        ioSpec.input.batch_execution = "direct_predict";
    end
else
    ioSpec.input.batch_execution = "direct_predict";
end
end

function [manifest, weights, ioSpec] = local_extract_context(modelOrBundleOrManifest, mode)
weights = [];

if nargin < 2
    mode = [];
end

if isstruct(modelOrBundleOrManifest) && isfield(modelOrBundleOrManifest, "manifest")
    manifest = modelOrBundleOrManifest.manifest;
    if isfield(modelOrBundleOrManifest, "weights")
        weights = modelOrBundleOrManifest.weights;
    elseif isfield(modelOrBundleOrManifest, "model") && isstruct(modelOrBundleOrManifest.model) && isfield(modelOrBundleOrManifest.model, "weights")
        weights = modelOrBundleOrManifest.model.weights;
    end
elseif isstruct(modelOrBundleOrManifest)
    manifest = modelOrBundleOrManifest;
else
    error("prepare_refactor_input:UnsupportedInput", ...
        "Expected a manifest, bundle, or imported model handle.");
end

ioSpec = describe_refactor_model_io(manifest, mode, false);
end

function featureDim = local_feature_dim(ioSpec)
shape = ioSpec.input.shape;
featureDim = double(shape(end));
end