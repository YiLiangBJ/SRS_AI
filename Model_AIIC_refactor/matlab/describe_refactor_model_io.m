function ioSpec = describe_refactor_model_io(modelOrManifest, mode, doPrint)
%DESCRIBE_REFACTOR_MODEL_IO Return normalized I/O metadata for ONNX or bundle APIs.
%
% Usage:
%   ioSpec = describe_refactor_model_io(manifest, "onnx")
%   ioSpec = describe_refactor_model_io(bundle, "bundle")
%   ioSpec = describe_refactor_model_io(modelHandle)
%   ioSpec = describe_refactor_model_io(..., ..., true)

if nargin < 2
    mode = [];
end
if nargin < 3 || isempty(doPrint)
    doPrint = false;
end

[manifest, inferredMode] = local_extract_manifest(modelOrManifest, mode);
ioSpec = local_build_io_spec(manifest, inferredMode);

if doPrint
    local_print_io_spec(ioSpec);
end
end

function [manifest, mode] = local_extract_manifest(modelOrManifest, mode)
if nargin < 2 || isempty(mode)
    mode = [];
end

if isstruct(modelOrManifest) && isfield(modelOrManifest, "manifest")
    manifest = modelOrManifest.manifest;
    if isempty(mode)
        if isfield(modelOrManifest, "mode")
            mode = string(modelOrManifest.mode);
        elseif isfield(manifest, "mat_file")
            mode = "bundle";
        else
            mode = "onnx";
        end
    end
elseif isstruct(modelOrManifest)
    manifest = modelOrManifest;
    if isempty(mode)
        if isfield(manifest, "mat_file")
            mode = "bundle";
        else
            mode = "onnx";
        end
    end
else
    error("describe_refactor_model_io:UnsupportedInput", ...
        "Expected a manifest, bundle, or imported model handle.");
end

mode = string(mode);
end

function ioSpec = local_build_io_spec(manifest, mode)
ioSpec = struct();
ioSpec.mode = mode;
ioSpec.input = struct();
ioSpec.output = struct();

switch mode
    case "onnx"
        inputShape = double(reshape(manifest.dummy_input_shape, 1, []));
        outputShape = double(reshape(manifest.dummy_output_shape, 1, []));
        dynamicBatch = isfield(manifest, "dynamic_batch") && logical(manifest.dynamic_batch);

        ioSpec.input.name = local_get_matlab_note(manifest, "input_name", "mixed_signal");
        ioSpec.output.name = local_get_matlab_note(manifest, "output_name", "separated_channels");
        ioSpec.input.layout = local_get_matlab_note(manifest, "input_layout", "N x (2*seq_len) real-stacked float32");
        ioSpec.output.layout = local_get_matlab_note(manifest, "output_layout", "N x num_ports x (2*seq_len) real-stacked float32");
        ioSpec.input.dynamic_mask = local_dynamic_mask(inputShape, dynamicBatch);
        ioSpec.output.dynamic_mask = local_dynamic_mask(outputShape, dynamicBatch);
        ioSpec.input.shape = local_apply_dynamic_mask(inputShape, ioSpec.input.dynamic_mask);
        ioSpec.output.shape = local_apply_dynamic_mask(outputShape, ioSpec.output.dynamic_mask);
        ioSpec.input.shape_string = local_format_shape(ioSpec.input.shape);
        ioSpec.output.shape_string = local_format_shape(ioSpec.output.shape);
    case "bundle"
        seqLen = double(manifest.model_spec.seq_len);
        numPorts = double(manifest.model_spec.num_ports);

        ioSpec.input.name = "inputData";
        ioSpec.output.name = "outputData";
        ioSpec.input.layout = local_get_manifest_field(manifest, "input_layout", "N x (2*seq_len) real-stacked float32");
        ioSpec.output.layout = local_get_manifest_field(manifest, "output_layout", "N x num_ports x (2*seq_len) real-stacked float32");
        ioSpec.input.dynamic_mask = [true false];
        ioSpec.output.dynamic_mask = [true false false];
        ioSpec.input.shape = [-1, seqLen * 2];
        ioSpec.output.shape = [-1, numPorts, seqLen * 2];
        ioSpec.input.shape_string = local_format_shape(ioSpec.input.shape);
        ioSpec.output.shape_string = local_format_shape(ioSpec.output.shape);
    otherwise
        error("describe_refactor_model_io:UnsupportedMode", "Unsupported mode: %s", mode);
end
end

function mask = local_dynamic_mask(shape, dynamicBatch)
mask = false(size(shape));
if ~isempty(shape) && dynamicBatch
    mask(1) = true;
end
end

function outShape = local_apply_dynamic_mask(shape, mask)
outShape = double(shape);
outShape(mask) = -1;
end

function local_print_io_spec(ioSpec)
disp("I/O specification:");
disp("  Mode: " + string(ioSpec.mode));
disp("  Input name: " + string(ioSpec.input.name));
disp("  Input shape: " + string(ioSpec.input.shape_string));
disp("  Input dynamic dims: " + local_format_dim_indices(ioSpec.input.dynamic_mask));
disp("  Input layout: " + string(ioSpec.input.layout));
disp("  Output name: " + string(ioSpec.output.name));
disp("  Output shape: " + string(ioSpec.output.shape_string));
disp("  Output dynamic dims: " + local_format_dim_indices(ioSpec.output.dynamic_mask));
disp("  Output layout: " + string(ioSpec.output.layout));
end

function text = local_format_dim_indices(mask)
idx = find(mask);
if isempty(idx)
    text = "none";
else
    text = "[" + join(string(idx), ", ") + "]";
end
end

function text = local_format_shape(shape)
shape = double(reshape(shape, 1, []));
text = "[" + join(string(shape), ", ") + "]";
end

function value = local_get_matlab_note(manifest, fieldName, defaultValue)
value = defaultValue;
if isfield(manifest, "matlab_notes") && isfield(manifest.matlab_notes, fieldName)
    value = manifest.matlab_notes.(fieldName);
end
end

function value = local_get_manifest_field(manifest, fieldName, defaultValue)
value = defaultValue;
if isfield(manifest, fieldName)
    value = manifest.(fieldName);
end
end