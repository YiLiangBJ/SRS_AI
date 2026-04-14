function [net, manifest] = import_refactor_onnx(exportDir)
%IMPORT_REFACTOR_ONNX Import an exported refactor ONNX model into Matlab.
%
% Usage:
%   net = import_refactor_onnx("experiments_refactored/.../onnx_exports/my_run")
%   [net, manifest] = import_refactor_onnx("experiments_refactored/.../onnx_exports/my_run")
%
% The export directory must contain:
%   - <run_name>.onnx
%   - export_manifest.json

exportDir = resolve_refactor_export_dir(exportDir);
manifestPath = fullfile(char(exportDir), 'export_manifest.json');
if ~isfile(manifestPath)
    error("import_refactor_onnx:ManifestNotFound", ...
    "export_manifest.json was not found under %s", char(exportDir));
end

manifest = jsondecode(fileread(manifestPath));

onnxPath = local_resolve_onnx_path(exportDir, manifest);
if ~isfile(onnxPath)
    error("import_refactor_onnx:OnnxNotFound", ...
        "ONNX file referenced by the manifest was not found: %s", char(onnxPath));
end

net = importNetworkFromONNX(char(onnxPath), OutputLayerType="regression");

inputLayout = local_get_matlab_note(manifest, "input_layout", "N x (2*seq_len) real-stacked float32");
outputLayout = local_get_matlab_note(manifest, "output_layout", "N x num_ports x (2*seq_len) real-stacked float32");
inputName = local_get_matlab_note(manifest, "input_name", "mixed_signal");
outputName = local_get_matlab_note(manifest, "output_name", "separated_channels");
ioSpec = describe_refactor_model_io(manifest, "onnx", false);
manifest.io_spec = ioSpec;

disp("Imported ONNX model:");
disp("  Run: " + string(manifest.run_name));
disp("  ONNX path: " + string(onnxPath));
disp("  Input layout: " + string(inputLayout));
disp("  Output layout: " + string(outputLayout));
disp("  Suggested input name: " + string(inputName));
disp("  Suggested output name: " + string(outputName));
disp("  Input shape: " + string(ioSpec.input.shape_string));
disp("  Output shape: " + string(ioSpec.output.shape_string));
disp("  Input dynamic dims: " + local_format_dim_indices(ioSpec.input.dynamic_mask));
disp("  Output dynamic dims: " + local_format_dim_indices(ioSpec.output.dynamic_mask));
end

function onnxPath = local_resolve_onnx_path(exportDir, manifest)
if isfield(manifest, "onnx_path")
    candidate = string(manifest.onnx_path);
    if isfile(candidate)
        onnxPath = candidate;
        return;
    end

    [~, onnxName, onnxExt] = fileparts(char(candidate));
    if ~isempty(onnxName) || ~isempty(onnxExt)
        candidate = string(fullfile(char(exportDir), [onnxName onnxExt]));
        if isfile(candidate)
            onnxPath = candidate;
            return;
        end
    end
end

runName = local_get_manifest_field(manifest, "run_name", string(char(exportDir)));
[~, exportDirName] = fileparts(char(exportDir));
fallbackNames = unique([runName, string(exportDirName)], "stable");

for idx = 1:numel(fallbackNames)
    candidate = string(fullfile(char(exportDir), [char(fallbackNames(idx)) '.onnx']));
    if isfile(candidate)
        onnxPath = candidate;
        return;
    end
end

onnxPath = string(fullfile(char(exportDir), [char(fallbackNames(1)) '.onnx']));
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

function text = local_format_dim_indices(mask)
idx = find(mask);
if isempty(idx)
    text = "none";
else
    text = "[" + join(string(idx), ", ") + "]";
end
end
