function [net, manifest] = import_refactor_onnx(exportPath)
%IMPORT_REFACTOR_ONNX Import an exported refactor ONNX model into Matlab.
%
% Usage:
%   net = import_refactor_onnx("experiments_refactored/.../<run_name>/onnx_exports")
%   net = import_refactor_onnx("experiments_refactored/.../<run_name>/checkpoint_batch_100000.onnx")
%   [net, manifest] = import_refactor_onnx("experiments_refactored/.../<run_name>/checkpoint_batch_100000.export_manifest.json")
%
% For direct .onnx file input, this helper looks for a sibling manifest first.
%

[exportDir, manifestPath, explicitOnnxPath] = local_resolve_export_targets(exportPath);

if ~isfile(manifestPath)
    error("import_refactor_onnx:ManifestNotFound", ...
    "Could not find an ONNX export manifest for %s", char(string(exportPath)));
end

manifest = jsondecode(fileread(manifestPath));
manifest.manifest_path = string(manifestPath);

onnxPath = local_resolve_onnx_path(exportDir, manifest, explicitOnnxPath);
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

function [exportDir, manifestPath, explicitOnnxPath] = local_resolve_export_targets(exportPath)
explicitOnnxPath = "";
exportPath = string(exportPath);
exportPath = strip(exportPath);

if strlength(exportPath) == 0
    error("import_refactor_onnx:EmptyPath", "exportPath must not be empty.");
end

if isfile(char(exportPath))
    [parentDir, ~, ext] = fileparts(char(exportPath));
    exportDir = string(char(java.io.File(parentDir).getCanonicalPath()));
    ext = string(lower(ext));

    if ext == ".onnx"
        explicitOnnxPath = string(char(java.io.File(char(exportPath)).getCanonicalPath()));
        manifestPath = local_manifest_for_onnx_path(explicitOnnxPath);
        return;
    end

    if ext == ".json"
        manifestPath = string(char(java.io.File(char(exportPath)).getCanonicalPath()));
        return;
    end

    error("import_refactor_onnx:UnsupportedFilePath", ...
        "Unsupported file input: %s", char(exportPath));
end

exportDir = resolve_refactor_export_dir(exportPath);
legacyManifest = fullfile(char(exportDir), 'export_manifest.json');
if isfile(legacyManifest)
    manifestPath = string(legacyManifest);
    return;
end

manifestFiles = dir(fullfile(char(exportDir), '*.export_manifest.json'));
if numel(manifestFiles) == 1
    manifestPath = string(fullfile(manifestFiles(1).folder, manifestFiles(1).name));
    return;
end
if numel(manifestFiles) > 1
    error("import_refactor_onnx:AmbiguousManifest", ...
        "Multiple ONNX manifests were found under %s. Pass the specific .onnx or .export_manifest.json file.", ...
        char(exportDir));
end

manifestPath = string(legacyManifest);
end

function manifestPath = local_manifest_for_onnx_path(onnxPath)
[parentDir, baseName] = fileparts(char(onnxPath));
preferred = fullfile(parentDir, strcat(baseName, '.export_manifest.json'));
if isfile(preferred)
    manifestPath = string(preferred);
    return;
end

legacy = fullfile(parentDir, 'export_manifest.json');
manifestPath = string(legacy);
end

function onnxPath = local_resolve_onnx_path(exportDir, manifest, explicitOnnxPath)
if nargin >= 3 && strlength(explicitOnnxPath) > 0 && isfile(char(explicitOnnxPath))
    onnxPath = explicitOnnxPath;
    return;
end

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

resolvedManifestPath = local_get_manifest_field(manifest, "manifest_path", "");
if strlength(string(resolvedManifestPath)) > 0
    [manifestDir, manifestName] = fileparts(char(string(resolvedManifestPath)));
    if endsWith(string(manifestName), ".export_manifest")
        stem = extractBefore(string(manifestName), strlength(string(manifestName)) - strlength(".export_manifest") + 1);
        candidate = string(fullfile(manifestDir, [char(stem) '.onnx']));
        if isfile(candidate)
            onnxPath = candidate;
            return;
        end
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
