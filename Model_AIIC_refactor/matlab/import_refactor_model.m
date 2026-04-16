function modelHandle = import_refactor_model(exportDir, mode)
%IMPORT_REFACTOR_MODEL Unified Matlab import entry for ONNX or Matlab bundle.
%
% Usage:
%   modelHandle = import_refactor_model("path/to/artifact")
%   modelHandle = import_refactor_model("path/to/artifact", "onnx")
%   modelHandle = import_refactor_model("path/to/artifact", "bundle")

if nargin < 2 || isempty(mode)
    mode = "auto";
end

mode = string(mode);

resolvedMode = local_resolve_mode(exportDir, mode);
modelHandle = struct();
modelHandle.mode = resolvedMode;

switch resolvedMode
    case "onnx"
        [net, manifest] = import_refactor_onnx(exportDir);
        modelHandle.model = net;
        modelHandle.manifest = manifest;
        modelHandle.export_dir = string(fileparts(char(manifest.manifest_path)));
    case "bundle"
        bundle = import_refactor_matlab_bundle(exportDir);
        modelHandle.model = bundle;
        modelHandle.manifest = bundle.manifest;
            modelHandle.export_dir = bundle.export_dir;
    otherwise
        error("import_refactor_model:UnsupportedMode", "Unsupported mode: %s", resolvedMode);
end

modelHandle.io_spec = describe_refactor_model_io(modelHandle.manifest, modelHandle.mode, false);
modelHandle.prepare_input = @(inputData) prepare_refactor_input(modelHandle, inputData, modelHandle.mode);
modelHandle.predict = @(inputData) predict_refactor_model(modelHandle, inputData);
end

function resolvedMode = local_resolve_mode(exportDir, mode)
if mode ~= "auto"
    resolvedMode = mode;
    return;
end

exportPath = string(exportDir);
if isfile(char(exportPath))
    [~, name, ext] = fileparts(char(exportPath));
    ext = string(lower(ext));
    if ext == ".onnx"
        resolvedMode = "onnx";
        return;
    end
    if ext == ".mat"
        resolvedMode = "bundle";
        return;
    end
    if ext == ".json"
        if strcmpi(strcat(name, char(ext)), 'matlab_model_bundle_manifest.json')
            resolvedMode = "bundle";
        else
            resolvedMode = "onnx";
        end
        return;
    end
end

exportDir = resolve_refactor_export_dir(exportDir);

onnxManifest = fullfile(char(exportDir), 'export_manifest.json');
matlabManifest = fullfile(char(exportDir), 'matlab_model_bundle_manifest.json');

if isfile(onnxManifest) || ~isempty(dir(fullfile(char(exportDir), '*.export_manifest.json')))
    resolvedMode = "onnx";
elseif isfile(matlabManifest)
    resolvedMode = "bundle";
else
    error("import_refactor_model:NoKnownManifest", ...
    "Could not detect ONNX or Matlab bundle manifest under %s", char(exportDir));
end
end