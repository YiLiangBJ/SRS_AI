function modelHandle = import_refactor_model(exportDir, mode)
%IMPORT_REFACTOR_MODEL Unified Matlab import entry for ONNX or Matlab bundle.
%
% Usage:
%   modelHandle = import_refactor_model("path/to/export")
%   modelHandle = import_refactor_model("path/to/export", "onnx")
%   modelHandle = import_refactor_model("path/to/export", "bundle")

if nargin < 2 || isempty(mode)
    mode = "auto";
end

exportDir = resolve_refactor_export_dir(exportDir);
mode = string(mode);

resolvedMode = local_resolve_mode(exportDir, mode);
modelHandle = struct();
modelHandle.mode = resolvedMode;
modelHandle.export_dir = exportDir;

switch resolvedMode
    case "onnx"
        [net, manifest] = import_refactor_onnx(exportDir);
        modelHandle.model = net;
        modelHandle.manifest = manifest;
    case "bundle"
        bundle = import_refactor_matlab_bundle(exportDir);
        modelHandle.model = bundle;
        modelHandle.manifest = bundle.manifest;
    otherwise
        error("import_refactor_model:UnsupportedMode", "Unsupported mode: %s", resolvedMode);
end
end

function resolvedMode = local_resolve_mode(exportDir, mode)
if mode ~= "auto"
    resolvedMode = mode;
    return;
end

onnxManifest = fullfile(char(exportDir), 'export_manifest.json');
matlabManifest = fullfile(char(exportDir), 'matlab_model_bundle_manifest.json');

if isfile(onnxManifest)
    resolvedMode = "onnx";
elseif isfile(matlabManifest)
    resolvedMode = "bundle";
else
    error("import_refactor_model:NoKnownManifest", ...
    "Could not detect ONNX or Matlab bundle manifest under %s", char(exportDir));
end
end