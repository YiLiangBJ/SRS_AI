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

onnxPath = string(manifest.onnx_path);
if ~isfile(onnxPath)
    [~, onnxName, onnxExt] = fileparts(char(onnxPath));
    onnxPath = fullfile(char(exportDir), onnxName + onnxExt);
end
if ~isfile(onnxPath)
    error("import_refactor_onnx:OnnxNotFound", ...
        "ONNX file referenced by the manifest was not found: %s", char(onnxPath));
end

net = importNetworkFromONNX(char(onnxPath), OutputLayerType="regression");

disp("Imported ONNX model:");
disp("  Run: " + string(manifest.run_name));
disp("  ONNX path: " + string(onnxPath));
disp("  Input layout: " + string(manifest.matlab_notes.input_layout));
disp("  Output layout: " + string(manifest.matlab_notes.output_layout));
disp("  Suggested input name: " + string(manifest.matlab_notes.input_name));
disp("  Suggested output name: " + string(manifest.matlab_notes.output_name));
end
