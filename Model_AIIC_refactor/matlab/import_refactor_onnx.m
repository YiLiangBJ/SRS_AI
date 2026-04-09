function net = import_refactor_onnx(exportDir)
%IMPORT_REFACTOR_ONNX Import an exported refactor ONNX model into Matlab.
%
% Usage:
%   net = import_refactor_onnx("experiments_refactored/.../onnx_exports/my_run")
%
% The export directory must contain:
%   - <run_name>.onnx
%   - export_manifest.json

manifestPath = fullfile(exportDir, "export_manifest.json");
manifest = jsondecode(fileread(manifestPath));

onnxPath = manifest.onnx_path;
if ~isfile(onnxPath)
    [~, onnxName, onnxExt] = fileparts(onnxPath);
    onnxPath = fullfile(exportDir, onnxName + onnxExt);
end

net = importNetworkFromONNX(onnxPath, OutputLayerType="regression");

disp("Imported ONNX model:");
disp("  Run: " + string(manifest.run_name));
disp("  Input layout: " + string(manifest.matlab_notes.input_layout));
disp("  Output layout: " + string(manifest.matlab_notes.output_layout));
disp("  Suggested input name: " + string(manifest.matlab_notes.input_name));
disp("  Suggested output name: " + string(manifest.matlab_notes.output_name));
end
