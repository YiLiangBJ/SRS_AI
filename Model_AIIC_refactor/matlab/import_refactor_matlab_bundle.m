function bundle = import_refactor_matlab_bundle(exportPath)
%IMPORT_REFACTOR_MATLAB_BUNDLE Load explicit Matlab bundle exported from a run.
%
% Usage:
%   bundle = import_refactor_matlab_bundle(".../<run_name>/matlab_exports")
%   bundle = import_refactor_matlab_bundle(".../<run_name>/matlab_model_bundle.mat")
%   bundle = import_refactor_matlab_bundle(".../<run_name>/matlab_model_bundle_manifest.json")
%
% For direct .mat file input, this helper expects the sibling manifest file
% matlab_model_bundle_manifest.json to be present in the same directory.

[exportDir, manifestPath, explicitMatPath] = local_resolve_bundle_targets(exportPath);
if ~isfile(manifestPath)
    error("import_refactor_matlab_bundle:ManifestNotFound", ...
    "Could not find a Matlab bundle manifest for %s", char(string(exportPath)));
end

manifest = jsondecode(fileread(manifestPath));
manifest.manifest_path = string(manifestPath);
matPath = local_resolve_mat_path(exportDir, manifest, explicitMatPath);
if ~isfile(matPath)
    error("import_refactor_matlab_bundle:MatFileNotFound", ...
    "Matlab bundle MAT file was not found: %s", char(matPath));
end

weights = load(matPath);
ioSpec = describe_refactor_model_io(manifest, "bundle", false);

bundle = struct();
bundle.export_dir = exportDir;
bundle.manifest = manifest;
bundle.manifest.mat_path = string(matPath);
bundle.weights = weights;
bundle.io_spec = ioSpec;

disp("Imported Matlab bundle:");
disp("  Run: " + string(manifest.run_name));
disp("  Model type: " + string(manifest.model_spec.model_type));
disp("  Energy normalization: " + local_bool_string(local_manifest_bool(manifest.model_spec, "normalize_energy", false)));
disp("  MAT file: " + string(matPath));
disp("  Input layout: " + string(manifest.input_layout));
disp("  Output layout: " + string(manifest.output_layout));
disp("  Input shape: " + string(ioSpec.input.shape_string));
disp("  Output shape: " + string(ioSpec.output.shape_string));
disp("  Input dynamic dims: [1]");
disp("  Output dynamic dims: [1]");
end

function [exportDir, manifestPath, explicitMatPath] = local_resolve_bundle_targets(exportPath)
explicitMatPath = "";
exportPath = string(exportPath);
exportPath = strip(exportPath);

if strlength(exportPath) == 0
    error("import_refactor_matlab_bundle:EmptyPath", "exportPath must not be empty.");
end

if isfile(char(exportPath))
    [parentDir, ~, ext] = fileparts(char(exportPath));
    exportDir = string(char(java.io.File(parentDir).getCanonicalPath()));
    ext = string(lower(ext));

    if ext == ".mat"
        explicitMatPath = string(char(java.io.File(char(exportPath)).getCanonicalPath()));
        manifestPath = string(fullfile(char(exportDir), 'matlab_model_bundle_manifest.json'));
        return;
    end

    if ext == ".json"
        manifestPath = string(char(java.io.File(char(exportPath)).getCanonicalPath()));
        return;
    end

    error("import_refactor_matlab_bundle:UnsupportedFilePath", ...
        "Unsupported file input: %s", char(exportPath));
end

exportDir = resolve_refactor_export_dir(exportPath);
manifestPath = string(fullfile(char(exportDir), 'matlab_model_bundle_manifest.json'));
end

function matPath = local_resolve_mat_path(exportDir, manifest, explicitMatPath)
if nargin >= 3 && strlength(explicitMatPath) > 0 && isfile(char(explicitMatPath))
    matPath = explicitMatPath;
    return;
end

if isfield(manifest, "mat_path")
    candidate = string(manifest.mat_path);
    if isfile(candidate)
        matPath = candidate;
        return;
    end
end

if isfield(manifest, "mat_file")
    candidate = string(fullfile(char(exportDir), char(string(manifest.mat_file))));
    if isfile(candidate)
        matPath = candidate;
        return;
    end
end

matPath = string(fullfile(char(exportDir), 'matlab_model_bundle.mat'));
end

function value = local_manifest_bool(structValue, fieldName, defaultValue)
if isstruct(structValue) && isfield(structValue, fieldName)
    value = logical(structValue.(fieldName));
else
    value = logical(defaultValue);
end
end

function text = local_bool_string(value)
if value
    text = "enabled";
else
    text = "disabled";
end
end