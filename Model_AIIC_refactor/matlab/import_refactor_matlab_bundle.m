function bundle = import_refactor_matlab_bundle(exportDir)
%IMPORT_REFACTOR_MATLAB_BUNDLE Load explicit Matlab bundle exported from a run.
%
% Usage:
%   bundle = import_refactor_matlab_bundle("matlab_exports/my_run")

exportDir = resolve_refactor_export_dir(exportDir);
manifestPath = fullfile(char(exportDir), 'matlab_model_bundle_manifest.json');
if ~isfile(manifestPath)
    error("import_refactor_matlab_bundle:ManifestNotFound", ...
    "matlab_model_bundle_manifest.json was not found under %s", char(exportDir));
end

manifest = jsondecode(fileread(manifestPath));
matPath = fullfile(char(exportDir), char(string(manifest.mat_file)));
if ~isfile(matPath)
    error("import_refactor_matlab_bundle:MatFileNotFound", ...
    "Matlab bundle MAT file was not found: %s", char(matPath));
end

weights = load(matPath);

bundle = struct();
bundle.export_dir = exportDir;
bundle.manifest = manifest;
bundle.weights = weights;

disp("Imported Matlab bundle:");
disp("  Run: " + string(manifest.run_name));
disp("  Model type: " + string(manifest.model_spec.model_type));
disp("  MAT file: " + string(manifest.mat_file));
disp("  Input layout: " + string(manifest.input_layout));
disp("  Output layout: " + string(manifest.output_layout));
end