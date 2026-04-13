function resolvedDir = resolve_refactor_export_dir(exportDir)
%RESOLVE_REFACTOR_EXPORT_DIR Resolve export paths robustly across platforms.
%
% Supports:
%   - absolute paths
%   - paths relative to current working directory
%   - paths relative to the repository root, such as
%     ./Model_AIIC_refactor/experiments_refactored/...

exportDir = string(exportDir);
exportDir = strip(exportDir);

if strlength(exportDir) == 0
    error("resolve_refactor_export_dir:EmptyPath", "exportDir must not be empty.");
end

if isfolder(char(exportDir))
    resolvedDir = string(char(java.io.File(char(exportDir)).getCanonicalPath()));
    return;
end

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));

normalized = replace(exportDir, "\\", "/");
relativeFromRepo = normalized;
if startsWith(relativeFromRepo, "./")
    relativeFromRepo = extractAfter(relativeFromRepo, 2);
end

candidatePaths = [ ...
    fullfile(repoRoot, char(relativeFromRepo)); ...
    fullfile(thisDir, char(relativeFromRepo)) ...
    ];

for idx = 1:numel(candidatePaths)
    candidate = candidatePaths{idx};
    if isfolder(candidate)
        resolvedDir = string(char(java.io.File(candidate).getCanonicalPath()));
        return;
    end
end

error("resolve_refactor_export_dir:DirectoryNotFound", ...
    "Could not resolve exportDir: %s\nChecked: %s\nChecked: %s", ...
    char(exportDir), candidatePaths{1}, candidatePaths{2});
end