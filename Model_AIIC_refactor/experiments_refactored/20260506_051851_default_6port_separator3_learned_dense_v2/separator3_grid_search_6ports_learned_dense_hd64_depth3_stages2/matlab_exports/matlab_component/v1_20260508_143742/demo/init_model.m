function state = init_model(componentDir)
%INIT_MODEL Demo-local wrapper so scripts under demo/ run directly.
if nargin < 1 || isempty(componentDir)
    componentDir = fileparts(fileparts(mfilename('fullpath')));
end
addpath(componentDir);
state = feval('init_model', componentDir);
end
