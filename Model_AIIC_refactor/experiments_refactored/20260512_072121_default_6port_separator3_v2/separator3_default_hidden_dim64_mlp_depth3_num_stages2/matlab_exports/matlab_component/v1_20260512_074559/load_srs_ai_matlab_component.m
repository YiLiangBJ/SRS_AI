function component = load_srs_ai_matlab_component(componentDir)
%LOAD_SRS_AI_MATLAB_COMPONENT Load the colocated SRS AI Matlab bundle component.
if nargin < 1 || isempty(componentDir)
    componentDir = fileparts(mfilename('fullpath'));
end
component = import_refactor_matlab_bundle(componentDir);
end
