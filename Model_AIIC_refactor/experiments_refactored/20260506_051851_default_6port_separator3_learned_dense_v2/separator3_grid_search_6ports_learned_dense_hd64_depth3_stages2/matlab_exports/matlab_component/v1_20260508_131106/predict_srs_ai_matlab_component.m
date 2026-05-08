function [outputData, debug, component] = predict_srs_ai_matlab_component(inputData, componentOrDir)
%PREDICT_SRS_AI_MATLAB_COMPONENT Off-the-shelf inference entrypoint for separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2.
%
% Usage:
%   outputData = predict_srs_ai_matlab_component(inputData)
%   outputData = predict_srs_ai_matlab_component(inputData, component)
%   outputData = predict_srs_ai_matlab_component(inputData, componentDir)
%
% Input shape:
%   N x 24 real-stacked float32 = [real_part, imag_part]
% Output shape:
%   N x 6 x 24 real-stacked float32
if nargin < 2 || isempty(componentOrDir)
    component = load_srs_ai_matlab_component(fileparts(mfilename('fullpath')));
elseif isstruct(componentOrDir)
    component = componentOrDir;
else
    component = load_srs_ai_matlab_component(componentOrDir);
end
[outputData, debug] = predict_refactor_matlab_bundle(component, inputData);
end
