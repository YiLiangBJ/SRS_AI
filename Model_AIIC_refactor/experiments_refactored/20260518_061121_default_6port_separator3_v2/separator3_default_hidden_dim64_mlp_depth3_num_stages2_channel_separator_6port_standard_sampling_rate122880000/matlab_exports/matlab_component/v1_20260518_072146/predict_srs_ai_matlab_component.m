function [outputData, debug, state] = predict_srs_ai_matlab_component(inputData, stateOrDir)
%PREDICT_SRS_AI_MATLAB_COMPONENT Lower-level inference entrypoint for separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000.
%
% Usage:
%   outputData = predict_srs_ai_matlab_component(inputData, state)
%   outputData = predict_srs_ai_matlab_component(inputData, componentDir)
%
% Input shape:
%   N x 24 real-stacked float32 = [real_part, imag_part]
% Output shape:
%   N x 6 x 24 real-stacked float32
if nargin < 2 || isempty(stateOrDir)
    state = init_model(fileparts(mfilename('fullpath')));
elseif isstruct(stateOrDir)
    state = stateOrDir;
else
    state = init_model(stateOrDir);
end
[outputData, debug] = predict_refactor_matlab_bundle(state, inputData);
end
