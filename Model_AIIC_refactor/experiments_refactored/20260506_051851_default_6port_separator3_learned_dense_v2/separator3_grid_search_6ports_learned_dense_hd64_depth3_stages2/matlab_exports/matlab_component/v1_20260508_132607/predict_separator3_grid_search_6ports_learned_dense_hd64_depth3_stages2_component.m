function [outputData, ports, debug, component] = predict_separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2_component(inputData, componentOrDir)
%PREDICT_SEPARATOR3_GRID_SEARCH_6PORTS_LEARNED_DENSE_HD64_DEPTH3_STAGES2_COMPONENT Model-specific wrapper for separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2.
if nargin < 1 || isempty(inputData)
    inputData = randn(1, 24, 'single');
end
[outputData, debug, component] = predict_srs_ai_matlab_component(single(inputData), componentOrDir);
ports = split_srs_ai_matlab_ports(outputData);
end
