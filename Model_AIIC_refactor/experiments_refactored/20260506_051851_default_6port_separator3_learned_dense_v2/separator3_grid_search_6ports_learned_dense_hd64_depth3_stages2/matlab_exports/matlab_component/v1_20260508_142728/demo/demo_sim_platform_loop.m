function [outputData, ports, state] = demo_sim_platform_loop(inputData, resetState)
%DEMO_SIM_PLATFORM_LOOP Template for first-slot init and later-slot reuse.
persistent cachedState
if nargin < 2
    resetState = false;
end
if resetState
    cachedState = [];
end
if isempty(cachedState)
    componentDir = fileparts(fileparts(mfilename('fullpath')));
    cachedState = init_model(componentDir);
end
[outputData, ports] = predict_model(cachedState, inputData);
state = cachedState;
end
