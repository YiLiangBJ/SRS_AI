function [outputData, ports, debug] = predict_model(state, inputData)
%PREDICT_MODEL Demo-local wrapper so scripts under demo/ run directly.
componentDir = fileparts(fileparts(mfilename('fullpath')));
addpath(componentDir);
[outputData, ports, debug] = feval('predict_model', state, inputData);
end
