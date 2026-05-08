%% Quick start: one-time init, then inference
componentDir = fileparts(fileparts(mfilename('fullpath')));
addpath(componentDir);
state = init_model(componentDir);
inputData = randn(8, 24, 'single');
[outputData, debug] = predict_model(state, inputData);
disp(size(inputData));
disp(size(outputData));
%#ok<NASGU>
