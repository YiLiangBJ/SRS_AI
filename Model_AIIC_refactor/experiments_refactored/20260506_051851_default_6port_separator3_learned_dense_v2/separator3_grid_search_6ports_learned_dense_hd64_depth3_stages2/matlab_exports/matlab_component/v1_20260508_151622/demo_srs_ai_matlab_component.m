function [inputData, outputData, debug, component] = demo_srs_ai_matlab_component(batchSize)
%DEMO_SRS_AI_MATLAB_COMPONENT Quick self-test for the colocated component package.
if nargin < 1 || isempty(batchSize)
    batchSize = 4;
end
component = init_model(fileparts(mfilename('fullpath')));
inputData = prepare_refactor_input(component, batchSize, "bundle");
[outputData, debug] = predict_srs_ai_matlab_component(inputData, component);
disp("Component demo finished.");
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));
end
