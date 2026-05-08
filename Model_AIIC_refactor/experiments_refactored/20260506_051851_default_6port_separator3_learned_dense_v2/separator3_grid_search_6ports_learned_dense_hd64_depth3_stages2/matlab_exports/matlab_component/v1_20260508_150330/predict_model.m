function [outputData, debug] = predict_model(state, inputData)
%PREDICT_MODEL Fast deployed inference using preinitialized state.
[outputData, debug] = predict_refactor_matlab_bundle(state, single(inputData));
end
