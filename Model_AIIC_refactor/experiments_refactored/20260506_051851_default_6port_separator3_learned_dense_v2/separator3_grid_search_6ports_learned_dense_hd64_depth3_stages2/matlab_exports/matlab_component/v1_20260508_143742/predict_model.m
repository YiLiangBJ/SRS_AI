function [outputData, ports, debug] = predict_model(state, inputData)
%PREDICT_MODEL Fast deployed inference using preinitialized state.
[outputData, debug] = predict_refactor_matlab_bundle(state, single(inputData));
if nargout >= 2
    ports = split_ports(outputData);
end
end
