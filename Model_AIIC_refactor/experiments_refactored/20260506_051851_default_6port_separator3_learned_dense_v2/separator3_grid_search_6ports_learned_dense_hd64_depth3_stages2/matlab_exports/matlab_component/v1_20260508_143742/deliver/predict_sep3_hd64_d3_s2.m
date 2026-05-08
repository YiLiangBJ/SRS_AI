function [outputData, ports, debug] = predict_sep3_hd64_d3_s2(state, inputData)
%PREDICT_SEP3_HD64_D3_S2 Short model-specific deployed inference entrypoint.
[outputData, ports, debug] = predict_model(state, inputData);
end
