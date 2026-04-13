function [modelHandle, inputData, outputData, info] = demo_refactor_model_inference(exportDir, mode, batchSize)
%DEMO_REFACTOR_MODEL_INFERENCE Unified inference demo for ONNX or Matlab bundle.
%
% Usage:
%   [modelHandle, inputData, outputData, info] = demo_refactor_model_inference("path/to/export")
%   [modelHandle, inputData, outputData, info] = demo_refactor_model_inference("path/to/export", "onnx", 2)
%   [modelHandle, inputData, outputData, info] = demo_refactor_model_inference("path/to/export", "bundle", 2)

if nargin < 2 || isempty(mode)
    mode = "auto";
end
if nargin < 3
    batchSize = [];
end

modelHandle = import_refactor_model(exportDir, mode);
info = struct();
info.mode = modelHandle.mode;
info.manifest = modelHandle.manifest;

switch modelHandle.mode
    case "onnx"
        [net, inputData, outputData, manifest] = demo_refactor_onnx_inference(exportDir, batchSize);
        modelHandle.model = net;
        modelHandle.manifest = manifest;
        info.manifest = manifest;
    case "bundle"
        bundle = modelHandle.model;
        if nargin < 3 || isempty(batchSize)
            inputData = single(bundle.weights.sample_input);
        else
            inputData = randn(batchSize, double(bundle.manifest.model_spec.seq_len) * 2, "single");
        end
        [outputData, debug] = predict_refactor_matlab_bundle(bundle, inputData);
        modelHandle.model = bundle;
        info.debug = debug;
    otherwise
        error("demo_refactor_model_inference:UnsupportedMode", "Unsupported mode: %s", modelHandle.mode);
end

disp("Unified Matlab demo finished.");
disp("  Mode: " + string(modelHandle.mode));
disp("  Input size: " + mat2str(size(inputData)));
disp("  Output size: " + mat2str(size(outputData)));
disp("  Run name: " + string(modelHandle.manifest.run_name));
end