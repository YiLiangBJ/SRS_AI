%% Step 1: locate the component package root
componentDir = fileparts(fileparts(mfilename('fullpath')));
addpath(componentDir);

%% Step 2: one-time initialization
state = init_model(componentDir);
manifest = state.manifest;
ioSpec = state.io_spec;
disp(manifest.run_name);
disp(ioSpec.input);
disp(ioSpec.output);

%% Step 3: inspect exported reference tensors
sampleInput = single(state.weights.sample_input);
referenceOutput = single(state.weights.reference_output);
disp(size(sampleInput));
disp(size(referenceOutput));

%% Step 4: create your own dynamic-batch input
batchSize = 4;
inputData = prepare_refactor_input(state, batchSize, "bundle");
disp(size(inputData));

%% Step 5: run deployed inference with preloaded state
[outputData, ports, debug] = predict_model(state, inputData);
disp(size(outputData));
disp(size(ports{1}));

%% Step 6: verify the reference sample path once
[referencePrediction, referencePorts, referenceDebug] = predict_model(state, sampleInput);
maxAbsDiff = max(abs(referencePrediction(:) - referenceOutput(:)));
disp("Max abs diff vs reference_output: " + string(maxAbsDiff));

%% Step 7: short model-specific aliases
state2 = init_sep3_hd64_d3_s2(componentDir);
[outputData2, ports2, debug2] = predict_sep3_hd64_d3_s2(state2, inputData);
disp(size(outputData2));
%#ok<NASGU>
