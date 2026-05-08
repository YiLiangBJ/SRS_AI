%% Step 0: choose the component folder
% Run this script from inside the copied component package, or edit componentDir.
componentDir = fileparts(mfilename('fullpath'));

%% Step 1: load the exported component and inspect metadata
component = load_srs_ai_matlab_component(componentDir);
manifest = component.manifest;
ioSpec = component.io_spec;
disp(manifest.run_name);
disp(ioSpec.input);
disp(ioSpec.output);

%% Step 2: inspect the exported reference tensors
sampleInput = single(component.weights.sample_input);
referenceOutput = single(component.weights.reference_output);
disp(size(sampleInput));
disp(size(referenceOutput));

%% Step 3: create your own dynamic-batch input (N x 24)
batchSize = 4;
inputData = prepare_refactor_input(component, batchSize, "bundle");
disp(size(inputData));

%% Step 4: run the off-the-shelf predictor
[outputData, debug] = predict_srs_ai_matlab_component(inputData, component);
disp(size(outputData));

%% Step 5: split the 6 ports into separate N x 24 matrices
ports = split_srs_ai_matlab_ports(outputData);
disp(size(ports{1}));

%% Step 6: verify the exported reference sample path
[referencePrediction, referenceDebug] = predict_srs_ai_matlab_component(sampleInput, component);
maxAbsDiff = max(abs(referencePrediction(:) - referenceOutput(:)));
disp("Max abs diff vs reference_output: " + string(maxAbsDiff));

%% Step 7: model-specific wrapper usage
[wrappedOutput, wrappedPorts, wrappedDebug] = predict_separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2_component(inputData, component);
disp(size(wrappedOutput));
%#ok<NASGU>
