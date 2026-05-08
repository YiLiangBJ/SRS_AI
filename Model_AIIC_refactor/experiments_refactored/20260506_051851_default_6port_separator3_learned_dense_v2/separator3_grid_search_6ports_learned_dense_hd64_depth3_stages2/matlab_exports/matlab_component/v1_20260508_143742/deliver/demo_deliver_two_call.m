function demo_deliver_two_call()
%DEMO_DELIVER_TWO_CALL Minimal delivery demo showing init once, infer twice.
%
% This simulates a slot-based caller:
%   - first call: load manifest / weights and cache state
%   - second call: reuse the cached state directly
persistent state
componentDir = fileparts(mfilename('fullpath'));

inputA = randn(2, 24, 'single');
if isempty(state)
    disp("First call: state is empty, running init_model(...)");
    state = init_model(componentDir);
else
    disp("First call: state already exists");
end
[outputA, portsA] = predict_model(state, inputA);
disp("First call output size: " + mat2str(size(outputA)));
disp("First call port[1] size: " + mat2str(size(portsA{1})));

inputB = randn(3, 24, 'single');
if isempty(state)
    error("demo_deliver_two_call:MissingState", "State should already be initialized before second call.");
else
    disp("Second call: reusing cached state, skipping init_model(...)");
end
[outputB, portsB] = predict_model(state, inputB);
disp("Second call output size: " + mat2str(size(outputB)));
disp("Second call port[1] size: " + mat2str(size(portsB{1})));

[refOutput, refPorts] = predict_model(state, single(state.weights.sample_input));
maxAbsDiff = max(abs(refOutput(:) - single(state.weights.reference_output(:))));
disp("Reference max abs diff: " + string(maxAbsDiff));
end
