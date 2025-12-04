% Load ONNX model
net = importONNXNetwork('model.onnx', 'OutputLayerType', 'regression');

% Prepare input (convert complex to [real; imag])
y = randn(1, 12) + 1i*randn(1, 12);  % Complex signal
y_real_imag = [real(y), imag(y)];  % Convert to real stacked

% Predict
h_real_imag = predict(net, y_real_imag);

% Convert back to complex
L = 12; P = 4;
h_real = h_real_imag(:, :, 1:L);
h_imag = h_real_imag(:, :, L+1:end);
h = complex(h_real, h_imag);  % (1, P, L)