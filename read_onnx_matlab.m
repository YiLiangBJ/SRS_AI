% read_onnx_matlab.m
% MATLAB script for loading and testing ONNX model (Opset 9 compatible)
%
% This script demonstrates:
% 1. Loading MATLAB-compatible ONNX model
% 2. Preparing input data (complex -> real stacked)
% 3. Energy normalization (done in MATLAB, not in model)
% 4. Running inference
% 5. Converting output back to complex
% 6. Visualizing results

%% Configuration
fprintf('========================================\n');
fprintf('ONNX Model Test (MATLAB Compatible)\n');
fprintf('========================================\n\n');

% Model parameters (adjust to match your model)
L = 12;  % Sequence length
P = 4;   % Number of ports

% ONNX model file
model_file = 'model_matlab.onnx';

%% Check if model exists
if ~exist(model_file, 'file')
    error('ONNX model not found: %s\nPlease run: python Model_AIIC_onnx/export_onnx_matlab.py --checkpoint <path> --output %s', ...
          model_file, model_file);
end

%% Load ONNX model
fprintf('Loading ONNX model: %s\n', model_file);
try
    net = importONNXNetwork(model_file, 'OutputLayerType', 'regression');
    fprintf('✓ Model loaded successfully!\n\n');
catch ME
    fprintf('✗ Failed to load model:\n');
    fprintf('  %s\n\n', ME.message);
    fprintf('Common issues:\n');
    fprintf('  1. Make sure you exported with export_onnx_matlab.py (Opset 9)\n');
    fprintf('  2. Check MATLAB version (R2020b+ recommended)\n');
    fprintf('  3. Verify Deep Learning Toolbox is installed\n');
    rethrow(ME);
end

%% Generate test data
fprintf('Generating test data...\n');

% Complex input signal (simulating received SRS signal)
y = randn(1, L) + 1i*randn(1, L);

% Convert to real stacked format: [real; imag]
y_stacked = [real(y), imag(y)];  % (1, 24)

fprintf('  Input shape: (1, %d)\n', size(y_stacked, 2));
fprintf('  Complex values: (%d)\n\n', L);

%% Energy normalization (IMPORTANT: Done in MATLAB, not in model)
fprintf('⚠️  Performing energy normalization...\n');

% Calculate energy
y_energy = sqrt(mean(abs(y).^2));
fprintf('  Original energy: %.6f\n', y_energy);

% Normalize
y_normalized = y_stacked / y_energy;
fprintf('  Normalized energy: %.6f\n', sqrt(mean(y_normalized.^2)));
fprintf('  ✓ Input normalized\n\n');

%% Inference
fprintf('Running inference...\n');
tic;
h_flat = predict(net, y_normalized);  % (1, P*L*2) - flattened output
inference_time = toc;
fprintf('  ✓ Inference complete! (%.2f ms)\n', inference_time * 1000);
fprintf('  Output shape (flat): (%d, %d)\n\n', size(h_flat));

%% Reshape output (IMPORTANT: Output is flattened)
fprintf('⚠️  Reshaping output...\n');
h_stacked = reshape(h_flat, [1, P, L*2]);
fprintf('  Reshaped to: (%d, %d, %d)\n', size(h_stacked));
fprintf('  ✓ Reshape complete\n\n');

%% Restore energy (IMPORTANT: Must restore after inference)
fprintf('⚠️  Restoring energy...\n');
h_stacked = h_stacked * y_energy;
fprintf('  ✓ Energy restored\n\n');

%% Convert back to complex
fprintf('Converting to complex format...\n');
h_real = h_stacked(:, :, 1:L);
h_imag = h_stacked(:, :, L+1:end);
h = complex(h_real, h_imag);  % (1, P, L)

fprintf('  Output shape: (%d, %d, %d)\n', size(h));
fprintf('  ✓ Conversion complete\n\n');

%% Analyze results
fprintf('Analyzing separated channels...\n');
fprintf('  Energy per port:\n');
for p = 1:P
    port_data = squeeze(h(:, p, :));
    port_energy = sqrt(mean(abs(port_data).^2));
    fprintf('    Port %d: %.6f\n', p, port_energy);
end
fprintf('\n');

%% Verify reconstruction
fprintf('Verifying reconstruction...\n');

% Sum all ports to reconstruct input
y_recon = squeeze(sum(h, 2));  % (1, L)

% Calculate reconstruction error
recon_error = norm(y - y_recon) / norm(y);
recon_error_db = 10 * log10(recon_error^2);

fprintf('  Reconstruction error: %.6e (%.2f%%)\n', recon_error, recon_error * 100);
fprintf('  Reconstruction error (dB): %.2f dB\n', recon_error_db);

if recon_error < 0.1
    fprintf('  ✓ Good reconstruction!\n\n');
elseif recon_error < 0.5
    fprintf('  ⚠️  Moderate reconstruction quality\n\n');
else
    fprintf('  ✗ Poor reconstruction - model may need more training\n\n');
end

%% Visualization
fprintf('Generating plots...\n');

figure('Name', 'ONNX Model Test Results', 'Position', [100 100 1400 600]);

% Plot 1: Input signal
subplot(2, 3, 1);
plot(1:L, real(y), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
plot(1:L, imag(y), 'r--s', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
title('Input Signal (y)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time Index');
ylabel('Amplitude');
legend('Real', 'Imaginary', 'Location', 'best');
xlim([0.5 L+0.5]);

% Plot 2: Separated channels (magnitude)
subplot(2, 3, 2);
colors = lines(P);
for p = 1:P
    port_data = squeeze(h(:, p, :));
    plot(1:L, abs(port_data), '-o', 'Color', colors(p,:), ...
         'LineWidth', 1.5, 'MarkerSize', 6); hold on;
end
grid on;
title('Separated Channels (Magnitude)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time Index');
ylabel('Magnitude');
legend_str = arrayfun(@(p) sprintf('Port %d', p), 1:P, 'UniformOutput', false);
legend(legend_str, 'Location', 'best');
xlim([0.5 L+0.5]);

% Plot 3: Reconstruction comparison (real part)
subplot(2, 3, 3);
plot(1:L, real(y), 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(1:L, real(y_recon), 'r--x', 'LineWidth', 1.5, 'MarkerSize', 8);
grid on;
title('Reconstruction (Real Part)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time Index');
ylabel('Amplitude');
legend('Original', 'Reconstructed', 'Location', 'best');
xlim([0.5 L+0.5]);

% Plot 4: Reconstruction comparison (imaginary part)
subplot(2, 3, 4);
plot(1:L, imag(y), 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(1:L, imag(y_recon), 'r--x', 'LineWidth', 1.5, 'MarkerSize', 8);
grid on;
title('Reconstruction (Imaginary Part)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time Index');
ylabel('Amplitude');
legend('Original', 'Reconstructed', 'Location', 'best');
xlim([0.5 L+0.5]);

% Plot 5: Channel energy distribution
subplot(2, 3, 5);
port_energies = zeros(1, P);
for p = 1:P
    port_data = squeeze(h(:, p, :));
    port_energies(p) = sqrt(mean(abs(port_data).^2));
end
bar(1:P, port_energies, 'FaceColor', [0.2 0.6 0.8]);
grid on;
title('Energy Distribution', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Port Index');
ylabel('Energy');
xlim([0.5 P+0.5]);
xticks(1:P);

% Plot 6: Reconstruction error
subplot(2, 3, 6);
error_per_sample = abs(y - y_recon);
plot(1:L, error_per_sample, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
title('Reconstruction Error', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time Index');
ylabel('Error Magnitude');
xlim([0.5 L+0.5]);
text(L/2, max(error_per_sample)*0.9, ...
     sprintf('RMSE: %.2e', recon_error), ...
     'HorizontalAlignment', 'center', ...
     'FontSize', 10, 'FontWeight', 'bold');

% Overall title
sgtitle(sprintf('ONNX Model Test - Reconstruction Error: %.2f%% (%.2f dB)', ...
        recon_error * 100, recon_error_db), ...
        'FontSize', 14, 'FontWeight', 'bold');

fprintf('  ✓ Plots generated\n\n');

%% Summary
fprintf('========================================\n');
fprintf('Summary\n');
fprintf('========================================\n');
fprintf('Model:              %s\n', model_file);
fprintf('Sequence length:    %d\n', L);
fprintf('Number of ports:    %d\n', P);
fprintf('Inference time:     %.2f ms\n', inference_time * 1000);
fprintf('Reconstruction err: %.2f%% (%.2f dB)\n', recon_error * 100, recon_error_db);
fprintf('========================================\n');
fprintf('✓ Test complete!\n\n');

%% Export results (optional)
% Uncomment to save results
% save('onnx_test_results.mat', 'y', 'h', 'y_recon', 'recon_error');
% fprintf('Results saved to: onnx_test_results.mat\n');
