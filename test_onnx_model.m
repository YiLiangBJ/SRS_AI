% test_onnx_model.m
% Quick test script for ONNX model in MATLAB
%
% Usage:
%   test_onnx_model('model.onnx')

function test_onnx_model(model_path)
    if nargin < 1
        model_path = 'model.onnx';
    end
    
    fprintf('========================================\n');
    fprintf('ONNX Model Test (Opset 9)\n');
    fprintf('========================================\n\n');
    
    %% Step 1: Import ONNX model
    fprintf('Step 1: Importing ONNX model...\n');
    if ~exist(model_path, 'file')
        error('Model file not found: %s\nPlease export model first:\n  python Model_AIIC_onnx/export_onnx.py --checkpoint <path> --output %s --opset 9', ...
              model_path, model_path);
    end
    
    try
        net = importONNXNetwork(model_path, 'OutputLayerType', 'regression');
        fprintf('  ✓ Model imported successfully!\n\n');
    catch ME
        fprintf('  ✗ Import failed: %s\n\n', ME.message);
        fprintf('Common issues:\n');
        fprintf('  1. Make sure model was exported with Opset 9\n');
        fprintf('  2. Check MATLAB version (R2020b+ required)\n');
        fprintf('  3. Verify Deep Learning Toolbox is installed\n\n');
        rethrow(ME);
    end
    
    %% Step 2: Generate test data
    fprintf('Step 2: Generating test data...\n');
    L = 12;  % Sequence length (adjust if your model uses different)
    
    % Generate complex input signal
    y_complex = randn(1, L) + 1i*randn(1, L);
    
    % Convert to real stacked format [real; imag]
    y_stacked = [real(y_complex), imag(y_complex)];  % (1, L*2)
    
    fprintf('  Input shape: (%d, %d)\n', size(y_stacked));
    fprintf('  Input energy: %.6f\n\n', sqrt(mean(abs(y_complex).^2)));
    
    %% Step 3: Energy normalization (IMPORTANT!)
    fprintf('Step 3: Energy normalization...\n');
    y_energy = sqrt(mean(abs(y_complex).^2));
    y_normalized = y_stacked / y_energy;
    
    fprintf('  ⚠️  Original energy: %.6f\n', y_energy);
    fprintf('  ⚠️  Normalized energy: %.6f\n', sqrt(mean(y_normalized.^2)));
    fprintf('  ✓ Normalization complete\n\n');
    
    %% Step 4: Run inference
    fprintf('Step 4: Running inference...\n');
    tic;
    h_stacked = predict(net, y_normalized);
    inference_time = toc;
    
    fprintf('  ✓ Inference complete! (%.2f ms)\n', inference_time * 1000);
    fprintf('  Output shape: (%d, %d, %d)\n\n', size(h_stacked));
    
    %% Step 5: Restore energy (IMPORTANT!)
    fprintf('Step 5: Restoring energy...\n');
    h_stacked = h_stacked * y_energy;
    fprintf('  ✓ Energy restored\n\n');
    
    %% Step 6: Convert to complex
    fprintf('Step 6: Converting to complex format...\n');
    P = size(h_stacked, 2);  % Number of ports
    
    h_real = h_stacked(:, :, 1:L);
    h_imag = h_stacked(:, :, L+1:end);
    h_complex = complex(h_real, h_imag);  % (1, P, L)
    
    fprintf('  Output shape: (%d, %d, %d)\n', size(h_complex));
    fprintf('  Number of ports: %d\n\n', P);
    
    % Display energy per port
    fprintf('  Energy per port:\n');
    for p = 1:P
        port_data = squeeze(h_complex(:, p, :));
        port_energy = sqrt(mean(abs(port_data).^2));
        fprintf('    Port %d: %.6f\n', p, port_energy);
    end
    fprintf('\n');
    
    %% Step 7: Verify reconstruction
    fprintf('Step 7: Verifying reconstruction...\n');
    
    % Sum all ports to reconstruct input
    y_recon_complex = squeeze(sum(h_complex, 2));  % (1, L)
    
    % Calculate reconstruction error
    recon_error = norm(y_complex - y_recon_complex) / norm(y_complex);
    recon_error_db = 10 * log10(recon_error^2);
    
    fprintf('  Reconstruction error: %.2e (%.2f%%)\n', recon_error, recon_error * 100);
    fprintf('  Reconstruction error (dB): %.2f dB\n', recon_error_db);
    
    if recon_error < 0.01
        fprintf('  ✓ Excellent reconstruction quality!\n\n');
    elseif recon_error < 0.05
        fprintf('  ✓ Good reconstruction quality\n\n');
    elseif recon_error < 0.20
        fprintf('  ⚠️  Moderate reconstruction quality\n');
        fprintf('     (Model may need more training)\n\n');
    else
        fprintf('  ⚠️  Poor reconstruction quality\n');
        fprintf('     (Model is likely untrained or incorrectly loaded)\n\n');
    end
    
    %% Step 8: Visualization
    fprintf('Step 8: Generating visualization...\n');
    
    figure('Name', 'ONNX Model Test Results', 'Position', [100 100 1400 600]);
    
    % Plot 1: Input signal
    subplot(2, 3, 1);
    plot(1:L, real(y_complex), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
    plot(1:L, imag(y_complex), 'r--s', 'LineWidth', 1.5, 'MarkerSize', 6);
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
        port_data = squeeze(h_complex(:, p, :));
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
    
    % Plot 3: Reconstruction comparison (real)
    subplot(2, 3, 3);
    plot(1:L, real(y_complex), 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
    plot(1:L, real(y_recon_complex), 'r--x', 'LineWidth', 1.5, 'MarkerSize', 8);
    grid on;
    title('Reconstruction (Real Part)', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Time Index');
    ylabel('Amplitude');
    legend('Original', 'Reconstructed', 'Location', 'best');
    xlim([0.5 L+0.5]);
    
    % Plot 4: Reconstruction comparison (imag)
    subplot(2, 3, 4);
    plot(1:L, imag(y_complex), 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
    plot(1:L, imag(y_recon_complex), 'r--x', 'LineWidth', 1.5, 'MarkerSize', 8);
    grid on;
    title('Reconstruction (Imaginary Part)', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Time Index');
    ylabel('Amplitude');
    legend('Original', 'Reconstructed', 'Location', 'best');
    xlim([0.5 L+0.5]);
    
    % Plot 5: Energy distribution
    subplot(2, 3, 5);
    port_energies = zeros(1, P);
    for p = 1:P
        port_data = squeeze(h_complex(:, p, :));
        port_energies(p) = sqrt(mean(abs(port_data).^2));
    end
    bar(1:P, port_energies, 'FaceColor', [0.2 0.6 0.8]);
    grid on;
    title('Energy Distribution', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Port Index');
    ylabel('Energy');
    xlim([0.5 P+0.5]);
    xticks(1:P);
    
    % Plot 6: Reconstruction error per sample
    subplot(2, 3, 6);
    error_per_sample = abs(y_complex - y_recon_complex);
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
    sgtitle(sprintf('ONNX Model Test - Error: %.2f%% (%.2f dB)', ...
            recon_error * 100, recon_error_db), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    fprintf('  ✓ Visualization generated\n\n');
    
    %% Summary
    fprintf('========================================\n');
    fprintf('Summary\n');
    fprintf('========================================\n');
    fprintf('Model:              %s\n', model_path);
    fprintf('Sequence length:    %d\n', L);
    fprintf('Number of ports:    %d\n', P);
    fprintf('Inference time:     %.2f ms\n', inference_time * 1000);
    fprintf('Reconstruction err: %.2f%% (%.2f dB)\n', recon_error * 100, recon_error_db);
    fprintf('========================================\n');
    fprintf('✓ Test complete!\n\n');
    
    %% Optional: Save results
    % Uncomment to save results
    % save('onnx_test_results.mat', 'y_complex', 'h_complex', 'y_recon_complex', 'recon_error');
    % fprintf('Results saved to: onnx_test_results.mat\n');
end
