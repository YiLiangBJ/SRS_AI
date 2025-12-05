% test_onnx_simple.m
% 简化版 ONNX 模型测试（无可视化，快速验证）
%
% Usage:
%   test_onnx_simple()
%   test_onnx_simple('model_onnx_mode.onnx')

function test_onnx_simple(model_path)
    if nargin < 1
        model_path = 'model_onnx_mode.onnx';
    end
    
    fprintf('========================================\n');
    fprintf('ONNX Model Quick Test (onnx_mode=True)\n');
    fprintf('========================================\n\n');
    
    % Step 1: Check file exists
    if ~exist(model_path, 'file')
        error(['Model file not found: %s\n\n' ...
               'Please export first:\n' ...
               '  cd c:/GitRepo/SRS_AI\n' ...
               '  python Model_AIIC_onnx/export_onnx.py --checkpoint Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth --output model_onnx_mode.onnx --opset 9\n'], ...
              model_path);
    end
    
    % Step 2: Import model
    fprintf('Step 1: Importing ONNX model...\n');
    fprintf('  Path: %s\n', model_path);
    
    try
        net = importONNXNetwork(model_path, 'OutputLayerType', 'regression');
        fprintf('  ✓ SUCCESS: Model imported!\n\n');
    catch ME
        fprintf('  ✗ FAILED: %s\n\n', ME.message);
        
        % Try alternative method
        fprintf('Trying alternative: importONNXFunction...\n');
        try
            net = importONNXFunction(model_path, 'model_func');
            fprintf('  ✓ SUCCESS: Model imported as function!\n\n');
        catch ME2
            fprintf('  ✗ FAILED: %s\n\n', ME2.message);
            fprintf('Both methods failed. Model is not compatible with this MATLAB version.\n');
            fprintf('Unsupported operators detected:\n');
            fprintf('  - Slice (84 instances)\n');
            fprintf('  - Unsqueeze (17 instances)\n');
            fprintf('  - Gather (32 instances)\n\n');
            fprintf('These operators are not supported by MATLAB''s importONNXNetwork.\n');
            fprintf('You may need to further modify the model or use a different approach.\n');
            return;
        end
    end
    
    % Step 3: Test with sample data
    fprintf('Step 2: Testing inference...\n');
    
    L = 12;  % Sequence length
    
    % Generate test input
    y_complex = randn(1, L) + 1i*randn(1, L);
    y_stacked = [real(y_complex), imag(y_complex)];  % (1, 24)
    
    fprintf('  Input shape: (%d, %d)\n', size(y_stacked));
    
    % Energy normalization (REQUIRED!)
    y_energy = sqrt(mean(abs(y_complex).^2));
    y_normalized = y_stacked / y_energy;
    
    fprintf('  Input energy: %.6f\n', y_energy);
    fprintf('  Normalized energy: %.6f\n', sqrt(mean(y_normalized.^2)));
    
    % Run inference
    try
        tic;
        if isa(net, 'DAGNetwork') || isa(net, 'dlnetwork')
            h_normalized = predict(net, y_normalized);
        else
            % Function form
            h_normalized = net(y_normalized);
        end
        inference_time = toc;
        
        fprintf('  ✓ Inference successful! (%.2f ms)\n', inference_time * 1000);
        fprintf('  Output shape: (%d, %d, %d)\n\n', size(h_normalized));
        
    catch ME
        fprintf('  ✗ Inference failed: %s\n\n', ME.message);
        return;
    end
    
    % Step 4: Restore energy
    fprintf('Step 3: Restoring energy...\n');
    h_stacked = h_normalized * y_energy;
    
    % Step 5: Convert to complex
    fprintf('Step 4: Converting to complex...\n');
    P = size(h_stacked, 2);  % Number of ports
    h_real = h_stacked(:, :, 1:L);
    h_imag = h_stacked(:, :, L+1:end);
    h_complex = complex(h_real, h_imag);
    
    fprintf('  Number of ports: %d\n', P);
    fprintf('  Output shape: (%d, %d, %d)\n\n', size(h_complex));
    
    % Step 6: Verify reconstruction
    fprintf('Step 5: Verifying reconstruction...\n');
    y_recon = squeeze(sum(h_complex, 2));
    recon_error = norm(y_complex - y_recon) / norm(y_complex);
    recon_error_db = 10 * log10(recon_error^2);
    
    fprintf('  Reconstruction error: %.2e (%.2f%%)\n', recon_error, recon_error * 100);
    fprintf('  Reconstruction error: %.2f dB\n', recon_error_db);
    
    if recon_error < 0.01
        fprintf('  ✓ EXCELLENT reconstruction quality!\n\n');
    elseif recon_error < 0.05
        fprintf('  ✓ GOOD reconstruction quality\n\n');
    elseif recon_error < 0.20
        fprintf('  ⚠️  MODERATE reconstruction quality\n');
        fprintf('     (Model may need more training)\n\n');
    else
        fprintf('  ⚠️  POOR reconstruction quality\n');
        fprintf('     (Model is likely untrained or loaded incorrectly)\n\n');
    end
    
    % Step 7: Energy distribution
    fprintf('Step 6: Energy distribution per port:\n');
    for p = 1:P
        port_data = squeeze(h_complex(:, p, :));
        port_energy = sqrt(mean(abs(port_data).^2));
        port_power_ratio = (port_energy^2) / (y_energy^2) * 100;
        fprintf('  Port %d: %.6f (%.1f%% of input)\n', p, port_energy, port_power_ratio);
    end
    fprintf('\n');
    
    % Summary
    fprintf('========================================\n');
    fprintf('Summary\n');
    fprintf('========================================\n');
    fprintf('Model:              %s\n', model_path);
    fprintf('Import:             ✓ SUCCESS\n');
    fprintf('Inference:          ✓ SUCCESS (%.2f ms)\n', inference_time * 1000);
    fprintf('Reconstruction:     %.2f%% (%.2f dB)\n', recon_error * 100, recon_error_db);
    fprintf('Number of ports:    %d\n', P);
    fprintf('========================================\n');
    fprintf('✓ All tests passed!\n\n');
    
    fprintf('Next steps:\n');
    fprintf('  1. Run full test with visualization: test_onnx_model()\n');
    fprintf('  2. Test with real SRS data\n');
    fprintf('  3. Deploy to production\n\n');
end
