% test_onnx_function.m
% 专门用于测试 importONNXFunction 导入的模型
%
% Usage:
%   test_onnx_function()
%   test_onnx_function('model_onnx_mode.onnx')

function test_onnx_function(model_path)
    if nargin < 1
        model_path = 'model_onnx_mode.onnx';
    end
    
    fprintf('========================================\n');
    fprintf('ONNX Function Test (importONNXFunction)\n');
    fprintf('========================================\n\n');
    
    % Step 1: Check file
    if ~exist(model_path, 'file')
        error('Model file not found: %s', model_path);
    end
    
    fprintf('Step 1: Importing ONNX as function...\n');
    fprintf('  Path: %s\n', model_path);
    
    % Import as function and extract parameters
    try
        % importONNXFunction 生成函数并返回参数对象
        params = importONNXFunction(model_path, 'model_func');
        fprintf('  ✓ Model imported as function!\n');
        fprintf('  Generated file: model_func.m\n');
        fprintf('  Parameters extracted: %s\n\n', class(params));
    catch ME
        fprintf('  ✗ Import failed: %s\n\n', ME.message);
        return;
    end
    
    % Step 2: Prepare test data
    fprintf('Step 2: Preparing test data...\n');
    
    L = 12;  % Sequence length
    
    % Generate complex signal
    y_complex = randn(1, L) + 1i*randn(1, L);
    y_stacked = [real(y_complex), imag(y_complex)];  % (1, 24)
    
    % Energy normalization (REQUIRED!)
    y_energy = sqrt(mean(abs(y_complex).^2));
    y_normalized = y_stacked / y_energy;
    
    fprintf('  Input shape: (%d, %d)\n', size(y_normalized));
    fprintf('  Input energy (before): %.6f\n', y_energy);
    fprintf('  Input energy (after): %.6f\n\n', sqrt(mean(y_normalized.^2)));
    
    % Step 3: Run inference
    fprintf('Step 3: Running inference...\n');
    
    try
        tic;
        % Call the imported function
        % 必须传递 params (ONNXParameters 对象)
        % 重要：使用 'none' 避免自动维度变换
        [h_normalized, ~] = model_func(y_normalized, params, ...
                                       'InputDataPermutation', 'none', ...
                                       'OutputDataPermutation', 'none');
        inference_time = toc;
        
        fprintf('  ✓ Inference successful! (%.2f ms)\n', inference_time * 1000);
        fprintf('  Output shape: %s\n', mat2str(size(h_normalized)));
        fprintf('  Output class: %s\n\n', class(h_normalized));
        
    catch ME
        fprintf('  ✗ Inference failed: %s\n', ME.message);
        fprintf('\n  Stack trace:\n');
        for i = 1:min(5, length(ME.stack))
            fprintf('    [%d] %s (line %d)\n', i, ME.stack(i).name, ME.stack(i).line);
        end
        fprintf('\n  Troubleshooting:\n');
        fprintf('    1. Check input dimensions: should be (1, 24)\n');
        fprintf('    2. Try different InputDataPermutation settings\n');
        fprintf('    3. Inspect model_func.m for expected input format\n\n');
        return;
    end
    
    % Step 4: Check and reshape output
    fprintf('Step 4: Processing output...\n');
    
    % 输出可能的维度格式：
    % - (24, 4, 1) - ONNX 格式 (C, P, B)
    % - (1, 4, 24) - 期望格式 (B, P, C)
    % - (4, 24, 1) - 其他可能格式
    
    fprintf('  Raw output size: %s\n', mat2str(size(h_normalized)));
    
    % 尝试重塑为 (B, P, L*2) 格式
    output_size = size(h_normalized);
    
    if length(output_size) == 3
        % 查找哪个维度是 24 (L*2)
        [~, L2_dim] = max(output_size == 24);
        % 查找哪个维度是 4 (P)
        [~, P_dim] = max(output_size == 4);
        % 剩下的是 batch 维度
        B_dim = setdiff([1 2 3], [L2_dim, P_dim]);
        
        fprintf('  Detected: Batch=%d, Ports=%d, Features=%d\n', ...
                B_dim, P_dim, L2_dim);
        
        % 重排为 (B, P, L*2)
        perm_order = [B_dim, P_dim, L2_dim];
        h_normalized = permute(h_normalized, perm_order);
        
        fprintf('  After permute: %s\n\n', mat2str(size(h_normalized)));
    elseif length(output_size) == 2
        % 2D 输出，需要 reshape
        if output_size(1) == 1 && output_size(2) == 96
            % (1, 96) -> (1, 4, 24)
            h_normalized = reshape(h_normalized, [1, 4, 24]);
            fprintf('  Reshaped from (1, 96) to (1, 4, 24)\n\n');
        elseif output_size(1) == 96 && output_size(2) == 1
            % (96, 1) -> (1, 4, 24)
            h_normalized = reshape(h_normalized, [1, 4, 24]);
            fprintf('  Reshaped from (96, 1) to (1, 4, 24)\n\n');
        else
            fprintf('  ⚠️  Unexpected 2D shape: %s\n\n', mat2str(output_size));
        end
    end
    
    % Step 5: Restore energy
    fprintf('Step 5: Restoring energy...\n');
    h_stacked = h_normalized * y_energy;
    
    % Step 6: Convert to complex
    fprintf('Step 6: Converting to complex...\n');
    
    P = size(h_stacked, 2);
    h_real = h_stacked(:, :, 1:L);
    h_imag = h_stacked(:, :, L+1:end);
    h_complex = complex(h_real, h_imag);
    
    fprintf('  Number of ports: %d\n', P);
    fprintf('  Complex shape: %s\n\n', mat2str(size(h_complex)));
    
    % Step 7: Verify reconstruction
    fprintf('Step 7: Verifying reconstruction...\n');
    
    y_recon = squeeze(sum(h_complex, 2));
    recon_error = norm(y_complex - y_recon) / norm(y_complex);
    recon_error_db = 10 * log10(recon_error^2);
    
    fprintf('  Reconstruction error: %.2e (%.2f%%)\n', recon_error, recon_error * 100);
    fprintf('  Reconstruction error: %.2f dB\n', recon_error_db);
    
    if recon_error < 0.01
        fprintf('  ✓ EXCELLENT reconstruction!\n\n');
        status = '✓ EXCELLENT';
    elseif recon_error < 0.05
        fprintf('  ✓ GOOD reconstruction\n\n');
        status = '✓ GOOD';
    elseif recon_error < 0.20
        fprintf('  ⚠️  MODERATE reconstruction\n\n');
        status = '⚠️  MODERATE';
    else
        fprintf('  ✗ POOR reconstruction\n\n');
        status = '✗ POOR';
    end
    
    % Step 8: Energy per port
    fprintf('Step 8: Energy distribution:\n');
    for p = 1:P
        port_data = squeeze(h_complex(:, p, :));
        port_energy = sqrt(mean(abs(port_data).^2));
        port_ratio = (port_energy^2) / (y_energy^2) * 100;
        fprintf('  Port %d: %.6f (%.1f%% of input)\n', p, port_energy, port_ratio);
    end
    fprintf('\n');
    
    % Summary
    fprintf('========================================\n');
    fprintf('Summary\n');
    fprintf('========================================\n');
    fprintf('Model:            %s\n', model_path);
    fprintf('Import method:    importONNXFunction\n');
    fprintf('Inference time:   %.2f ms\n', inference_time * 1000);
    fprintf('Reconstruction:   %s (%.2f%%, %.2f dB)\n', status, recon_error * 100, recon_error_db);
    fprintf('Number of ports:  %d\n', P);
    fprintf('========================================\n\n');
    
    if recon_error < 0.20
        fprintf('✓ Test PASSED!\n\n');
        fprintf('The model works in MATLAB using importONNXFunction.\n');
        fprintf('You can now integrate it into your application.\n\n');
    else
        fprintf('⚠️  Test completed with warnings.\n\n');
        fprintf('The model runs but reconstruction quality is poor.\n');
        fprintf('This might indicate:\n');
        fprintf('  1. Model needs more training\n');
        fprintf('  2. Incorrect energy normalization\n');
        fprintf('  3. Dimension mismatch in processing\n\n');
    end
    
    fprintf('Generated files:\n');
    fprintf('  - model_func.m (can be used directly)\n\n');
end
