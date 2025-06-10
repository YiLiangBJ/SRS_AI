import torch
import torch.nn as nn
import torch.optim as optim
from model_cholesky import TrainableMMSEModule
import matplotlib.pyplot as plt
import numpy as np

def simulate_training_and_test_properties():
    """模拟训练过程并测试生成的矩阵是否保持厄米特和正定性"""
    
    print("模拟训练过程并测试生成的矩阵属性\n")
    
    # 创建模型
    model = TrainableMMSEModule(seq_length=144, mmse_block_size=4, hidden_dim=64, use_complex_input=True)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟训练过程
    num_epochs = 100
    batch_size = 16
    
    # 用于存储指标的列表
    hermitian_violations = []
    min_eigenvalues = []
    
    for epoch in range(num_epochs):
        # 生成随机批次
        inputs = torch.randn(batch_size, 144, dtype=torch.complex64)
        
        # 前向传播
        batch_C_matrices = []
        batch_R_matrices = []
        
        for i in range(batch_size):
            C, R = model(inputs[i])
            batch_C_matrices.append(C)
            batch_R_matrices.append(R)
        
        # 计算虚构损失函数 (此处仅用于演示训练过程)
        # 在实际应用中应该有一个适当的任务相关损失函数
        loss = 0.0
        for i in range(batch_size):
            C = batch_C_matrices[i]
            R = batch_R_matrices[i]
            
            # 示例损失：希望C与R之间有一定差异
            loss += torch.norm(C - R)
            
            # 示例约束：希望C的迹接近某个目标值
            trace_target = 4.0  # 假设的目标值
            loss += torch.abs(torch.trace(C).real - trace_target)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每10个轮次检查一次矩阵属性
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            # 使用固定输入进行评估
            eval_input = torch.randn(144, dtype=torch.complex64)
            C, R = model(eval_input)
            
            # 检查厄米特性违规程度
            hermitian_diff = torch.norm(C - C.conj().transpose(0, 1)).item()
            hermitian_violations.append(hermitian_diff)
            
            # 检查最小特征值
            try:
                eigenvalues = torch.linalg.eigvalsh(C)
                min_eig = eigenvalues.min().item()
                min_eigenvalues.append(min_eig)
                
                print(f"Epoch {epoch}: 损失 = {loss.item():.6f}, 厄米特违规 = {hermitian_diff:.6e}, 最小特征值 = {min_eig:.6e}")
            except Exception as e:
                print(f"Epoch {epoch}: 计算特征值出错: {e}")
                min_eigenvalues.append(float('nan'))
    
    print("\n训练完成!")
    
    # 绘制训练过程中的指标
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(range(0, num_epochs, 10), hermitian_violations)
    plt.xlabel('Epoch')
    plt.ylabel('厄米特违规 (对数刻度)')
    plt.title('训练过程中的厄米特性违规')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(0, num_epochs, 10), min_eigenvalues)
    plt.xlabel('Epoch')
    plt.ylabel('最小特征值')
    plt.title('训练过程中的最小特征值')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_matrix_properties.png')
    print("训练过程中的矩阵属性图形已保存为 training_matrix_properties.png")
    
    # 最终检查：输入大量随机样本，验证属性
    print("\n使用多个随机输入进行最终验证...")
    num_samples = 1000
    hermitian_ok = 0
    positive_definite_ok = 0
    
    for i in range(num_samples):
        random_input = torch.randn(144, dtype=torch.complex64)
        C, R = model(random_input)
        
        # 检查厄米特性
        if torch.allclose(C, C.conj().transpose(0, 1), atol=1e-5):
            hermitian_ok += 1
        
        # 检查正定性
        try:
            eigenvalues = torch.linalg.eigvalsh(C)
            if torch.all(eigenvalues > 0):
                positive_definite_ok += 1
        except:
            pass
    
    print(f"在 {num_samples} 个随机输入中:")
    print(f"厄米特性满足: {hermitian_ok}/{num_samples} ({100*hermitian_ok/num_samples:.2f}%)")
    print(f"正定性满足: {positive_definite_ok}/{num_samples} ({100*positive_definite_ok/num_samples:.2f}%)")

if __name__ == "__main__":
    simulate_training_and_test_properties()
