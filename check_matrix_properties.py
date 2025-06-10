import torch
import numpy as np
from model_cholesky import TrainableMMSEModule

def display_matrix_hermitian_property():
    """显示矩阵的厄米特性的详细信息"""
    
    print("详细检查Cholesky因子构造法生成的矩阵的厄米特性\n")
    
    # 创建一个小矩阵以便于显示
    module = TrainableMMSEModule(seq_length=144, mmse_block_size=3, hidden_dim=64, use_complex_input=True)
    
    # 固定随机种子以重现结果
    torch.manual_seed(42)
    input_tensor = torch.randn(144, dtype=torch.complex64)
    
    # 获取C矩阵
    C, R = module(input_tensor)
    
    # 打印整个C矩阵
    print("=== C矩阵 ===")
    print(C)
    print("\n=== C矩阵的共轭转置 ===")
    print(C.conj().transpose(0, 1))
    
    # 验证两者是否相等
    is_hermitian = torch.allclose(C, C.conj().transpose(0, 1), atol=1e-5)
    print(f"\nC矩阵是厄米特的? {is_hermitian}")
    
    # 详细检查每个元素对
    n = C.shape[0]
    print("\n=== 检查C矩阵每对元素的共轭关系 ===")
    for i in range(n):
        for j in range(i+1, n):  # 只检查对角线上方的元素
            is_conj = torch.isclose(C[i, j], C[j, i].conj(), atol=1e-5)
            print(f"C[{i},{j}] = {C[i,j]:.5f}, C[{j},{i}] = {C[j,i]:.5f}, 是共轭对? {is_conj}")
    
    # 检查对角元素是否为实数
    print("\n=== 检查C矩阵对角元素是否为实数 ===")
    for i in range(n):
        is_real = torch.isclose(C[i, i].imag, torch.tensor(0.0), atol=1e-5)
        print(f"C[{i},{i}] = {C[i,i]:.5f}, 是实数? {is_real}")
    
    # 检查矩阵的正定性
    eigenvalues = torch.linalg.eigvalsh(C).tolist()
    print("\n=== C矩阵的特征值 ===")
    for i, eig in enumerate(eigenvalues):
        print(f"λ_{i} = {eig:.10f}")
    
    # 计算和打印Cholesky因子
    print("\n=== 验证通过Cholesky分解还原的L矩阵 ===")
    try:
        # 注意：对于厄米特正定矩阵，torch.linalg.cholesky将计算下三角矩阵L使得C = L @ L^H
        L = torch.linalg.cholesky(C)
        print("Cholesky因子L:")
        print(L)
        
        # 验证C = L @ L^H
        C_reconstructed = L @ L.conj().transpose(0, 1)
        reconstruction_accurate = torch.allclose(C, C_reconstructed, atol=1e-5)
        print(f"\n重建的C矩阵与原始C矩阵匹配? {reconstruction_accurate}")
        
        if not reconstruction_accurate:
            print("原始C矩阵:")
            print(C)
            print("\n重建的C矩阵:")
            print(C_reconstructed)
            print("\n差异:")
            print(C - C_reconstructed)
    except Exception as e:
        print(f"计算Cholesky分解时出错: {e}")

if __name__ == "__main__":
    display_matrix_hermitian_property()
    print("\n验证完成!")
