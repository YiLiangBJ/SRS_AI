import torch
import numpy as np
from model_cholesky import TrainableMMSEModule

def verify_hermitian():
    """验证使用Cholesky因子构造法生成的矩阵是否满足Hermitian性质"""
    
    # 创建模型实例
    module = TrainableMMSEModule(seq_length=144, mmse_block_size=4, hidden_dim=64, use_complex_input=True)
    
    # 生成随机输入
    input_tensor = torch.randn(144, dtype=torch.complex64)
    
    # 获取C矩阵
    C, _ = module(input_tensor)
    
    # 打印C矩阵的形状和类型信息
    print(f"C矩阵形状: {C.shape}, 类型: {C.dtype}")
    
    # 打印C矩阵的第一行和第一列
    print("\nC矩阵第0行:")
    print(C[0,:])
    
    print("\nC矩阵第0列:")
    print(C[:,0])
    
    # 验证C[0,1]和C[1,0]是否为共轭关系
    print("\n检查C[0,1]和C[1,0]是否为共轭关系:")
    print(f"C[0,1] = {C[0,1]}")
    print(f"C[1,0] = {C[1,0]}")
    print(f"C[1,0]是C[0,1]的共轭? {torch.allclose(C[1,0], C[0,1].conj())}")
    
    # 验证第0行第1列和第1行第0列的共轭关系
    print("\n检查第0行第1列和第1行第0列的元素:")
    print(f"实部: C[0,1].real = {C[0,1].real}, C[1,0].real = {C[1,0].real}")
    print(f"虚部: C[0,1].imag = {C[0,1].imag}, C[1,0].imag = {C[1,0].imag}")
    print(f"实部相等? {torch.isclose(C[0,1].real, C[1,0].real)}")
    print(f"虚部相反? {torch.isclose(C[0,1].imag, -C[1,0].imag)}")
    
    # 检查整个矩阵是否是Hermitian的
    is_hermitian = torch.allclose(C, C.conj().transpose(0, 1))
    print(f"\n整个矩阵是Hermitian的? {is_hermitian}")

if __name__ == "__main__":
    verify_hermitian()
