import torch
import numpy as np
from model_cholesky import TrainableMMSEModule
import matplotlib.pyplot as plt

def test_hermitian_property():
    """测试使用Cholesky因子构造法生成的矩阵是否满足Hermitian性质和正定性"""
    
    print("测试Cholesky因子构造法生成矩阵的厄米特性和正定性\n")
    
    # 创建模型实例，测试不同的矩阵大小
    block_sizes = [2, 4, 8, 12]
    
    for size in block_sizes:
        print(f"\n=== 测试矩阵大小: {size}x{size} ===")
        module = TrainableMMSEModule(seq_length=144, mmse_block_size=size, hidden_dim=64, use_complex_input=True)
        
        # 生成随机输入
        input_tensor = torch.randn(144, dtype=torch.complex64)
        
        # 获取C矩阵和R矩阵
        C, R = module(input_tensor)
        
        # 验证C矩阵的厄米特性
        c_is_hermitian = torch.allclose(C, C.conj().transpose(0, 1), atol=1e-5)
        print(f"C矩阵是厄米特的? {c_is_hermitian}")
        
        # 验证R矩阵的厄米特性
        r_is_hermitian = torch.allclose(R, R.conj().transpose(0, 1), atol=1e-5)
        print(f"R矩阵是厄米特的? {r_is_hermitian}")
        
        # 检查C矩阵的对角元素是否为实数
        c_diag = torch.diag(C)
        c_diag_is_real = torch.allclose(c_diag.imag, torch.zeros_like(c_diag.imag), atol=1e-5)
        print(f"C矩阵对角元素是实数? {c_diag_is_real}")
        
        # 检查R矩阵的对角元素是否为实数
        r_diag = torch.diag(R)
        r_diag_is_real = torch.allclose(r_diag.imag, torch.zeros_like(r_diag.imag), atol=1e-5)
        print(f"R矩阵对角元素是实数? {r_diag_is_real}")
        
        # 检查C矩阵的正定性（所有特征值应为正）
        try:
            # 将复数矩阵转换为实数矩阵以计算特征值
            # 对于厄米特矩阵，特征值总是实数
            c_eigenvalues = torch.linalg.eigvalsh(C)
            c_is_positive_definite = torch.all(c_eigenvalues > 0)
            print(f"C矩阵是正定的? {c_is_positive_definite}")
            print(f"C矩阵特征值范围: [{c_eigenvalues.min().item():.6e}, {c_eigenvalues.max().item():.6e}]")
        except Exception as e:
            print(f"计算C矩阵特征值时出错: {e}")
        
        # 检查R矩阵的正定性
        try:
            r_eigenvalues = torch.linalg.eigvalsh(R)
            r_is_positive_definite = torch.all(r_eigenvalues > 0)
            print(f"R矩阵是正定的? {r_is_positive_definite}")
            print(f"R矩阵特征值范围: [{r_eigenvalues.min().item():.6e}, {r_eigenvalues.max().item():.6e}]")
        except Exception as e:
            print(f"计算R矩阵特征值时出错: {e}")

        # 如果矩阵大小合适，打印一个具体示例
        if size <= 4:
            print("\nC矩阵的一个具体示例:")
            print("第0行第1列:", C[0, 1])
            print("第1行第0列:", C[1, 0])
            print(f"C[1,0]是C[0,1]的共轭? {torch.isclose(C[1,0], C[0,1].conj(), atol=1e-5)}")

def visualize_matrix(matrix, title):
    """可视化复数矩阵的幅度和相位"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制幅度
    im1 = ax1.imshow(torch.abs(matrix).detach().cpu().numpy(), cmap='viridis')
    ax1.set_title(f"{title} - 幅度")
    fig.colorbar(im1, ax=ax1)
    
    # 绘制相位
    im2 = ax2.imshow(torch.angle(matrix).detach().cpu().numpy(), cmap='hsv')
    ax2.set_title(f"{title} - 相位")
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    return fig

def run_numerical_stability_test():
    """测试数值稳定性：对输入做小扰动，观察矩阵输出的变化"""
    print("\n=== 数值稳定性测试 ===")
    
    module = TrainableMMSEModule(seq_length=144, mmse_block_size=8, hidden_dim=64, use_complex_input=True)
    
    # 创建基础输入
    base_input = torch.randn(144, dtype=torch.complex64)
    
    # 创建扰动输入
    perturbed_input = base_input + 0.01 * torch.randn(144, dtype=torch.complex64)
    
    # 获取矩阵
    C1, R1 = module(base_input)
    C2, R2 = module(perturbed_input)
    
    # 计算相对变化
    c_rel_diff = torch.norm(C2 - C1) / torch.norm(C1)
    r_rel_diff = torch.norm(R2 - R1) / torch.norm(R1)
    
    print(f"输入相对扰动: {torch.norm(perturbed_input - base_input) / torch.norm(base_input):.6f}")
    print(f"C矩阵相对变化: {c_rel_diff.item():.6f}")
    print(f"R矩阵相对变化: {r_rel_diff.item():.6f}")
    
    # 创建可视化
    fig_c = visualize_matrix(C1, "C矩阵")
    fig_r = visualize_matrix(R1, "R矩阵")
    
    # 保存图像
    fig_c.savefig("c_matrix_visualization.png")
    fig_r.savefig("r_matrix_visualization.png")
    
    print("矩阵可视化保存为PNG文件")

if __name__ == "__main__":
    test_hermitian_property()
    run_numerical_stability_test()
    print("\n测试完成!")