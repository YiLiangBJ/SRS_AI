import torch

def test_residual_unet():
    """
    单元测试示例：验证SimpleResidualUNet的基本功能。
    重点检查输入输出维度、残差学习思想是否正确体现。
    """
    model = SimpleResidualUNet(
        input_channels=3,
        output_channels=1,
        base_channels=16,  # 使用小模型进行快速测试
        depth=2,
        attention_flag=True
    )
    
    # 模拟一个批量数据
    batch_size = 2
    seq_len = 12
    dummy_input = torch.randn(batch_size, 3, seq_len)
    
    # 测试前向传播
    with torch.no_grad():
        predicted_residual = model(dummy_input)
        # 检查残差输出维度
        assert predicted_residual.shape == (batch_size, 1, seq_len), f"残差输出维度错误: {predicted_residual.shape}"
        
        # 检查最终输出计算
        original_signal = dummy_input[:, 0:1, :]
        enhanced_output = original_signal + predicted_residual
        assert enhanced_output.shape == (batch_size, 1, seq_len), f"最终输出维度错误: {enhanced_output.shape}"
    
    print("✓ 前向传播维度测试通过！")
    print(f"  输入维度: {dummy_input.shape}")
    print(f"  网络预测残差形状: {predicted_residual.shape}")
    print(f"  增强输出形状: {enhanced_output.shape}")

# 运行测试
if __name__ == "__main__":
    test_residual_unet()