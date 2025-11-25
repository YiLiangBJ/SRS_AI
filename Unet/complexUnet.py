"""
Complex-valued U-Net for SRS Channel Estimation and Denoising
支持复数输入的 U-Net，用于信道估计去噪和解复用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ComplexReLU(nn.Module):
    """
    复数域的 ReLU 激活函数
    分别对实部和虚部应用 ReLU
    """
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class ComplexModReLU(nn.Module):
    """
    基于模值的复数 ReLU (modReLU)
    公式: modReLU(z) = ReLU(|z| + b) * (z / |z|)
    """
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        # x: (Batch, Channels, Length)
        magnitude = torch.abs(x) + 1e-8  # 避免除零
        bias = self.bias.view(1, -1, 1)  # 广播到 (1, Channels, 1)
        
        # modReLU: ReLU(|z| + b) * (z / |z|)
        activated_mag = F.relu(magnitude + bias)
        phase = x / magnitude
        return activated_mag * phase


class ComplexBatchNorm1d(nn.Module):
    """
    复数批归一化
    分别对实部和虚部进行归一化
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn_real = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.bn_imag = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    
    def forward(self, x):
        real_normalized = self.bn_real(x.real)
        imag_normalized = self.bn_imag(x.imag)
        return torch.complex(real_normalized, imag_normalized)


class ComplexConv1d(nn.Module):
    """
    复数卷积层
    将复数卷积分解为实部和虚部的运算
    支持循环卷积（circular convolution）
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, circular=False):
        super().__init__()
        self.circular = circular
        self.kernel_size = kernel_size
        self.stride = stride
        
        # 如果使用循环卷积，内部padding设为0，手动处理
        internal_padding = 0 if circular else padding
        
        # 实部和虚部各自的卷积
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, internal_padding, bias=bias)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, internal_padding, bias=bias)
        
        # 如果不是循环卷积，保存padding用于后续
        self.padding = padding if not circular else 0
    
    def forward(self, x):
        # x 是复数张量: (Batch, Channels, Length)
        
        # 如果使用循环卷积，先进行循环填充
        if self.circular:
            # 计算需要的填充量（kernel_size - 1）
            # 对于 kernel_size=3: 左边填充1个，右边填充1个
            pad_total = self.kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            
            # 循环填充：使用序列的首尾元素
            # 例如 [0,1,2,...,11] -> [11,0,1,2,...,11,0]
            x_real_padded = F.pad(x.real, (pad_left, pad_right), mode='circular')
            x_imag_padded = F.pad(x.imag, (pad_left, pad_right), mode='circular')
            
            x_padded = torch.complex(x_real_padded, x_imag_padded)
        else:
            x_padded = x
        
        # 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        real_real = self.conv_real(x_padded.real)  # ac
        real_imag = self.conv_real(x_padded.imag)  # ad
        imag_real = self.conv_imag(x_padded.real)  # bc
        imag_imag = self.conv_imag(x_padded.imag)  # bd
        
        real_part = real_real - imag_imag  # ac - bd
        imag_part = real_imag + imag_real  # ad + bc
        
        return torch.complex(real_part, imag_part)


class ComplexConvTranspose1d(nn.Module):
    """
    复数转置卷积（上采样）
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, output_padding=0):
        super().__init__()
        self.conv_real = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_imag = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
    
    def forward(self, x):
        real_real = self.conv_real(x.real)
        real_imag = self.conv_real(x.imag)
        imag_real = self.conv_imag(x.real)
        imag_imag = self.conv_imag(x.imag)
        
        real_part = real_real - imag_imag
        imag_part = real_imag + imag_real
        
        return torch.complex(real_part, imag_part)


class ComplexAttention(nn.Module):
    """
    复数域的通道注意力模块
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 确保 reduction 后的通道数至少为 1
        reduced_channels = max(1, channels // reduction)
        
        self.fc = nn.Sequential(
            ComplexConv1d(channels, reduced_channels, 1, bias=False),
            ComplexReLU(),
            ComplexConv1d(reduced_channels, channels, 1, bias=False)
        )
    
    def forward(self, x):
        # 对幅值进行池化
        magnitude = torch.abs(x)
        y = self.avg_pool(magnitude)
        
        # 使用原始复数输入的相位信息
        phase = x / (torch.abs(x) + 1e-8)
        avg_complex = y * phase.mean(dim=-1, keepdim=True)
        
        # 通过全连接层
        attention = self.fc(avg_complex)
        
        # Sigmoid 应用于幅值
        attention_mag = torch.sigmoid(torch.abs(attention))
        attention_phase = attention / (torch.abs(attention) + 1e-8)
        
        return x * (attention_mag * attention_phase).expand_as(x)


class ComplexResidualBlock(nn.Module):
    """
    复数残差块
    """
    def __init__(self, in_channels, out_channels, use_attention=False, activation='modrelu', circular=False):
        super().__init__()
        
        # 选择激活函数
        if activation == 'modrelu':
            self.activation1 = ComplexModReLU(out_channels)
            self.activation2 = ComplexModReLU(out_channels)
        else:  # 'relu'
            self.activation1 = ComplexReLU()
            self.activation2 = ComplexReLU()
        
        # 使用普通卷积，padding填充0
        self.conv1 = ComplexConv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, circular=circular)
        self.bn1 = ComplexBatchNorm1d(out_channels)
        
        self.conv2 = ComplexConv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, circular=circular)
        self.bn2 = ComplexBatchNorm1d(out_channels)
        
        # 残差连接
        if in_channels != out_channels:
            self.shortcut = ComplexConv1d(in_channels, out_channels, kernel_size=1, bias=False, circular=False)
        else:
            self.shortcut = nn.Identity()
        
        # 注意力
        self.attention = ComplexAttention(out_channels) if use_attention else nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.attention(out)
        out = out + residual
        out = self.activation2(out)
        
        return out


class ComplexPositionalEncoding(nn.Module):
    """
    生成复数位置编码（改进版）
    
    新版本：为序列的每个位置生成不同的相位
    pos_encoding[k] = exp(j * 2π * pos / seq_len * k), k ∈ [0, seq_len-1]
    
    注意：N 自动等于 seq_len，这样归一化更合理
    
    这样同时编码了：
    1. port的位置信息（pos）
    2. 序列内的位置信息（k）
    
    pos_values: 每个 port 的位置值列表 (num_ports,)
    """
    def __init__(self):
        super().__init__()
        # 不再需要预定义N，每次forward时使用seq_len作为N
    
    def forward(self, batch_size, seq_len, pos_values, device='cpu'):
        """
        Args:
            batch_size: batch 大小
            seq_len: 序列长度
            pos_values: 位置值 tensor, shape (num_ports,)
            device: 设备
        
        Returns:
            位置编码: (batch_size, num_ports, seq_len) 复数张量
        """
        num_ports = len(pos_values)
        
        # N自动等于seq_len
        N = seq_len
        
        # pos_values: (num_ports,) -> (1, num_ports, 1)
        pos = pos_values.view(1, num_ports, 1).float().to(device)
        
        # 序列索引: [0, 1, 2, ..., seq_len-1]
        # shape: (seq_len,) -> (1, 1, seq_len)
        seq_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len).float()
        
        # 计算相位: 2π * pos / seq_len * seq_idx
        # pos: (1, num_ports, 1)
        # seq_idx: (1, 1, seq_len)
        # 结果: (1, num_ports, seq_len)
        phase = 2 * np.pi * pos / N * seq_idx
        
        # 生成复数编码: exp(j * phase)
        # 每个位置k的编码为: exp(j * 2π * pos / seq_len * k)
        encoding = torch.exp(1j * phase).expand(batch_size, num_ports, seq_len)
        
        return encoding


class ComplexResidualUNet(nn.Module):
    """
    复数 U-Net 用于 SRS 信道估计
    
    输入格式: (batch_size, num_ports, channels, seq_len) 复数张量
        - channels=2: [原始信道估计, 位置编码]
        - 每个 port 有独立的位置编码
    
    输出格式: (batch_size, num_ports, 1, seq_len) 复数张量
        - 去噪后的信道估计残差
    """
    def __init__(self, input_channels=2, output_channels=1, base_channels=32, 
                 depth=3, attention_flag=False, activation='modrelu', circular=False):
        """
        Args:
            input_channels: 输入通道数 (默认2: 原始信号 + 位置编码)
            output_channels: 输出通道数 (默认1: 残差)
            base_channels: 基础通道数
            depth: 网络深度
            attention_flag: 是否使用注意力
            activation: 激活函数类型 ('modrelu' 或 'relu')
            circular: 是否使用循环卷积 (默认False，使用普通卷积padding填充0)
        """
        super().__init__()
        self.depth = depth
        self.attention_flag = attention_flag
        self.circular = circular
        
        # 编码器
        self.enc_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        in_ch = input_channels
        out_ch = base_channels
        
        for i in range(depth):
            self.enc_blocks.append(
                ComplexResidualBlock(in_ch, out_ch, use_attention=attention_flag, activation=activation, circular=circular)
            )
            self.down_samples.append(
                ComplexConv1d(out_ch, out_ch, kernel_size=2, stride=2, circular=False)  # 下采样不使用循环
            )
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)
        
        # 瓶颈层
        self.bottleneck = ComplexResidualBlock(in_ch, out_ch, use_attention=attention_flag, activation=activation, circular=circular)
        
        # 解码器
        self.up_samples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        
        for i in range(depth):
            self.up_samples.append(
                ComplexConvTranspose1d(out_ch, out_ch // 2, kernel_size=2, stride=2)
            )
            # 计算解码器输入通道数（跳跃连接）
            # 编码器通道数：base_ch, base_ch*2, base_ch*4, ...
            # 从深到浅：depth-1, depth-2, ..., 0
            encoder_level = depth - i - 1
            skip_ch = min(base_channels * (2 ** encoder_level), 256)
            in_ch_dec = out_ch // 2 + skip_ch
            
            self.dec_blocks.append(
                ComplexResidualBlock(in_ch_dec, out_ch // 2, use_attention=attention_flag, activation=activation, circular=circular)
            )
            out_ch = out_ch // 2
        
        # 最终输出层 - 使用循环卷积保持长度
        self.final_conv = ComplexConv1d(out_ch, output_channels, kernel_size=1, circular=False)
    
    def forward(self, x):
        """
        Args:
            x: 复数输入 (batch_size, num_ports, input_channels, seq_len)
        
        Returns:
            残差输出 (batch_size, num_ports, output_channels, seq_len)
        """
        batch_size, num_ports, in_ch, seq_len = x.shape
        
        # 将 ports 维度合并到 batch 维度
        # (B, P, C, L) -> (B*P, C, L)
        x = x.reshape(batch_size * num_ports, in_ch, seq_len)
        
        skips = []
        
        # 编码器
        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            skips.append(x)
            x = self.down_samples[i](x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器
        for i in range(self.depth):
            x = self.up_samples[i](x)
            skip = skips[-(i+1)]
            
            # 处理尺寸不匹配
            if x.shape[-1] != skip.shape[-1]:
                # 对实部和虚部分别插值
                x_real = F.interpolate(x.real, size=skip.shape[-1], mode='linear', align_corners=False)
                x_imag = F.interpolate(x.imag, size=skip.shape[-1], mode='linear', align_corners=False)
                x = torch.complex(x_real, x_imag)
            
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)
        
        # 最终输出
        residual = self.final_conv(x)
        
        # 恢复形状: (B*P, out_ch, L) -> (B, P, out_ch, L)
        residual = residual.reshape(batch_size, num_ports, -1, seq_len)
        
        return residual


def create_input_with_positional_encoding(channel_estimates, pos_values, device='cpu'):
    """
    创建带位置编码的输入张量
    
    Args:
        channel_estimates: 原始信道估计 (batch_size, num_ports, seq_len) 复数张量
        pos_values: 每个 port 的位置值 (num_ports,) 整数 tensor
        device: 设备
    
    Returns:
        (batch_size, num_ports, 2, seq_len) 复数张量
            - 第1通道: 原始信道估计
            - 第2通道: 位置编码
    
    注意: N 自动等于 seq_len
    """
    batch_size, num_ports, seq_len = channel_estimates.shape
    
    # 生成位置编码（N自动等于seq_len）
    pos_encoder = ComplexPositionalEncoding()
    pos_encoding = pos_encoder(batch_size, seq_len, pos_values, device)
    
    # Stack: (batch_size, num_ports, 2, seq_len)
    # 沿着通道维度拼接
    input_tensor = torch.stack([channel_estimates, pos_encoding], dim=2)
    
    return input_tensor


# 使用示例
if __name__ == "__main__":
    # 参数配置
    batch_size = 8
    num_ports = 4
    seq_len = 12  # rb_num=4, seq_len=12
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = ComplexResidualUNet(
        input_channels=2,
        output_channels=1,
        base_channels=8,
        depth=2,
        attention_flag=True,
        activation='modrelu'
    ).to(device)
    
    # 模拟输入数据
    # 原始信道估计: (batch_size, num_ports, seq_len)
    channel_estimates = torch.randn(batch_size, num_ports, seq_len, dtype=torch.complex64).to(device)
    
    # 每个 port 的位置值（手动指定）
    pos_values = torch.tensor([0, 2, 6, 8], dtype=torch.long)  # 4个port的位置
    
    # 创建输入（包含位置编码，N自动等于seq_len）
    input_tensor = create_input_with_positional_encoding(
        channel_estimates, pos_values, device
    )
    
    print("=" * 80)
    print("Complex U-Net Input/Output Verification")
    print("=" * 80)
    print(f"\nInput shape:  {input_tensor.shape}")
    print(f"  - Batch size:      {batch_size}")
    print(f"  - Num ports:       {num_ports}")
    print(f"  - Input channels:  2 (Channel estimate + Position encoding)")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - N (normalization): {seq_len} (自动等于seq_len)")
    print(f"Input dtype:  {input_tensor.dtype}")
    
    # 前向传播
    output_residual = model(input_tensor)
    
    print(f"\nOutput residual shape: {output_residual.shape}")
    print(f"  - Batch size:       {batch_size}")
    print(f"  - Num ports:        {num_ports}")
    print(f"  - Output channels:  1 (Residual)")
    print(f"  - Sequence length:  {seq_len}")
    print(f"Output dtype: {output_residual.dtype}")
    
    # 最终输出 = 原始估计 + 残差
    final_output = channel_estimates.unsqueeze(2) + output_residual
    
    print(f"\nFinal output shape: {final_output.shape}")
    
    # 验证每个port的输出
    print("\n" + "=" * 80)
    print("Verification: Each port has independent output")
    print("=" * 80)
    
    for port_idx in range(num_ports):
        port_output = final_output[0, port_idx, 0, :]  # Batch 0, Port i
        print(f"\nPort {port_idx} (position={pos_values[port_idx]}):")
        print(f"  Output shape:     {port_output.shape}")
        print(f"  Output magnitude: {torch.abs(port_output).mean():.4f}")
        print(f"  First 3 values:   {port_output[:3]}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 80)
    print("MODEL PARAMETERS")
    print("=" * 80)
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    
    # 按模块统计
    module_params = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if module_name not in module_params:
            module_params[module_name] = 0
        module_params[module_name] += param.numel()
    
    print("\n  Parameters by module:")
    for module_name in sorted(module_params.keys()):
        params = module_params[module_name]
        percentage = (params / total_params) * 100
        print(f"    {module_name:<15} {params:>12,}  ({percentage:>5.2f}%)")
    
    print("=" * 80)
    
    # 测试不同序列长度
    print("\n" + "=" * 80)
    print("Testing different sequence lengths:")
    print("=" * 80)
    for rb_num in [4, 68, 272]:
        test_seq_len = rb_num * 3
        test_input = torch.randn(2, num_ports, test_seq_len, dtype=torch.complex64).to(device)
        test_input_full = create_input_with_positional_encoding(test_input, pos_values, N, device)
        
        try:
            test_output = model(test_input_full)
            print(f"  rb_num={rb_num:3d}, seq_len={test_seq_len:4d} -> output: {test_output.shape} [OK]")
        except Exception as e:
            print(f"  rb_num={rb_num:3d}, seq_len={test_seq_len:4d} -> Error: {e}")
