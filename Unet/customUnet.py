import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    """一个轻量的通道注意力模块（ECA-Net变体），计算开销极小[12](@ref)"""
    def __init__(self, channel, reduction=16):
        super(SimpleAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        # 使用1D卷积代替全连接层以降低参数量和计算量
        self.conv = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  # 形状: (Batch, Channel, 1)
        y = self.conv(y)      # 形状: (Batch, Channel, 1)
        return x * y.expand_as(x)  # 将注意力权重广播回原尺寸并相乘

class ResidualBlock(nn.Module):
    """一个简单的残差块，用于构建U-Net的编码器和解码器基本单元[2,14](@ref)"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(ResidualBlock, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.attention = SimpleAttention(out_channels) if use_attention else nn.Identity()
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_layers(x)
        out = self.attention(out)  # 可选应用注意力
        out += residual  # 残差连接
        out = self.final_activation(out)
        return out

class SimpleResidualUNet(nn.Module):
    """轻量级残差U-Net，用于信道估计的残差学习[2,16](@ref)"""
    def __init__(self, input_channels=3, output_channels=1, base_channels=32, depth=3, attention_flag=False):
        """
        Args:
            input_channels: 输入特征数（例如：1个原始信号 + 2个位置编码 = 3）
            output_channels: 输出通道数（通常为1，即每个端口的残差）
            base_channels: 基础通道数，控制网络宽度，影响复杂度
            depth: 网络深度（下采样次数），控制网络规模
            attention_flag: 是否引入注意力机制的标志
        """
        super(SimpleResidualUNet, self).__init__()
        self.depth = depth
        self.attention_flag = attention_flag
        
        # 编码器路径 - 使用残差块
        self.enc_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        in_ch = input_channels
        out_ch = base_channels
        for i in range(depth):
            self.enc_blocks.append(ResidualBlock(in_ch, out_ch, use_attention=attention_flag))
            self.down_samples.append(nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2))  # 下采样
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)  # 通道数递增，但设置上限防止过宽
        
        # 瓶颈层
        self.bottleneck = ResidualBlock(in_ch, out_ch, use_attention=attention_flag)
        
        # 解码器路径 - 使用残差块
        self.up_samples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_samples.append(nn.ConvTranspose1d(out_ch, out_ch // 2, kernel_size=2, stride=2))  # 上采样
            in_ch_dec = out_ch + (base_channels * (2 ** (depth - i - 2)) if i < depth - 1 else base_channels)
            self.dec_blocks.append(ResidualBlock(in_ch_dec, out_ch // 2, use_attention=attention_flag))
            out_ch = out_ch // 2
        
        # 最终输出层
        self.final_conv = nn.Conv1d(out_ch, output_channels, kernel_size=1)

    def forward(self, x):
        # x 形状: (Batch, Features, Sequence_Length)
        skips = []
        
        # 编码器前向传播
        for i in range(self.depth):
            x = self.enc_blocks[i](x)
            skips.append(x)  # 保存跳跃连接特征
            x = self.down_samples[i](x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器前向传播
        for i in range(self.depth):
            x = self.up_samples[i](x)
            # 跳跃连接：拼接编码器对应层的特征
            skip = skips[-(i+1)]
            # 如果尺寸不匹配，进行裁剪或插值（确保全卷积兼容性）
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)  # 沿通道维度拼接
            x = self.dec_blocks[i](x)
        
        # 最终输出残差
        residual = self.final_conv(x)
        return residual

# 使用示例
if __name__ == "__main__":
    # 网络参数配置
    batch_size = 4
    seq_len = 12  # 或816，网络支持任意长度
    input_channels = 3  # 假设输入特征：原始信号 + sin编码 + cos编码
    output_channels = 1 # 输出每个端口的残差
    
    # 实例化网络（控制复杂度关键：调整base_channels和depth）
    model = SimpleResidualUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        base_channels=32,  # 减小此值可显著降低计算量（如16, 32, 64）
        depth=3,           # 减小深度可降低复杂度（如2, 3, 4）
        attention_flag=True # 根据需要开启/关闭注意力
    )
    
    # 模拟输入
    dummy_input = torch.randn(batch_size, input_channels, seq_len)
    
    # 前向传播：网络预测残差
    predicted_residual = model(dummy_input)  # 形状: (4, 1, 12)
    
    # 最终输出 = 原始输入（信号部分） + 预测残差
    # 假设原始信号在输入的第一个通道
    original_signal = dummy_input[:, 0:1, :]  # 形状: (4, 1, 12)
    enhanced_output = original_signal + predicted_residual
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"网络预测残差形状: {predicted_residual.shape}")
    print(f"增强输出形状: {enhanced_output.shape}")
    
    # 计算参数数量评估复杂度
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,} (可通过调整base_channels和depth控制)")