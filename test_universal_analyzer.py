"""
测试通用模型分析工具在不同模型上的效果
"""
import torch
import torch.nn as nn
from AnalyzeModelStructure import analyze_model_structure

# 测试1: 简单的ResNet块
class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

# 测试2: 简单的Transformer编码器层
class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
    def forward(self, x):
        x2, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        x2 = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x


if __name__ == "__main__":
    print("\n" + "="*120)
    print(" " * 40 + "测试1: 简单ResNet块")
    print("="*120)
    
    resnet_block = SimpleResBlock(64, 128)
    analyze_model_structure(resnet_block, "SimpleResBlock (64→128)", max_depth=None)
    
    print("\n\n" + "="*120)
    print(" " * 40 + "测试2: Transformer编码器层")
    print("="*120)
    
    transformer_block = SimpleTransformerBlock(d_model=256, nhead=8)
    analyze_model_structure(transformer_block, "SimpleTransformerBlock", max_depth=None)
    
    print("\n\n" + "="*120)
    print(" " * 35 + "测试3: 标准PyTorch模型 - ResNet18")
    print("="*120)
    
    from torchvision.models import resnet18
    resnet = resnet18(pretrained=False)
    analyze_model_structure(resnet, "ResNet18 (概览)", max_depth=2)
