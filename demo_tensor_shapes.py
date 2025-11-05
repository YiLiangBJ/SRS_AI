"""
演示：张量形状标注功能
对比改进前后的输出
"""

print("\n" + "="*100)
print(" " * 30 + "📐 张量形状标注演示")
print("="*100)

print("""
## ❌ 改进前的输出

├─ conv_real: Conv1d
│   【参数统计】
│     • weight: (8, 2, 3) = 48 (✓可训练)
│                ↑  ↑  ↑
│                什么意思？？


## ✅ 改进后的输出

├─ conv_real: Conv1d

    【张量形状 Tensor Shapes】           ← 🆕 新增！显示输入输出
      Input:  (B, 2, L)
      Output: (B, 8, L')
      说明: B=batch, L=length

    【参数统计】
      • weight: (8, 2, 3)  # (out_channels, in_channels, kernel_size) = 48
                ↑  ↑  ↑      ↑              ↑               ↑
                │  │  │      └──────────────┴───────────────┘
                │  │  │          🆕 每个维度的含义！
                │  │  └─ 卷积核大小 = 3
                │  └──── 输入通道数 = 2
                └─────── 输出通道数 = 8
                
      清晰明了！✨
""")

print("\n" + "="*100)
print(" " * 35 + "🎯 关键改进点")
print("="*100)

improvements = [
    ("1. 输入输出形状", "显示每个模块的Input和Output张量形状"),
    ("2. 维度含义说明", "每个维度代表什么(batch/channels/length等)"),
    ("3. 参数维度标注", "weight/bias每个维度的具体含义"),
    ("4. 符号解释", "B=batch, C=channels, L=length等"),
]

for title, desc in improvements:
    print(f"\n✅ {title}")
    print(f"   {desc}")

print("\n" + "="*100)
print(" " * 30 + "📊 实际使用示例")
print("="*100)

print("""
场景1：调试维度不匹配错误
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

错误信息:
  RuntimeError: size mismatch, expected (B, 16, L), got (B, 8, L)

查看报告:
  ├─ layer1: Conv1d
  │   Output: (B, 8, L)    ← 输出8通道
  │
  ├─ layer2: Conv1d
  │   Input:  (B, 16, L)   ← 期望16通道

→ 一眼看出问题：通道数不匹配！


场景2：理解参数量
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

看到参数:
  weight: (64, 32, 3)  # (out_channels, in_channels, kernel_size)

立即理解:
  - 64个输出通道（输出特征）
  - 32个输入通道（输入特征）
  - 3大小的卷积核
  - 参数量 = 64 × 32 × 3 = 6,144


场景3：追踪数据流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: (32, 2, 100)  ← 32个样本，2通道，100长度
  ↓
Conv1d: Input (B, 2, L) → Output (B, 8, L')
  ↓
Output: (32, 8, 100) ← 32个样本，8通道，100长度

清晰追踪每一步的形状变化！
""")

print("\n" + "="*100)
print(" " * 25 + "💡 现在运行 python AnalyzeModelStructure.py 查看效果！")
print("="*100)
