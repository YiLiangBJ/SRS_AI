"""
演示：模型结构分析 - 显示Forward执行顺序
"""

from complexUnet import ComplexResidualUNet
from AnalyzeModelStructure import analyze_model_structure

print("\n" + "="*120)
print(" " * 35 + "🎯 演示：Forward执行顺序可视化")
print("="*120)

# 创建一个小型模型用于演示
model = ComplexResidualUNet(
    input_channels=2,
    output_channels=1,
    base_channels=8,
    depth=2,
    attention_flag=True
)

# 分析时显示forward执行顺序（深度限制为3，避免输出过长）
print("\n📌 显示主要模块的forward执行顺序（depth=3）\n")
analyze_model_structure(
    model, 
    "ComplexResidualUNet - Forward Flow演示", 
    max_depth=3,
    show_forward_order=True  # 显示执行顺序
)

print("\n\n" + "="*120)
print(" " * 30 + "💡 执行顺序说明")
print("="*120)
print("""
【执行顺序 Forward Flow】显示每个模块在forward()方法中的调用顺序。

例如，ComplexResidualBlock的执行顺序：
  1. residual = self.shortcut(...)     ← 先保存shortcut
  2. out = self.conv1(...)              ← 第一个卷积
  3. out = self.bn1(...)                ← 批归一化
  4. out = self.activation1(...)        ← 激活函数
  5. out = self.conv2(...)              ← 第二个卷积
  6. out = self.bn2(...)                ← 批归一化
  7. out = self.attention(...)          ← 注意力机制
  8. out = self.activation2(...)        ← 最终激活

这清楚地展示了残差块的实际执行流程！

【对比】：
- 模块定义顺序（named_children）：按代码中__init__定义的顺序
- 执行顺序（Forward Flow）：按forward()中实际调用的顺序

这两者通常不同！例如，activation1和activation2在__init__中先定义，
但在forward中，它们是在conv和bn之后才被调用的。
""")

print("="*120)
