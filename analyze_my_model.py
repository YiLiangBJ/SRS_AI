"""
快速分析模型结构并输出到文件

使用方法：
    python analyze_my_model.py
"""

from complexUnet import ComplexResidualUNet
from AnalyzeModelStructure import analyze_model_structure

# ============================================
# 配置：修改这里来分析不同的模型
# ============================================

# 创建你的模型
model = ComplexResidualUNet(
    input_channels=2,
    output_channels=1,
    base_channels=16,
    depth=3,
    attention_flag=True,
    activation='modrelu'
)

# 配置输出
OUTPUT_DIR = "analysis_reports"  # 输出目录
MODEL_NAME = "ComplexResidualUNet_d3_b16"  # 模型名称

# ============================================
# 执行分析
# ============================================

import os

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*100)
print(f" 📊 模型结构分析工具")
print("="*100)

# 1. 生成完整详细报告
print(f"\n1️⃣  生成完整详细分析...")
full_report = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_full.txt")
analyze_model_structure(
    model,
    model_name=f"{MODEL_NAME} - 完整分析",
    max_depth=None,  # 所有层级
    show_forward_order=True,
    output_file=full_report
)

# 2. 生成概览报告（深度限制）
print(f"\n2️⃣  生成概览报告...")
summary_report = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_summary.txt")
analyze_model_structure(
    model,
    model_name=f"{MODEL_NAME} - 概览",
    max_depth=3,  # 只显示3层
    show_forward_order=True,
    output_file=summary_report
)

# 3. 生成纯结构报告（不显示执行顺序）
print(f"\n3️⃣  生成纯结构报告...")
structure_report = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_structure.txt")
analyze_model_structure(
    model,
    model_name=f"{MODEL_NAME} - 结构",
    max_depth=None,
    show_forward_order=False,  # 不显示forward顺序
    output_file=structure_report
)

# ============================================
# 总结
# ============================================

print("\n" + "="*100)
print("✅ 分析完成！生成了以下报告：")
print("="*100)

reports = [
    (full_report, "完整详细分析（所有层级 + Forward流程）"),
    (summary_report, "概览报告（3层深度 + Forward流程）"),
    (structure_report, "纯结构报告（所有层级，不含Forward流程）")
]

for i, (path, desc) in enumerate(reports, 1):
    size = os.path.getsize(path) / 1024  # KB
    print(f"{i}. {desc}")
    print(f"   📄 {path}")
    print(f"   💾 {size:.1f} KB")
    print()

print("="*100)
print("\n💡 提示：")
print("  - 使用文本编辑器打开 .txt 文件查看完整报告")
print("  - full 版本包含所有细节，适合深入分析")
print("  - summary 版本适合快速了解主要结构")
print("  - structure 版本更简洁，不含执行顺序信息")
print("="*100)
