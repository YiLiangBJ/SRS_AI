"""
测试参数量显示功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_AIIC.plot_results import format_num_params, get_model_label

# 测试参数量格式化
print("="*60)
print("测试参数量格式化")
print("="*60)

test_cases = [
    (None, "N/A"),
    (500, "500"),
    (1_500, "1.5K"),
    (104_640, "104.6K"),
    (156_960, "157.0K"),
    (1_234_567, "1.2M"),
    (10_000_000, "10.0M"),
]

for num_params, expected in test_cases:
    result = format_num_params(num_params)
    status = "✓" if result == expected else "✗"
    num_str = str(num_params) if num_params is not None else "None"
    print(f"{status} {num_str:>12} -> {result:>8} (expected: {expected})")

print()

# 测试模型标签生成
print("="*60)
print("测试模型标签生成")
print("="*60)

configs = [
    {
        'model_name': 'stages=2_share=False',
        'config': {'num_params': 104_640}
    },
    {
        'model_name': 'stages=3_share=True',
        'config': {'num_params': 156_960}
    },
    {
        'model_name': 'stages=4_share=False',
        'config': {'num_params': 209_280}
    },
    {
        'model_name': 'stages=2_share=False_loss=normalized',
        'config': {}  # 没有 num_params
    }
]

for item in configs:
    label = get_model_label(item['model_name'], item['config'])
    print(f"  {label}")

print()
print("="*60)
print("✓ 所有测试完成！")
print("="*60)
