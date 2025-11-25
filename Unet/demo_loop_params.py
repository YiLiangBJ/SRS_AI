"""
演示：为什么循环参数量计算需要注意
"""

from complexUnet import ComplexResidualUNet

print("\n" + "="*80)
print(" " * 25 + "🔢 循环参数量分析")
print("="*80)

model = ComplexResidualUNet(input_channels=2, output_channels=1, base_channels=8, depth=2)

print("\n❌ 错误假设：每次循环参数量相同")
print("-" * 80)
print("  假设 enc_blocks[i] 每次都是 624 params")
print("  假设 down_samples[i] 每次都是 272 params")
print("  假设 up_samples[i] 每次都是 2,080 params")
print("  假设 dec_blocks[i] 每次都是 5,856 params")
print()
print("  错误计算：")
print("    编码器: (624 + 272) × 2 = 1,792")
print("    解码器: (2,080 + 5,856) × 2 = 15,872")
print("    瓶颈 + 最终: 10,816 + 18 = 10,834")
print("    ─────────────────────────────")
print("    总计: 28,498 params  ← 错误！")

print("\n" + "="*80)
print("✅ 正确理解：每次循环通道数不同，参数量也不同")
print("="*80)

# 编码器
print("\n【编码器循环】depth=2")
enc_total = 0
for i in range(2):
    enc_params = sum(p.numel() for p in model.enc_blocks[i].parameters())
    down_params = sum(p.numel() for p in model.down_samples[i].parameters())
    
    # 获取通道信息
    in_ch = model.enc_blocks[i].conv1.conv_real.in_channels
    out_ch = model.enc_blocks[i].conv1.conv_real.out_channels
    
    print(f"\n  迭代 i={i}:")
    print(f"    enc_blocks[{i}]:   {in_ch:2d}→{out_ch:2d} channels = {enc_params:>6,} params")
    print(f"    down_samples[{i}]: {out_ch:2d}→{out_ch:2d} channels = {down_params:>6,} params")
    print(f"    ─────────────────────────────────────")
    print(f"    小计:                       {enc_params + down_params:>6,} params")
    
    enc_total += enc_params + down_params

print(f"\n  编码器总计: {enc_total:,} params")

# 瓶颈
bottleneck_params = sum(p.numel() for p in model.bottleneck.parameters())
in_ch = model.bottleneck.conv1.conv_real.in_channels
out_ch = model.bottleneck.conv1.conv_real.out_channels
print(f"\n【瓶颈层】")
print(f"  bottleneck: {in_ch:2d}→{out_ch:2d} channels = {bottleneck_params:,} params")

# 解码器
print(f"\n【解码器循环】depth=2")
dec_total = 0
for i in range(2):
    up_params = sum(p.numel() for p in model.up_samples[i].parameters())
    dec_params = sum(p.numel() for p in model.dec_blocks[i].parameters())
    
    # 获取通道信息
    in_ch_up = model.up_samples[i].conv_real.in_channels
    out_ch_up = model.up_samples[i].conv_real.out_channels
    in_ch_dec = model.dec_blocks[i].conv1.conv_real.in_channels
    out_ch_dec = model.dec_blocks[i].conv1.conv_real.out_channels
    
    print(f"\n  迭代 i={i}:")
    print(f"    up_samples[{i}]:   {in_ch_up:2d}→{out_ch_up:2d} channels = {up_params:>6,} params")
    print(f"    dec_blocks[{i}]:   {in_ch_dec:2d}→{out_ch_dec:2d} channels = {dec_params:>6,} params")
    print(f"    ─────────────────────────────────────")
    print(f"    小计:                       {up_params + dec_params:>6,} params")
    
    dec_total += up_params + dec_params

print(f"\n  解码器总计: {dec_total:,} params")

# 最终
final_params = sum(p.numel() for p in model.final_conv.parameters())
print(f"\n【最终卷积】")
print(f"  final_conv: 8→1 channels = {final_params} params")

# 总计
total = sum(p.numel() for p in model.parameters())
manual_total = enc_total + bottleneck_params + dec_total + final_params

print("\n" + "="*80)
print("📊 总计对比")
print("="*80)
print(f"  手动累加: {manual_total:,} params")
print(f"  模型统计: {total:,} params")
print(f"  差异:     {abs(total - manual_total)} params  ← {'✓ 一致！' if total == manual_total else '✗ 不一致'}")

print("\n" + "="*80)
print("💡 关键理解")
print("="*80)
print("""
1. U-Net的编码器：通道数逐层增加 (2→8→16→32)
   - 越深的层，参数越多（与通道数平方成正比）

2. U-Net的解码器：通道数逐层减少 (32→16→8→1)
   - 越浅的层，参数越少

3. 因此：
   - enc_blocks[0] (2→8)  < enc_blocks[1] (8→16)
   - dec_blocks[0] (32→16) > dec_blocks[1] (16→8)

4. 工具现在显示：
   - "循环总参数" - 所有迭代的总和
   - "首次迭代示例" - 仅显示i=0的参数量
   - 注意提示 - 提醒参数量可能变化
""")

print("="*80)
