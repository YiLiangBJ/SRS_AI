"""
SNR-Aware Loss Functions for Channel Estimation

示例代码：在 test_separator.py 中添加 SNR 感知的损失函数
"""

import torch
import numpy as np


def calculate_loss(h_pred, h_targets, snr_db, loss_type='normalized'):
    """
    统一的损失计算函数，支持多种损失类型
    
    Args:
        h_pred: 预测的信道 (B, P, L) complex
        h_targets: 目标信道 (B, P, L) complex
        snr_db: SNR in dB，可以是:
                - float: 所有样本相同 SNR
                - list/array: 每个样本不同 SNR
        loss_type: 损失类型
                - 'nmse': 原始 NMSE（默认，无 SNR 考虑）
                - 'normalized': SNR 归一化损失（推荐）⭐
                - 'log': 对数空间（dB）损失
                - 'weighted': SNR 区间加权损失
    
    Returns:
        loss: 标量损失值
    """
    # 计算基础 NMSE
    mse = (h_pred - h_targets).abs().pow(2).mean()
    signal_power = h_targets.abs().pow(2).mean()
    nmse = mse / (signal_power + 1e-10)
    
    if loss_type == 'nmse':
        # 原始 NMSE（无 SNR 考虑）
        return nmse
    
    elif loss_type == 'normalized':
        # SNR 归一化损失（推荐）⭐
        # 核心思想：loss = actual_nmse / theoretical_best_nmse
        
        # 处理 SNR 输入
        if isinstance(snr_db, (list, np.ndarray, torch.Tensor)):
            # 每个样本不同 SNR，取平均
            if isinstance(snr_db, torch.Tensor):
                snr_db_mean = snr_db.float().mean().item()
            else:
                snr_db_mean = np.mean(snr_db)
        else:
            snr_db_mean = snr_db
        
        # 理论最优 NMSE（考虑噪声影响）
        # 在高斯噪声下，理论 MMSE ≈ σ²_noise / σ²_signal = 1 / (1 + SNR)
        snr_linear = 10 ** (snr_db_mean / 10)
        theoretical_nmse = 1.0 / (1.0 + snr_linear)
        
        # 归一化损失
        # - 如果模型达到理论最优，normalized_loss = 1.0
        # - 如果模型比理论差 2 倍，normalized_loss = 2.0
        normalized_loss = nmse / (theoretical_nmse + 1e-10)
        
        return normalized_loss
    
    elif loss_type == 'log':
        # 对数空间（dB）损失
        # 优点：自动压缩大值，放大小值
        nmse_db = 10 * torch.log10(nmse + 1e-10)
        
        # 可以选择：
        # 1. 直接用 dB 值（可能为负）
        # 2. 用绝对值
        # 3. 用平方（强调大误差）
        
        return nmse_db  # 或 abs(nmse_db) 或 nmse_db**2
    
    elif loss_type == 'weighted':
        # SNR 区间加权损失
        # 优点：可以针对应用场景调整优化重点
        
        # 处理 SNR 输入
        if isinstance(snr_db, (list, np.ndarray, torch.Tensor)):
            if isinstance(snr_db, torch.Tensor):
                snr_db_mean = snr_db.float().mean().item()
            else:
                snr_db_mean = np.mean(snr_db)
        else:
            snr_db_mean = snr_db
        
        # 根据 SNR 区间设置权重
        if snr_db_mean < 0:
            weight = 0.5  # 低 SNR: 降低权重（接近理论极限）
        elif snr_db_mean < 10:
            weight = 0.8  # 中低 SNR
        elif snr_db_mean < 20:
            weight = 1.0  # 中 SNR: 标准权重
        else:
            weight = 1.5  # 高 SNR: 增加权重（更有改进空间）
        
        return weight * nmse
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def calculate_loss_per_sample(h_pred, h_targets, snr_db_list, loss_type='normalized'):
    """
    Per-sample 损失计算（更精确，但计算量大）
    
    适用场景：batch 中每个样本有不同 SNR
    
    Args:
        h_pred: (B, P, L)
        h_targets: (B, P, L)
        snr_db_list: (B,) 每个样本的 SNR
        loss_type: 同上
    
    Returns:
        loss: 标量（batch 平均）
    """
    batch_size = h_pred.shape[0]
    loss_sum = 0.0
    
    for i in range(batch_size):
        # 单个样本的 NMSE
        mse_i = (h_pred[i] - h_targets[i]).abs().pow(2).mean()
        signal_power_i = h_targets[i].abs().pow(2).mean()
        nmse_i = mse_i / (signal_power_i + 1e-10)
        
        if loss_type == 'nmse':
            loss_i = nmse_i
        
        elif loss_type == 'normalized':
            snr_linear = 10 ** (snr_db_list[i] / 10)
            theoretical_nmse = 1.0 / (1.0 + snr_linear)
            loss_i = nmse_i / (theoretical_nmse + 1e-10)
        
        elif loss_type == 'log':
            nmse_db = 10 * torch.log10(nmse_i + 1e-10)
            loss_i = nmse_db
        
        elif loss_type == 'weighted':
            snr_db = snr_db_list[i]
            if snr_db < 0:
                weight = 0.5
            elif snr_db < 10:
                weight = 0.8
            elif snr_db < 20:
                weight = 1.0
            else:
                weight = 1.5
            loss_i = weight * nmse_i
        
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        loss_sum += loss_i
    
    return loss_sum / batch_size


def adaptive_snr_sampling(batch_size, snr_range=(0, 30), strategy='uniform_bins'):
    """
    自适应 SNR 采样策略
    
    Args:
        batch_size: 批大小
        snr_range: SNR 范围 (min, max)
        strategy: 采样策略
                - 'uniform': 均匀随机
                - 'uniform_bins': 均匀分桶（推荐）⭐
                - 'importance': 重要性采样（关注难点）
    
    Returns:
        snr_list: (batch_size,) 每个样本的 SNR
    """
    snr_min, snr_max = snr_range
    
    if strategy == 'uniform':
        # 简单均匀随机
        return np.random.uniform(snr_min, snr_max, batch_size)
    
    elif strategy == 'uniform_bins':
        # 均匀分桶（推荐）⭐
        # 保证每个 SNR 区间都有足够样本
        
        # 定义 SNR 区间
        snr_bins = np.linspace(snr_min, snr_max, 5)  # 4 个区间
        samples_per_bin = batch_size // (len(snr_bins) - 1)
        remainder = batch_size % (len(snr_bins) - 1)
        
        snr_list = []
        for i in range(len(snr_bins) - 1):
            bin_low, bin_high = snr_bins[i], snr_bins[i+1]
            n_samples = samples_per_bin + (1 if i < remainder else 0)
            snrs = np.random.uniform(bin_low, bin_high, n_samples)
            snr_list.extend(snrs)
        
        return np.array(snr_list)
    
    elif strategy == 'importance':
        # 重要性采样：更多样本集中在中间 SNR（最难的区域）
        # 使用 Beta 分布
        from scipy.stats import beta
        
        # Beta(2, 2) 在 [0, 1] 上中间密度大
        samples_01 = beta.rvs(2, 2, size=batch_size)
        snr_list = snr_min + samples_01 * (snr_max - snr_min)
        
        return snr_list
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# 使用示例
# ============================================================================

def training_loop_example():
    """
    使用 SNR 感知损失的训练循环示例
    """
    # ... 模型定义 ...
    # model = ResidualRefinementSeparator(...)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 配置
    num_batches = 1000
    batch_size = 128
    snr_range = (0, 30)
    loss_type = 'normalized'  # 选择损失类型
    
    for batch_idx in range(num_batches):
        # 1. 自适应 SNR 采样
        snr_batch = adaptive_snr_sampling(
            batch_size, 
            snr_range=snr_range, 
            strategy='uniform_bins'
        )
        
        # 2. 生成数据（每个样本不同 SNR）
        y, h_targets = generate_training_data(
            batch_size=batch_size,
            snr_db=snr_batch  # 传入 SNR 数组
        )
        
        # 3. 前向传播
        optimizer.zero_grad()
        h_pred = model(y)
        
        # 4. 计算损失（选择方法）
        
        # 方法 A: Batch 平均 SNR（快速）
        loss = calculate_loss(h_pred, h_targets, snr_batch, loss_type=loss_type)
        
        # 方法 B: Per-sample 精确计算（更准确）
        # loss = calculate_loss_per_sample(h_pred, h_targets, snr_batch, loss_type=loss_type)
        
        # 5. 反向传播
        loss.backward()
        optimizer.step()
        
        # 6. 记录（可选）
        if (batch_idx + 1) % 20 == 0:
            print(f"Batch {batch_idx+1}, Loss: {loss.item():.6f}, "
                  f"SNR range: [{snr_batch.min():.1f}, {snr_batch.max():.1f}] dB")


# ============================================================================
# 对比实验建议
# ============================================================================

def compare_loss_types():
    """
    对比不同损失类型的实验设置
    """
    loss_types = ['nmse', 'normalized', 'log', 'weighted']
    
    for loss_type in loss_types:
        print(f"\n{'='*60}")
        print(f"Training with loss_type='{loss_type}'")
        print(f"{'='*60}")
        
        # 训练
        model, losses = test_model(
            num_batches=1000,
            batch_size=128,
            snr_db=(0, 30),  # 范围 SNR
            loss_type=loss_type,
            save_dir=f'./experiments/loss_{loss_type}'
        )
        
        # 评估（在多个 SNR 点）
        evaluate_model_at_multiple_snrs(
            model,
            snr_list=[0, 5, 10, 15, 20, 25, 30],
            output_dir=f'./results/loss_{loss_type}'
        )


# ============================================================================
# 可视化损失行为
# ============================================================================

def visualize_loss_behavior():
    """
    可视化不同损失函数在不同 SNR 下的行为
    """
    import matplotlib.pyplot as plt
    
    # 模拟不同 SNR 和 NMSE
    snr_range = np.arange(-10, 35, 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for loss_type, ax in zip(['nmse', 'normalized', 'log', 'weighted'], axes.flat):
        losses_good = []
        losses_bad = []
        
        for snr_db in snr_range:
            # 理论最优
            snr_linear = 10 ** (snr_db / 10)
            theoretical_nmse = 1.0 / (1.0 + snr_linear)
            
            # 模拟两个模型：
            # - 好模型：接近理论最优
            # - 差模型：比理论差 3 倍
            nmse_good = theoretical_nmse * 1.2
            nmse_bad = theoretical_nmse * 3.0
            
            # 计算损失
            h_pred_good = torch.tensor(0.0)  # dummy
            h_targets = torch.tensor(0.0)    # dummy
            
            # Manually calculate loss
            if loss_type == 'nmse':
                loss_good = nmse_good
                loss_bad = nmse_bad
            elif loss_type == 'normalized':
                loss_good = nmse_good / theoretical_nmse
                loss_bad = nmse_bad / theoretical_nmse
            elif loss_type == 'log':
                loss_good = 10 * np.log10(nmse_good)
                loss_bad = 10 * np.log10(nmse_bad)
            elif loss_type == 'weighted':
                weight = 0.5 if snr_db < 0 else (0.8 if snr_db < 10 else (1.0 if snr_db < 20 else 1.5))
                loss_good = weight * nmse_good
                loss_bad = weight * nmse_bad
            
            losses_good.append(loss_good)
            losses_bad.append(loss_bad)
        
        # 绘图
        ax.plot(snr_range, losses_good, 'g-', label='Good Model', linewidth=2)
        ax.plot(snr_range, losses_bad, 'r-', label='Bad Model', linewidth=2)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(f'Loss ({loss_type})')
        ax.set_title(f'Loss Type: {loss_type}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_behavior_comparison.png', dpi=150)
    print("Saved: loss_behavior_comparison.png")


if __name__ == '__main__':
    # 可视化损失行为
    visualize_loss_behavior()
    
    print("\n" + "="*60)
    print("SNR-Aware Loss Functions - Usage Examples")
    print("="*60)
    print("\n1. 在 test_separator.py 中替换损失计算:")
    print("   loss = calculate_loss(h_pred, h_targets, snr_db, loss_type='normalized')")
    print("\n2. 使用自适应 SNR 采样:")
    print("   snr_batch = adaptive_snr_sampling(batch_size, (0, 30), 'uniform_bins')")
    print("\n3. 对比实验:")
    print("   python test_separator.py --loss_type normalized ...")
    print("="*60)
