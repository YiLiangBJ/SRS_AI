import torch
import argparse
from train import SRSTrainer
from config import SRSConfig


def main():
    """训练使用自定义配置的 MMSE 矩阵"""
    
    # 创建一个空的命名空间对象，用于存放参数
    args = argparse.Namespace()
    
    # 直接设置参数，不需要通过命令行指定
    args.epochs = 50              # 训练轮数
    args.train_batches = 100      # 每轮训练的批次数
    args.val_batches = 20         # 每轮验证的批次数
    args.batch_size = 16          # 批次大小
    args.save_dir = './custom_mmse_checkpoints'  # 保存检查点的目录
    args.load_checkpoint = None   # 不加载现有检查点
    args.enable_plotting = False  # 禁用绘图
    
    # 创建自定义配置
    config = SRSConfig(
        seq_length=1200,
        ktc=4,  # K=12
        num_users=3,
        ports_per_user=[2, 3, 1],  # 不同用户有不同数量的端口
        cyclic_shifts=[
            [0, 6],     # 用户0的端口移位 (2个端口)
            [2, 4, 8],  # 用户1的端口移位 (3个端口)
            [10]        # 用户2的端口移位 (1个端口)
        ],
        mmse_block_size=12  # MMSE 滤波器块大小
    )
    
    # 创建训练器，始终使用可训练的MMSE模块，默认禁用绘图功能
    trainer = SRSTrainer(
        config=config,
        save_dir=args.save_dir,
        use_trainable_mmse=True,  # 总是使用可训练的MMSE模块
        enable_plotting=args.enable_plotting  # 使用命令行参数控制是否启用绘图功能
    )
    
    # 如果指定，加载检查点
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # 训练
    trainer.train(
        num_epochs=args.epochs,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
