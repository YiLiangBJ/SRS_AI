"""Thin CLI entrypoint for the refactored training workflow."""

import argparse
from pathlib import Path

from workflows import TrainRequest, run_training_experiment


DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / 'experiments_refactored'


def build_parser():
    """Build the train CLI parser."""
    parser = argparse.ArgumentParser(description='Train channel separator models')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name from experiments.yaml')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num_batches', type=int, default=None, help='Override number of batches')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, cuda:0, ...)')
    parser.add_argument('--save_dir', type=str, default=str(DEFAULT_SAVE_DIR), help='Parent directory where timestamped experiment directories are created')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false', help='Disable mixed precision training (FP16)')
    parser.add_argument('--no-compile', dest='compile_model', action='store_false', help='Disable model compilation (torch.compile)')
    parser.set_defaults(use_amp=True, compile_model=None)
    parser.add_argument('--eval_after_train', action='store_true', help='自动评估训练后的模型')
    parser.add_argument('--eval_snr_range', type=str, default='30:-3:0', help='评估SNR设置，支持 "30:-3:0" 或 "30,25,20,15,10,5,0"')
    parser.add_argument('--eval_tdl', type=str, default='A-30,B-100,C-300', help='评估TDL配置 (逗号分隔)')
    parser.add_argument('--eval_num_batches', type=int, default=100, help='评估批次数')
    parser.add_argument('--eval_batch_size', type=int, default=2048, help='评估批大小')
    parser.add_argument('--plot_after_eval', action='store_true', help='评估后自动绘图')
    parser.add_argument('--export_onnx_after_train', action='store_true', help='训练完成后导出 ONNX 供 Matlab/部署使用')
    parser.add_argument('--onnx_export_selection', type=str, default='best', choices=['best', 'all'], help='导出最佳 run 或全部 run')
    parser.add_argument('--onnx_output_dir', type=str, default=None, help='ONNX 导出目录（仅单 run 自定义导出时使用；多 run 默认各自写入 run/onnx_exports）')
    parser.add_argument('--onnx_opset', type=int, default=13, help='ONNX opset 版本，默认 13 兼顾 Matlab 兼容性')
    parser.add_argument('--onnx_batch_size', type=int, default=1, help='ONNX tracing 用的 dummy batch size')
    onnx_batch_group = parser.add_mutually_exclusive_group()
    onnx_batch_group.add_argument('--onnx_dynamic_batch', dest='onnx_dynamic_batch', action='store_true', help='导出动态 batch 维（默认）')
    onnx_batch_group.add_argument('--onnx_fixed_batch', dest='onnx_dynamic_batch', action='store_false', help='导出固定 batch 维')
    parser.set_defaults(onnx_dynamic_batch=True)
    parser.add_argument('--onnx_validate', action='store_true', help='导出后运行 ONNX checker / ONNX Runtime 烟雾验证')
    parser.add_argument('--export_matlab_after_train', action='store_true', help='训练完成后导出 Matlab 显式权重 bundle')
    parser.add_argument('--matlab_export_selection', type=str, default='best', choices=['best', 'all'], help='导出最佳 run 或全部 run 的 Matlab bundle')
    parser.add_argument('--matlab_output_dir', type=str, default=None, help='Matlab bundle 导出目录（仅单 run 自定义导出时使用；多 run 默认各自写入 run/matlab_exports）')
    parser.add_argument('--plan_only', action='store_true', help='仅解析并打印实验计划，不执行训练')
    return parser


def main():
    """Parse CLI args and dispatch to the workflow layer."""
    args = build_parser().parse_args()
    request = TrainRequest.from_namespace(args)
    run_training_experiment(request)


if __name__ == '__main__':
    main()
