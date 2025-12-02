@echo off
REM 完整的训练-评估-绘图示例 (Windows)

echo ==================================
echo 完整工作流示例
echo ==================================

REM 1. 训练模型（小规模，快速演示）
echo.
echo 步骤 1/3: 训练模型...
python Model_AIIC/test_separator.py ^
  --batches 50 ^
  --batch_size 128 ^
  --stages "2,3" ^
  --share_weights "False" ^
  --save_dir "./demo_exp"

if errorlevel 1 (
    echo 训练失败！
    exit /b 1
)

REM 2. 评估性能
echo.
echo 步骤 2/3: 评估性能...
python Model_AIIC/evaluate_models.py ^
  --exp_dir "./demo_exp" ^
  --tdl "A-30,B-100" ^
  --snr_range "30:-6:0" ^
  --num_samples 500 ^
  --output "./demo_results"

if errorlevel 1 (
    echo 评估失败！
    exit /b 1
)

REM 3. 绘制曲线
echo.
echo 步骤 3/3: 绘制曲线...

REM 单图
python Model_AIIC/plot_results.py ^
  --input "./demo_results" ^
  --layout single

REM 按 TDL 分图
python Model_AIIC/plot_results.py ^
  --input "./demo_results" ^
  --layout subplots_tdl

echo.
echo ==================================
echo 完成！查看结果：
echo   - 训练结果: ./demo_exp/
echo   - 评估数据: ./demo_results/evaluation_results.json
echo   - 图像文件: ./demo_results/*.png
echo ==================================

pause
