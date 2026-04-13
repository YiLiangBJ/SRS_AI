# 在训练脚本中添加线程监控（可选）
# 
# 添加到 test_separator.py 的训练循环中

# 在文件开头添加
try:
    from .thread_monitor import ThreadMonitor
    THREAD_MONITOR_AVAILABLE = True
except ImportError:
    THREAD_MONITOR_AVAILABLE = False
    ThreadMonitor = None

# 在 argparse 部分添加
parser.add_argument('--monitor_threads', action='store_true',
                   help='Enable thread usage monitoring (for debugging)')

# 在训练开始前
if args.monitor_threads and THREAD_MONITOR_AVAILABLE:
    print("📊 Thread monitoring enabled")
    thread_monitor = ThreadMonitor(sample_interval=0.05)
    thread_monitor.start()
else:
    thread_monitor = None

# 在训练循环中
for batch_idx in range(num_batches):
    # Data generation
    if thread_monitor:
        thread_monitor.set_phase('data')
    
    t0 = time.time()
    y, h_targets, _, _ = generate_training_data(...)
    data_gen_time += time.time() - t0
    
    # Forward
    if thread_monitor:
        thread_monitor.set_phase('forward')
    
    t0 = time.time()
    optimizer.zero_grad()
    h_pred = model(y)
    forward_time += time.time() - t0
    
    # Loss
    loss = calculate_loss(h_pred, h_targets, snr_db, loss_type=loss_type)
    
    # Backward
    if thread_monitor:
        thread_monitor.set_phase('backward')
    
    t0 = time.time()
    loss.backward()
    optimizer.step()
    backward_time += time.time() - t0
    
    if thread_monitor:
        thread_monitor.set_phase('idle')

# 训练结束后
if thread_monitor:
    thread_monitor.stop()
    print("\n")
    thread_monitor.print_report()

# 使用方法：
# python ./Model_AIIC_onnx/test_separator.py --batches 100 --monitor_threads
