"""Standalone latency benchmarking workflow for trained runs."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import re
import socket
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import (
    build_dummy_input,
    discover_run_dirs,
    find_checkpoint_path,
    load_run_artifacts,
    load_trained_model_from_run,
    outside_legend_figure_size,
    place_legend_outside_right,
    resolve_existing_path,
    resolve_run_selection,
    style_for_series,
)


DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_CPU_PRECISIONS = ['fp32', 'bf16']
DEFAULT_CUDA_PRECISIONS = ['fp32', 'fp16', 'bf16']
DEFAULT_CPU_EXECUTION_MODES = ['eager', 'jit', 'compile']
DEFAULT_CUDA_EXECUTION_MODES = ['eager']
DEFAULT_CPU_RUNTIME_BACKENDS = ['pytorch']
DEFAULT_CUDA_RUNTIME_BACKENDS = ['pytorch']


def _runtime_backend_execution_modes(runtime_backend: str, execution_modes: List[str]) -> List[str]:
    if runtime_backend == 'pytorch':
        return execution_modes
    if runtime_backend == 'onnxruntime':
        return ['onnxruntime']
    if runtime_backend == 'openvino':
        return ['openvino']
    return execution_modes


def _runtime_backend_precision_profiles(runtime_backend: str, precision_profiles: List[str]) -> List[str]:
    if runtime_backend in {'onnxruntime', 'openvino'}:
        return [precision for precision in precision_profiles if precision == 'fp32']
    return precision_profiles


@dataclass(frozen=True)
class LatencyTask:
    run_dir: str
    device: str
    runtime_backend: str
    execution_mode: str
    precision: str
    batch_size: int
    num_threads: int
    warmup_iters: int
    measure_iters: int


def normalize_latency_selection(exp_dir=None, run_dir=None, run_dirs=None, runs=None):
    """Normalize selectors for latency benchmarking with benchmark-specific ergonomics."""
    if runs and not exp_dir:
        raise ValueError('--runs requires --exp_dir')

    if run_dir is not None:
        resolved = resolve_existing_path(run_dir)
        if isinstance(resolved, tuple):
            return exp_dir, run_dir, run_dirs, runs
        if resolved.is_dir() and find_checkpoint_path(resolved) is None:
            nested_run_dirs = discover_run_dirs(resolved)
            if nested_run_dirs:
                return str(resolved), None, None, runs

    if exp_dir is not None:
        resolved = resolve_existing_path(exp_dir)
        if not isinstance(resolved, tuple):
            benchmarkable_runs = discover_run_dirs(resolved)
            if not benchmarkable_runs:
                raise FileNotFoundError(
                    'No benchmarkable run directories were found under the given experiment directory. '
                    'Expected child run directories containing model checkpoints.'
                )

    return exp_dir, run_dir, run_dirs, runs


def _available_cpu_count() -> int:
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _physical_cpu_count() -> int:
    try:
        import psutil  # type: ignore

        physical = psutil.cpu_count(logical=False)
        if physical:
            return int(physical)
    except Exception:
        pass
    logical = _available_cpu_count()
    return max(1, logical // 2)


def default_thread_counts(device_type: str) -> List[int]:
    if device_type != 'cpu':
        return [1]
    counts = [1, 2, 4, 8]
    sanitized = [count for count in counts if count <= _available_cpu_count()]
    return list(dict.fromkeys(max(1, int(count)) for count in sanitized))


def parse_csv_ints(value: Optional[str], default: Iterable[int]) -> List[int]:
    if value is None:
        return list(default)
    tokens = [token.strip() for token in str(value).split(',') if token.strip()]
    resolved: List[int] = []
    for token in tokens:
        if token == 'all-physical':
            resolved.append(_physical_cpu_count())
        else:
            resolved.append(int(token))
    return list(dict.fromkeys(max(1, item) for item in resolved))


def generate_product_batch_sizes(batch_antennas: Optional[str], batch_rbgs: Optional[str]) -> List[int]:
    if batch_antennas is None and batch_rbgs is None:
        return []
    if not batch_antennas or not batch_rbgs:
        raise ValueError('batch_antennas and batch_rbgs must be provided together')
    antennas = parse_csv_ints(batch_antennas, [])
    rbgs = parse_csv_ints(batch_rbgs, [])
    if not antennas or not rbgs:
        raise ValueError('batch_antennas and batch_rbgs must both resolve to at least one positive integer')
    return sorted({antenna_count * rbg_count for antenna_count in antennas for rbg_count in rbgs})


def resolve_batch_sizes(
    batch_sizes: Optional[str],
    batch_antennas: Optional[str] = None,
    batch_rbgs: Optional[str] = None,
) -> List[int]:
    if batch_antennas is None and batch_rbgs is None:
        return parse_csv_ints(batch_sizes, DEFAULT_BATCH_SIZES)

    manual_sizes = parse_csv_ints(batch_sizes, []) if batch_sizes is not None else []
    generated_sizes = generate_product_batch_sizes(batch_antennas, batch_rbgs)
    return sorted({*manual_sizes, *generated_sizes})


def parse_precision_profiles(device_type: str, value: Optional[str]) -> List[str]:
    if value is None:
        return list(DEFAULT_CPU_PRECISIONS if device_type == 'cpu' else DEFAULT_CUDA_PRECISIONS)
    profiles = [token.strip().lower() for token in str(value).split(',') if token.strip()]
    return list(dict.fromkeys(profiles))


def parse_execution_modes(device_type: str, value: Optional[str]) -> List[str]:
    if value is None:
        return list(DEFAULT_CPU_EXECUTION_MODES if device_type == 'cpu' else DEFAULT_CUDA_EXECUTION_MODES)
    modes = [token.strip().lower() for token in str(value).split(',') if token.strip()]
    return list(dict.fromkeys(modes))


def parse_runtime_backends(device_type: str, value: Optional[str]) -> List[str]:
    if value is None:
        return list(DEFAULT_CPU_RUNTIME_BACKENDS if device_type == 'cpu' else DEFAULT_CUDA_RUNTIME_BACKENDS)
    backends = [token.strip().lower() for token in str(value).split(',') if token.strip()]
    return list(dict.fromkeys(backends))


def resolve_latency_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _build_scope_label(run_dirs: List[Path]) -> str:
    if not run_dirs:
        return 'no-runs'
    if len(run_dirs) == 1:
        return run_dirs[0].name
    joined_names = '|'.join(run_dir.name for run_dir in sorted(run_dirs, key=lambda path: path.name))
    digest = hashlib.sha1(joined_names.encode('utf-8')).hexdigest()[:10]
    return f'all-runs_{len(run_dirs)}runs_{digest}'


def resolve_latency_output_dir(exp_dir: Path | None = None, run_dirs: Optional[List[Path]] = None, explicit_output=None, device_type: str = 'cpu', benchmark_id: Optional[str] = None) -> Path:
    if explicit_output is not None:
        return Path(explicit_output)
    timestamp = benchmark_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dirs = run_dirs or []
    if exp_dir is not None:
        return Path(exp_dir) / 'latency' / f'{timestamp}_{_build_scope_label(run_dirs)}_{device_type}'
    if len(run_dirs) == 1:
        return run_dirs[0] / 'latency' / f'{timestamp}_{device_type}'
    if run_dirs:
        common_parent = run_dirs[0].parent
        if all(run_dir.parent == common_parent for run_dir in run_dirs):
            return common_parent / 'latency' / f'{timestamp}_{_build_scope_label(run_dirs)}_{device_type}'
    return Path('latency') / f'{timestamp}_{_build_scope_label(run_dirs)}_{device_type}'


def build_latency_task_matrix(
    run_dirs: List[Path],
    device: torch.device,
    runtime_backends: List[str],
    execution_modes: List[str],
    precision_profiles: List[str],
    batch_sizes: List[int],
    thread_counts: List[int],
    warmup_iters: int,
    measure_iters: int,
) -> List[LatencyTask]:
    tasks: List[LatencyTask] = []
    active_threads = thread_counts if device.type == 'cpu' else [1]
    for run_dir in run_dirs:
        for runtime_backend in runtime_backends:
            backend_execution_modes = _runtime_backend_execution_modes(runtime_backend, execution_modes)
            backend_precision_profiles = _runtime_backend_precision_profiles(runtime_backend, precision_profiles)
            for execution_mode in backend_execution_modes:
                for precision in backend_precision_profiles:
                    for batch_size in batch_sizes:
                        for num_threads in active_threads:
                            tasks.append(LatencyTask(
                                run_dir=str(run_dir),
                                device=str(device),
                                runtime_backend=runtime_backend,
                                execution_mode=execution_mode,
                                precision=precision,
                                batch_size=batch_size,
                                num_threads=num_threads,
                                warmup_iters=warmup_iters,
                                measure_iters=measure_iters,
                            ))
    return tasks


def _optional_package_version(package_name: str) -> Optional[str]:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return None


def _torch_config_summary() -> str:
    try:
        return torch.__config__.show()
    except Exception:
        return 'Unavailable'


def _cpu_model_name() -> Optional[str]:
    cpuinfo_path = Path('/proc/cpuinfo')
    if not cpuinfo_path.exists():
        return None
    with open(cpuinfo_path, 'r', encoding='utf-8', errors='ignore') as input_file:
        for line in input_file:
            if line.lower().startswith('model name'):
                return line.split(':', 1)[1].strip()
    return None


def _cpu_flags() -> List[str]:
    cpuinfo_path = Path('/proc/cpuinfo')
    if not cpuinfo_path.exists():
        return []
    with open(cpuinfo_path, 'r', encoding='utf-8', errors='ignore') as input_file:
        for line in input_file:
            if line.lower().startswith('flags'):
                return line.split(':', 1)[1].strip().split()
    return []


def _cpu_capability() -> Optional[str]:
    backend = getattr(torch.backends, 'cpu', None)
    getter = getattr(backend, 'get_cpu_capability', None)
    if getter is None:
        return None
    try:
        return str(getter())
    except Exception:
        return None


def _onednn_version(torch_config_summary: str) -> Optional[str]:
    match = re.search(r'Intel MKL-DNN v([^\s]+)', torch_config_summary)
    if match:
        return match.group(1)
    match = re.search(r'oneDNN v([^\s]+)', torch_config_summary)
    if match:
        return match.group(1)
    return None


def _hardware_manifest(device: torch.device, num_threads: int, precision: str, runtime_backend: str = 'pytorch') -> Dict[str, Any]:
    torch_config_summary = _torch_config_summary()
    cpu_flags = _cpu_flags()
    manifest: Dict[str, Any] = {
        'hostname': socket.gethostname(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'device': str(device),
        'runtime_backend': runtime_backend,
        'precision_profile': precision,
        'num_threads': int(num_threads),
        'logical_cpu_count': _available_cpu_count(),
        'physical_cpu_count': _physical_cpu_count(),
        'cpu_model_name': _cpu_model_name(),
        'cpu_capability': _cpu_capability(),
        'cpu_flags': cpu_flags,
        'cpu_flag_summary': [flag for flag in ['avx2', 'avx512f', 'avx512bw', 'avx512vl', 'avx512_vnni', 'avx512_bf16', 'amx_bf16', 'amx_int8', 'amx_tile', 'fma'] if flag in cpu_flags],
        'mkldnn_available': bool(hasattr(torch.backends, 'mkldnn')),
        'mkldnn_enabled': bool(getattr(torch.backends, 'mkldnn', None) and torch.backends.mkldnn.enabled),
        'onednn_version': _onednn_version(torch_config_summary),
        'torch_compile_available': bool(hasattr(torch, 'compile')),
        'torch_config_summary': torch_config_summary,
        'env': {
            key: os.environ.get(key)
            for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'KMP_AFFINITY', 'CUDA_VISIBLE_DEVICES']
            if os.environ.get(key) is not None
        },
    }
    if runtime_backend == 'onnxruntime':
        manifest['onnxruntime_available'] = importlib.util.find_spec('onnxruntime') is not None
        manifest['onnxruntime_version'] = _optional_package_version('onnxruntime')
    if runtime_backend == 'openvino':
        manifest['openvino_available'] = importlib.util.find_spec('openvino') is not None
        manifest['openvino_version'] = _optional_package_version('openvino')
    if device.type == 'cuda' and torch.cuda.is_available():
        manifest['cuda'] = {
            'device_name': torch.cuda.get_device_name(device),
            'device_capability': list(torch.cuda.get_device_capability(device)),
            'cuda_version': torch.version.cuda,
            'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
            'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
        }
    return manifest


def _materialize_precision(
    device: torch.device,
    precision: str,
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
):
    if precision == 'fp32':
        return model, dummy_input, None, 'float32'
    if device.type == 'cpu' and precision == 'bf16':
        return model.to(dtype=torch.bfloat16), dummy_input.to(dtype=torch.bfloat16), None, 'bfloat16'
    if device.type == 'cuda' and precision == 'fp16':
        return model, dummy_input, torch.autocast(device_type='cuda', dtype=torch.float16), 'float16'
    if device.type == 'cuda' and precision == 'bf16':
        return model, dummy_input, torch.autocast(device_type='cuda', dtype=torch.bfloat16), 'bfloat16'
    raise ValueError(f'Unsupported precision profile {precision!r} for device {device}')


def _validate_execution_mode_support(device: torch.device, execution_mode: str) -> Optional[str]:
    supported_modes = DEFAULT_CPU_EXECUTION_MODES if device.type == 'cpu' else DEFAULT_CUDA_EXECUTION_MODES
    if execution_mode not in supported_modes:
        return f'Execution mode {execution_mode} is not supported for device {device.type}'
    if execution_mode == 'compile' and not hasattr(torch, 'compile'):
        return 'torch.compile is not available in this PyTorch build'
    return None


def _validate_runtime_backend_support(device: torch.device, runtime_backend: str) -> Optional[str]:
    if runtime_backend == 'pytorch':
        return None
    if runtime_backend == 'onnxruntime':
        if device.type != 'cpu':
            return 'ONNX Runtime benchmark currently supports CPU only'
        if importlib.util.find_spec('onnxruntime') is None:
            return 'onnxruntime is not installed in the current Python environment'
        return None
    if runtime_backend == 'openvino':
        if device.type != 'cpu':
            return 'OpenVINO benchmark currently supports CPU only'
        if importlib.util.find_spec('openvino') is None:
            return 'openvino is not installed in the current Python environment'
        return None
    return f'Unsupported runtime backend {runtime_backend!r}'


def _prepare_model_for_execution_mode(model: torch.nn.Module, dummy_input: torch.Tensor, execution_mode: str):
    if execution_mode == 'eager':
        return model, 0.0

    start_ns = time.perf_counter_ns()
    if execution_mode == 'jit':
        optimized = torch.jit.trace(model, dummy_input, strict=False).eval()
        try:
            optimized = torch.jit.freeze(optimized)
        except Exception:
            pass
        try:
            optimized = torch.jit.optimize_for_inference(optimized)
        except Exception:
            pass
        prep_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        return optimized, prep_ms

    if execution_mode == 'compile':
        compiled = torch.compile(model, mode='reduce-overhead')
        prep_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        return compiled, prep_ms

    raise ValueError(f'Unsupported execution mode {execution_mode!r}')


def _validate_precision_support(device: torch.device, precision: str, runtime_backend: str = 'pytorch') -> Optional[str]:
    if runtime_backend == 'onnxruntime':
        if precision != 'fp32':
            return 'ONNX Runtime benchmark currently supports precision profile fp32 only'
        return None
    if runtime_backend == 'openvino':
        if precision != 'fp32':
            return 'OpenVINO benchmark currently supports precision profile fp32 only'
        return None
    if precision == 'fp32':
        return None
    if device.type == 'cpu':
        if precision != 'bf16':
            return f'CPU benchmark does not support precision profile {precision}'
        if not (getattr(torch.backends, 'mkldnn', None) and torch.backends.mkldnn.enabled):
            return 'mkldnn/oneDNN is disabled'
        return None
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            return 'CUDA is not available'
        if precision == 'fp16':
            return None
        if precision == 'bf16':
            try:
                if not torch.cuda.is_bf16_supported():
                    return 'CUDA device does not support bf16'
            except Exception:
                return 'Unable to determine CUDA bf16 support'
            return None
        return f'CUDA benchmark does not support precision profile {precision}'
    return f'Unsupported device type {device.type}'


def _build_skipped_result(task: LatencyTask, device: torch.device, skip_reason: str) -> Dict[str, Any]:
    return {
        'status': 'skipped',
        'skip_reason': skip_reason,
        'run_dir': task.run_dir,
        'run_name': Path(task.run_dir).name,
        'device': task.device,
        'runtime_backend': task.runtime_backend,
        'execution_mode': task.execution_mode,
        'requested_precision_profile': task.precision,
        'effective_execution_dtype': None,
        'graph_prep_time_ms': 0.0,
        'batch_size': task.batch_size,
        'num_threads': task.num_threads,
        'hardware_manifest': _hardware_manifest(device, task.num_threads, task.precision, runtime_backend=task.runtime_backend),
    }


def _ensure_onnxruntime_export(run_dir: Path) -> Dict[str, Any]:
    export_dir = run_dir / 'onnx_exports'
    manifest_path = export_dir / 'export_manifest.json'
    onnx_path = export_dir / f'{run_dir.name}.onnx'
    if manifest_path.exists() and onnx_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as input_file:
            return json.load(input_file)

    from workflows.export_workflow import export_run_to_onnx

    return export_run_to_onnx(
        run_dir=run_dir,
        output_root=None,
        opset_version=13,
        batch_size=1,
        dynamic_batch=True,
        validate=False,
    )


def _ensure_openvino_export(run_dir: Path) -> Dict[str, Any]:
    return _ensure_onnxruntime_export(run_dir)


def _measure_pytorch_model(task: LatencyTask, device: torch.device) -> Dict[str, Any]:
    mode_skip_reason = _validate_execution_mode_support(device, task.execution_mode)
    if mode_skip_reason is not None:
        return _build_skipped_result(task, device, mode_skip_reason)
    skip_reason = _validate_precision_support(device, task.precision, runtime_backend=task.runtime_backend)
    if skip_reason is not None:
        return _build_skipped_result(task, device, skip_reason)

    if device.type == 'cpu':
        torch.set_num_threads(task.num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    model, artifacts = load_trained_model_from_run(task.run_dir, device=device)
    model.eval()
    dummy_input = build_dummy_input(
        artifacts.model_spec,
        batch_size=task.batch_size,
        component_specs=artifacts.component_specs,
    ).to(device)
    model, dummy_input, autocast_context, effective_dtype = _materialize_precision(
        device=device,
        precision=task.precision,
        model=model,
        dummy_input=dummy_input,
    )
    model, graph_prep_time_ms = _prepare_model_for_execution_mode(model, dummy_input, task.execution_mode)

    def _run_once() -> None:
        with torch.inference_mode():
            if autocast_context is None:
                _ = model(dummy_input)
            else:
                with autocast_context:
                    _ = model(dummy_input)

    for _ in range(task.warmup_iters):
        _run_once()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    samples_ns: List[int] = []
    for _ in range(task.measure_iters):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start_ns = time.perf_counter_ns()
        _run_once()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        end_ns = time.perf_counter_ns()
        samples_ns.append(end_ns - start_ns)

    latency_ms = [sample / 1_000_000.0 for sample in samples_ns]
    mean_ms = statistics.mean(latency_ms)
    throughput = task.batch_size / (mean_ms / 1000.0) if mean_ms > 0 else 0.0

    return {
        'status': 'ok',
        'skip_reason': None,
        'run_dir': task.run_dir,
        'run_name': Path(task.run_dir).name,
        'device': task.device,
        'runtime_backend': task.runtime_backend,
        'execution_mode': task.execution_mode,
        'requested_precision_profile': task.precision,
        'effective_execution_dtype': effective_dtype,
        'graph_prep_time_ms': float(graph_prep_time_ms),
        'batch_size': task.batch_size,
        'num_threads': task.num_threads,
        'warmup_iters': task.warmup_iters,
        'measure_iters': task.measure_iters,
        'p50_latency_ms': float(np.percentile(latency_ms, 50)),
        'p90_latency_ms': float(np.percentile(latency_ms, 90)),
        'p95_latency_ms': float(np.percentile(latency_ms, 95)),
        'p99_latency_ms': float(np.percentile(latency_ms, 99)),
        'mean_latency_ms': float(mean_ms),
        'std_latency_ms': float(statistics.pstdev(latency_ms) if len(latency_ms) > 1 else 0.0),
        'min_latency_ms': float(min(latency_ms)),
        'max_latency_ms': float(max(latency_ms)),
        'throughput_samples_per_sec': float(throughput),
        'latency_samples_ms': latency_ms,
        'hardware_manifest': _hardware_manifest(device, task.num_threads, task.precision, runtime_backend=task.runtime_backend),
        'model_spec': artifacts.model_spec,
        'metadata': artifacts.metadata,
        'component_specs': artifacts.component_specs,
    }


def _measure_onnxruntime_model(task: LatencyTask, device: torch.device) -> Dict[str, Any]:
    skip_reason = _validate_precision_support(device, task.precision, runtime_backend=task.runtime_backend)
    if skip_reason is not None:
        return _build_skipped_result(task, device, skip_reason)

    import onnxruntime as ort

    run_dir = Path(task.run_dir)
    artifacts = load_run_artifacts(run_dir, device='cpu')
    export_manifest = _ensure_onnxruntime_export(run_dir)
    dummy_input = build_dummy_input(
        artifacts.model_spec,
        batch_size=task.batch_size,
        component_specs=artifacts.component_specs,
    ).cpu()
    input_feed = {export_manifest['input_names'][0]: dummy_input.detach().numpy()}

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(task.num_threads)
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    start_ns = time.perf_counter_ns()
    session = ort.InferenceSession(
        str(export_manifest['onnx_path']),
        sess_options=session_options,
        providers=['CPUExecutionProvider'],
    )
    graph_prep_time_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    def _run_once() -> None:
        session.run(None, input_feed)

    for _ in range(task.warmup_iters):
        _run_once()

    samples_ns: List[int] = []
    for _ in range(task.measure_iters):
        start_ns = time.perf_counter_ns()
        _run_once()
        end_ns = time.perf_counter_ns()
        samples_ns.append(end_ns - start_ns)

    latency_ms = [sample / 1_000_000.0 for sample in samples_ns]
    mean_ms = statistics.mean(latency_ms)
    throughput = task.batch_size / (mean_ms / 1000.0) if mean_ms > 0 else 0.0
    hardware_manifest = _hardware_manifest(device, task.num_threads, task.precision, runtime_backend=task.runtime_backend)
    hardware_manifest['onnxruntime_version'] = _optional_package_version('onnxruntime')
    hardware_manifest['onnxruntime_providers'] = session.get_providers()

    return {
        'status': 'ok',
        'skip_reason': None,
        'run_dir': task.run_dir,
        'run_name': Path(task.run_dir).name,
        'device': task.device,
        'runtime_backend': task.runtime_backend,
        'execution_mode': task.execution_mode,
        'requested_precision_profile': task.precision,
        'effective_execution_dtype': 'float32',
        'graph_prep_time_ms': float(graph_prep_time_ms),
        'batch_size': task.batch_size,
        'num_threads': task.num_threads,
        'warmup_iters': task.warmup_iters,
        'measure_iters': task.measure_iters,
        'p50_latency_ms': float(np.percentile(latency_ms, 50)),
        'p90_latency_ms': float(np.percentile(latency_ms, 90)),
        'p95_latency_ms': float(np.percentile(latency_ms, 95)),
        'p99_latency_ms': float(np.percentile(latency_ms, 99)),
        'mean_latency_ms': float(mean_ms),
        'std_latency_ms': float(statistics.pstdev(latency_ms) if len(latency_ms) > 1 else 0.0),
        'min_latency_ms': float(min(latency_ms)),
        'max_latency_ms': float(max(latency_ms)),
        'throughput_samples_per_sec': float(throughput),
        'latency_samples_ms': latency_ms,
        'hardware_manifest': hardware_manifest,
        'model_spec': artifacts.model_spec,
        'metadata': artifacts.metadata,
        'component_specs': artifacts.component_specs,
    }


def _measure_openvino_model(task: LatencyTask, device: torch.device) -> Dict[str, Any]:
    skip_reason = _validate_precision_support(device, task.precision, runtime_backend=task.runtime_backend)
    if skip_reason is not None:
        return _build_skipped_result(task, device, skip_reason)

    import importlib

    run_dir = Path(task.run_dir)
    artifacts = load_run_artifacts(run_dir, device='cpu')
    export_manifest = _ensure_openvino_export(run_dir)
    dummy_input = build_dummy_input(
        artifacts.model_spec,
        batch_size=task.batch_size,
        component_specs=artifacts.component_specs,
    ).cpu()
    input_feed = {export_manifest['input_names'][0]: dummy_input.detach().numpy()}

    ov_module = importlib.import_module('openvino')
    core_factory = getattr(ov_module, 'Core', None)
    if core_factory is None:
        runtime_module = importlib.import_module('openvino.runtime')
        core_factory = getattr(runtime_module, 'Core')
    core = core_factory()
    start_ns = time.perf_counter_ns()
    ov_model = core.read_model(model=export_manifest['onnx_path'])
    compiled_model = core.compile_model(ov_model, 'CPU')
    graph_prep_time_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    def _run_once() -> None:
        compiled_model(input_feed)

    for _ in range(task.warmup_iters):
        _run_once()

    samples_ns: List[int] = []
    for _ in range(task.measure_iters):
        start_ns = time.perf_counter_ns()
        _run_once()
        end_ns = time.perf_counter_ns()
        samples_ns.append(end_ns - start_ns)

    latency_ms = [sample / 1_000_000.0 for sample in samples_ns]
    mean_ms = statistics.mean(latency_ms)
    throughput = task.batch_size / (mean_ms / 1000.0) if mean_ms > 0 else 0.0
    hardware_manifest = _hardware_manifest(device, task.num_threads, task.precision, runtime_backend=task.runtime_backend)
    hardware_manifest['openvino_version'] = _optional_package_version('openvino')
    hardware_manifest['openvino_device'] = 'CPU'

    return {
        'status': 'ok',
        'skip_reason': None,
        'run_dir': task.run_dir,
        'run_name': Path(task.run_dir).name,
        'device': task.device,
        'runtime_backend': task.runtime_backend,
        'execution_mode': task.execution_mode,
        'requested_precision_profile': task.precision,
        'effective_execution_dtype': 'float32',
        'graph_prep_time_ms': float(graph_prep_time_ms),
        'batch_size': task.batch_size,
        'num_threads': task.num_threads,
        'warmup_iters': task.warmup_iters,
        'measure_iters': task.measure_iters,
        'p50_latency_ms': float(np.percentile(latency_ms, 50)),
        'p90_latency_ms': float(np.percentile(latency_ms, 90)),
        'p95_latency_ms': float(np.percentile(latency_ms, 95)),
        'p99_latency_ms': float(np.percentile(latency_ms, 99)),
        'mean_latency_ms': float(mean_ms),
        'std_latency_ms': float(statistics.pstdev(latency_ms) if len(latency_ms) > 1 else 0.0),
        'min_latency_ms': float(min(latency_ms)),
        'max_latency_ms': float(max(latency_ms)),
        'throughput_samples_per_sec': float(throughput),
        'latency_samples_ms': latency_ms,
        'hardware_manifest': hardware_manifest,
        'model_spec': artifacts.model_spec,
        'metadata': artifacts.metadata,
        'component_specs': artifacts.component_specs,
    }


def _measure_model(task: LatencyTask) -> Dict[str, Any]:
    device = torch.device(task.device)
    backend_skip_reason = _validate_runtime_backend_support(device, task.runtime_backend)
    if backend_skip_reason is not None:
        return _build_skipped_result(task, device, backend_skip_reason)
    if task.runtime_backend == 'onnxruntime':
        return _measure_onnxruntime_model(task, device)
    if task.runtime_backend == 'openvino':
        return _measure_openvino_model(task, device)
    return _measure_pytorch_model(task, device)


def _worker_entry(task: LatencyTask, queue):
    try:
        queue.put({'ok': True, 'result': _measure_model(task)})
    except Exception as error:
        queue.put({
            'ok': False,
            'error': str(error),
            'task': {
                'run_dir': task.run_dir,
                'device': task.device,
                'runtime_backend': task.runtime_backend,
                'execution_mode': task.execution_mode,
                'precision': task.precision,
                'batch_size': task.batch_size,
                'num_threads': task.num_threads,
            },
        })


def execute_latency_task(task: LatencyTask) -> Dict[str, Any]:
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    process = ctx.Process(target=_worker_entry, args=(task, queue))
    process.start()
    payload = queue.get()
    process.join()
    if process.exitcode != 0 and payload.get('ok') is False:
        raise RuntimeError(f"Latency worker failed for {task.run_dir}: {payload['error']}")
    if payload.get('ok') is False:
        raise RuntimeError(f"Latency worker failed for {task.run_dir}: {payload['error']}")
    return payload['result']


def _save_latency_samples(results: List[Dict[str, Any]], output_dir: Path) -> Path:
    samples_payload = {
        f"{item['run_name']}__{item.get('runtime_backend', 'pytorch')}__{item['device']}__{item['execution_mode']}__{item['requested_precision_profile']}__bs{item['batch_size']}__t{item['num_threads']}": np.asarray(item.get('latency_samples_ms', []), dtype=np.float64)
        for item in results
        if item.get('status') == 'ok'
    }
    output_path = output_dir / 'latency_samples.npz'
    np.savez(output_path, **samples_payload)
    return output_path


def _group_ok_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in results if item.get('status') == 'ok']


def _load_complexity_reference(run_dir: Path) -> Dict[str, Any] | None:
    complexity_path = run_dir / 'model_complexity.json'
    if not complexity_path.exists():
        return None
    with open(complexity_path, 'r', encoding='utf-8') as input_file:
        complexity = json.load(input_file)
    return {
        'json_path': str(complexity_path),
        'markdown_path': str(run_dir / 'MODEL_COMPLEXITY.md'),
        'summary': complexity.get('summary', {}),
    }


def _build_run_references(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    references: Dict[str, Any] = {}
    for item in results:
        run_name = item['run_name']
        if run_name in references:
            continue
        run_dir = Path(item['run_dir'])
        references[run_name] = {
            'run_dir': str(run_dir),
            'model_flow_markdown_path': str(run_dir / 'MODEL_FLOW.md') if (run_dir / 'MODEL_FLOW.md').exists() else None,
            'model_complexity': _load_complexity_reference(run_dir),
        }
    return references


def resolve_latency_results_json(input_path) -> Path:
    resolved = resolve_existing_path(input_path)
    if isinstance(resolved, tuple):
        _, candidates = resolved
        candidate_text = '\n'.join(str(path) for path in candidates)
        raise FileNotFoundError('Latency results input not found. Checked:\n' + candidate_text)

    resolved = Path(resolved)
    if resolved.is_file():
        if resolved.name != 'latency_results.json':
            raise ValueError('Latency results input file must be latency_results.json')
        return resolved

    json_path = resolved / 'latency_results.json'
    if not json_path.exists():
        raise FileNotFoundError('Expected latency_results.json inside the provided latency directory')
    return json_path


def _thread_group(num_threads: Any, physical_cpu_count: Any, logical_cpu_count: Any) -> Optional[str]:
    if num_threads is None:
        return None
    num_threads = int(num_threads)
    physical = int(physical_cpu_count) if physical_cpu_count not in (None, '') else None
    logical = int(logical_cpu_count) if logical_cpu_count not in (None, '') else None
    if num_threads == 1:
        return 'single-thread'
    if physical is not None:
        if num_threads < physical:
            return 'sub-physical'
        if num_threads == physical:
            return 'all-physical'
    if logical is not None and num_threads == logical:
        return 'all-logical'
    if physical is not None and num_threads > physical:
        return 'beyond-physical'
    return 'scaled'


def _safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    if numerator in (None, '') or denominator in (None, '', 0):
        return None
    denominator = float(denominator)
    if denominator == 0.0:
        return None
    return float(numerator) / denominator


def _flatten_result_rows(results_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    run_references = results_payload.get('run_references', {})
    flattened_rows: List[Dict[str, Any]] = []
    for item in results_payload['results']:
        reference = run_references.get(item['run_name'], {})
        complexity_summary = ((reference.get('model_complexity') or {}).get('summary') or {})
        metadata = item.get('metadata') or {}
        model_spec = item.get('model_spec') or {}
        hardware_manifest = item.get('hardware_manifest') or {}
        batch_size = item.get('batch_size')
        num_threads = item.get('num_threads')
        p50_latency_ms = item.get('p50_latency_ms')
        throughput = item.get('throughput_samples_per_sec')
        logical_cpu_count = hardware_manifest.get('logical_cpu_count')
        physical_cpu_count = hardware_manifest.get('physical_cpu_count')
        flattened_rows.append({
            'benchmark_id': results_payload.get('benchmark_id'),
            'timestamp': results_payload.get('timestamp'),
            'device': results_payload.get('device'),
            'runtime_backend': item.get('runtime_backend', 'pytorch'),
            'run_name': item.get('run_name'),
            'run_dir': item.get('run_dir'),
            'experiment_name': metadata.get('experiment_name'),
            'model_label': metadata.get('model_label'),
            'training_label': metadata.get('training_label'),
            'model_type': model_spec.get('model_type'),
            'seq_len': model_spec.get('seq_len'),
            'num_ports': model_spec.get('num_ports'),
            'execution_mode': item.get('execution_mode'),
            'requested_precision_profile': item.get('requested_precision_profile'),
            'effective_execution_dtype': item.get('effective_execution_dtype'),
            'batch_size': batch_size,
            'num_threads': num_threads,
            'status': item.get('status'),
            'skip_reason': item.get('skip_reason'),
            'graph_prep_time_ms': item.get('graph_prep_time_ms'),
            'warmup_iters': item.get('warmup_iters'),
            'measure_iters': item.get('measure_iters'),
            'mean_latency_ms': item.get('mean_latency_ms'),
            'std_latency_ms': item.get('std_latency_ms'),
            'min_latency_ms': item.get('min_latency_ms'),
            'p50_latency_ms': item.get('p50_latency_ms'),
            'p90_latency_ms': item.get('p90_latency_ms'),
            'p95_latency_ms': item.get('p95_latency_ms'),
            'p99_latency_ms': item.get('p99_latency_ms'),
            'max_latency_ms': item.get('max_latency_ms'),
            'throughput_samples_per_sec': throughput,
            'samples_per_ms': None if throughput in (None, '') else float(throughput) / 1000.0,
            'p50_latency_us': None if p50_latency_ms in (None, '') else float(p50_latency_ms) * 1000.0,
            'latency_per_sample_us': None if p50_latency_ms in (None, '') or batch_size in (None, '', 0) else (float(p50_latency_ms) * 1000.0) / float(batch_size),
            'throughput_per_thread': None if throughput in (None, '') or num_threads in (None, '', 0) else float(throughput) / float(num_threads),
            'trainable_parameters': complexity_summary.get('trainable_parameters'),
            'macs_per_sample': complexity_summary.get('macs_per_sample'),
            'flops_per_sample_estimate': complexity_summary.get('flops_per_sample_estimate'),
            'thread_group': _thread_group(num_threads, physical_cpu_count, logical_cpu_count),
            'threads_per_physical_core_ratio': _safe_ratio(num_threads, physical_cpu_count),
            'threads_per_logical_cpu_ratio': _safe_ratio(num_threads, logical_cpu_count),
            'cpu_model_name': hardware_manifest.get('cpu_model_name'),
            'cpu_capability': hardware_manifest.get('cpu_capability'),
            'cpu_flag_summary': ','.join(hardware_manifest.get('cpu_flag_summary') or []),
            'mkldnn_available': hardware_manifest.get('mkldnn_available'),
            'mkldnn_enabled': hardware_manifest.get('mkldnn_enabled'),
            'onednn_version': hardware_manifest.get('onednn_version'),
            'onnxruntime_version': hardware_manifest.get('onnxruntime_version'),
            'onnxruntime_providers': ','.join(hardware_manifest.get('onnxruntime_providers') or []),
            'openvino_version': hardware_manifest.get('openvino_version'),
            'openvino_device': hardware_manifest.get('openvino_device'),
            'logical_cpu_count': hardware_manifest.get('logical_cpu_count'),
            'physical_cpu_count': hardware_manifest.get('physical_cpu_count'),
            'hostname': hardware_manifest.get('hostname'),
            'python_version': hardware_manifest.get('python_version'),
            'pytorch_version': hardware_manifest.get('pytorch_version'),
        })
    return flattened_rows


def _save_latency_csv(results_payload: Dict[str, Any], output_dir: Path) -> Path:
    rows = _flatten_result_rows(results_payload)
    output_path = output_dir / 'latency_results.csv'
    if not rows:
        with open(output_path, 'w', encoding='utf-8', newline='') as output_file:
            output_file.write('')
        return output_path

    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', encoding='utf-8', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def export_latency_csv_from_results(input_path, output_path=None) -> Path:
    json_path = resolve_latency_results_json(input_path)
    with open(json_path, 'r', encoding='utf-8') as input_file:
        payload = json.load(input_file)
    output_dir = json_path.parent if output_path is None else Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _save_latency_csv(payload, output_dir)
    if output_path is not None:
        requested_output = Path(output_path)
        if requested_output != csv_path:
            requested_output.write_text(csv_path.read_text(encoding='utf-8'), encoding='utf-8')
            return requested_output
    return csv_path


def backfill_latency_csv_tree(root_path) -> List[Path]:
    resolved = resolve_existing_path(root_path)
    if isinstance(resolved, tuple):
        _, candidates = resolved
        candidate_text = '\n'.join(str(path) for path in candidates)
        raise FileNotFoundError('Backfill root not found. Checked:\n' + candidate_text)
    resolved = Path(resolved)
    json_paths = sorted(resolved.rglob('latency_results.json')) if resolved.is_dir() else [resolve_latency_results_json(resolved)]
    generated: List[Path] = []
    for json_path in json_paths:
        generated.append(export_latency_csv_from_results(json_path))
    return generated


def _render_hardware_summary(results_payload: Dict[str, Any]) -> List[str]:
    manifests = [item.get('hardware_manifest') for item in results_payload['results'] if item.get('hardware_manifest')]
    if not manifests:
        return []
    manifest = manifests[0]
    lines = [
        '## Hardware Summary',
        '',
        f"- Runtime backend: `{manifest.get('runtime_backend', 'pytorch')}`",
        f"- Hostname: `{manifest.get('hostname', '-')}`",
        f"- CPU model: `{manifest.get('cpu_model_name', '-')}`",
        f"- CPU capability: `{manifest.get('cpu_capability', '-')}`",
        f"- CPU flag summary: `{manifest.get('cpu_flag_summary', [])}`",
        f"- Logical CPU count: `{manifest.get('logical_cpu_count', '-')}`",
        f"- Physical CPU count: `{manifest.get('physical_cpu_count', '-')}`",
        f"- mkldnn available: `{manifest.get('mkldnn_available', '-')}`",
        f"- mkldnn enabled: `{manifest.get('mkldnn_enabled', '-')}`",
        f"- oneDNN version: `{manifest.get('onednn_version', '-')}`",
        f"- torch.compile available: `{manifest.get('torch_compile_available', '-')}`",
        f"- Python: `{manifest.get('python_version', '-')}`",
        f"- PyTorch: `{manifest.get('pytorch_version', '-')}`",
    ]
    if manifest.get('onnxruntime_version'):
        lines.append(f"- ONNX Runtime version: `{manifest.get('onnxruntime_version', '-')}`")
    if manifest.get('onnxruntime_providers'):
        lines.append(f"- ONNX Runtime providers: `{manifest.get('onnxruntime_providers', [])}`")
    if manifest.get('openvino_version'):
        lines.append(f"- OpenVINO version: `{manifest.get('openvino_version', '-')}`")
    if manifest.get('openvino_device'):
        lines.append(f"- OpenVINO device: `{manifest.get('openvino_device', '-')}`")
    env = manifest.get('env') or {}
    if env:
        lines.append(f"- Relevant env: `{env}`")
    lines.append('')
    return lines


def _build_cpu_thread_scaling_summary(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    ok_results = [item for item in _group_ok_results(results) if item.get('device', '').startswith('cpu')]
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
    for item in ok_results:
        grouped.setdefault((item['run_name'], item.get('runtime_backend', 'pytorch'), item['execution_mode']), []).append(item)

    for (run_name, runtime_backend, execution_mode), items in grouped.items():
        best_throughput = max(items, key=lambda entry: entry['throughput_samples_per_sec'])
        bs1_candidates = [entry for entry in items if entry['batch_size'] == 1]
        min_bs1_latency = min(bs1_candidates, key=lambda entry: entry['p50_latency_ms']) if bs1_candidates else None
        summaries[f'{run_name}::{runtime_backend}::{execution_mode}'] = {
            'runtime_backend': runtime_backend,
            'execution_mode': execution_mode,
            'best_throughput': {
                'thread_count': int(best_throughput['num_threads']),
                'batch_size': int(best_throughput['batch_size']),
                'precision': best_throughput['requested_precision_profile'],
                'throughput_samples_per_sec': float(best_throughput['throughput_samples_per_sec']),
                'p50_latency_ms': float(best_throughput['p50_latency_ms']),
            },
            'min_bs1_latency': None if min_bs1_latency is None else {
                'thread_count': int(min_bs1_latency['num_threads']),
                'precision': min_bs1_latency['requested_precision_profile'],
                'p50_latency_ms': float(min_bs1_latency['p50_latency_ms']),
                'throughput_samples_per_sec': float(min_bs1_latency['throughput_samples_per_sec']),
            },
        }
    return summaries


def _plot_run_latency(results: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    ok_results = _group_ok_results(results)
    if not ok_results:
        return generated

    grouped: Dict[tuple[str, str, str, int], List[Dict[str, Any]]] = {}
    for item in ok_results:
        grouped.setdefault((item.get('runtime_backend', 'pytorch'), item['execution_mode'], item['requested_precision_profile'], item['num_threads']), []).append(item)

    width, height, _, _ = outside_legend_figure_size(len(grouped), base_width=12.0, base_height=7.0)
    fig, axis = plt.subplots(figsize=(width, height))
    for series_index, ((runtime_backend, execution_mode, precision, num_threads), items) in enumerate(sorted(grouped.items())):
        items = sorted(items, key=lambda entry: entry['batch_size'])
        plot_style = style_for_series(series_index)
        axis.plot(
            [entry['batch_size'] for entry in items],
            [entry['p50_latency_ms'] for entry in items],
            label=f'{runtime_backend} / {execution_mode} / {precision} / t={num_threads}',
            **plot_style,
        )
    axis.set_xlabel('Batch Size')
    axis.set_ylabel('P50 Latency (ms)')
    axis.set_title('P50 Latency vs Batch Size')
    axis.grid(True, alpha=0.3)
    place_legend_outside_right(fig, axis)
    latency_plot = output_dir / 'p50_latency_vs_batch.jpg'
    fig.savefig(latency_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(latency_plot)

    fig, axis = plt.subplots(figsize=(width, height))
    for series_index, ((runtime_backend, execution_mode, precision, num_threads), items) in enumerate(sorted(grouped.items())):
        items = sorted(items, key=lambda entry: entry['batch_size'])
        plot_style = style_for_series(series_index)
        axis.plot(
            [entry['batch_size'] for entry in items],
            [entry['throughput_samples_per_sec'] for entry in items],
            label=f'{runtime_backend} / {execution_mode} / {precision} / t={num_threads}',
            **plot_style,
        )
    axis.set_xlabel('Batch Size')
    axis.set_ylabel('Throughput (samples/s)')
    axis.set_title('Throughput vs Batch Size')
    axis.grid(True, alpha=0.3)
    place_legend_outside_right(fig, axis)
    throughput_plot = output_dir / 'throughput_vs_batch.jpg'
    fig.savefig(throughput_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    generated.append(throughput_plot)

    thread_candidates = {(item.get('runtime_backend', 'pytorch'), item['execution_mode'], item['requested_precision_profile'], item['batch_size']) for item in ok_results if item['num_threads'] != 1}
    if thread_candidates:
        width, height, _, _ = outside_legend_figure_size(len(thread_candidates), base_width=12.0, base_height=7.0)
        fig, axis = plt.subplots(figsize=(width, height))
        for series_index, (runtime_backend, execution_mode, precision, batch_size) in enumerate(sorted(thread_candidates)):
            items = sorted(
                [entry for entry in ok_results if entry.get('runtime_backend', 'pytorch') == runtime_backend and entry['execution_mode'] == execution_mode and entry['requested_precision_profile'] == precision and entry['batch_size'] == batch_size],
                key=lambda entry: entry['num_threads'],
            )
            plot_style = style_for_series(series_index)
            axis.plot(
                [entry['num_threads'] for entry in items],
                [entry['p50_latency_ms'] for entry in items],
                label=f'{runtime_backend} / {execution_mode} / {precision} / bs={batch_size}',
                **plot_style,
            )
        axis.set_xlabel('Threads')
        axis.set_ylabel('P50 Latency (ms)')
        axis.set_title('P50 Latency vs Threads')
        axis.grid(True, alpha=0.3)
        place_legend_outside_right(fig, axis)
        threads_plot = output_dir / 'p50_latency_vs_threads.jpg'
        fig.savefig(threads_plot, dpi=150, bbox_inches='tight')
        plt.close(fig)
        generated.append(threads_plot)
    return generated


def _plot_aggregate_latency(results: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files: List[Path] = []
    ok_results = _group_ok_results(results)
    if not ok_results:
        return plot_files

    bs1_results = [item for item in ok_results if item['batch_size'] == 1]
    if bs1_results:
        width, height, _, _ = outside_legend_figure_size(len(bs1_results), base_width=12.0, base_height=7.0)
        fig, axis = plt.subplots(figsize=(width, height))
        labels = [f"{item['run_name']}\n{item['execution_mode']}\n{item['requested_precision_profile']}\nt={item['num_threads']}" for item in bs1_results]
        values = [item['p50_latency_ms'] for item in bs1_results]
        axis.bar(labels, values)
        axis.set_ylabel('P50 Latency (ms)')
        axis.set_title('Batch-1 P50 Latency Comparison')
        axis.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        output_path = output_dir / 'bs1_p50_comparison.jpg'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_files.append(output_path)

    fixed_thread_candidates = sorted({item['num_threads'] for item in ok_results})
    for num_threads in fixed_thread_candidates:
        thread_results = [item for item in ok_results if item['num_threads'] == num_threads]
        if len({item['run_name'] for item in thread_results}) < 2:
            continue

        grouped: Dict[tuple[str, str, str, str], List[Dict[str, Any]]] = {}
        for item in thread_results:
            grouped.setdefault((item['run_name'], item.get('runtime_backend', 'pytorch'), item['execution_mode'], item['requested_precision_profile']), []).append(item)

        width, height, _, _ = outside_legend_figure_size(len(grouped), base_width=12.0, base_height=7.0)
        fig, axis = plt.subplots(figsize=(width, height))
        for series_index, ((run_name, runtime_backend, execution_mode, precision), items) in enumerate(sorted(grouped.items())):
            items = sorted(items, key=lambda entry: entry['batch_size'])
            plot_style = style_for_series(series_index)
            axis.plot(
                [entry['batch_size'] for entry in items],
                [entry['p50_latency_ms'] for entry in items],
                label=f'{run_name} / {runtime_backend} / {execution_mode} / {precision}',
                **plot_style,
            )
        axis.set_xlabel('Batch Size')
        axis.set_ylabel('P50 Latency (ms)')
        axis.set_title(f'Cross-Run P50 Latency vs Batch Size (threads={num_threads})')
        axis.grid(True, alpha=0.3)
        place_legend_outside_right(fig, axis)
        latency_plot = output_dir / f'p50_latency_vs_batch_threads_{num_threads}.jpg'
        fig.savefig(latency_plot, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_files.append(latency_plot)

        fig, axis = plt.subplots(figsize=(width, height))
        for series_index, ((run_name, runtime_backend, execution_mode, precision), items) in enumerate(sorted(grouped.items())):
            items = sorted(items, key=lambda entry: entry['batch_size'])
            plot_style = style_for_series(series_index)
            axis.plot(
                [entry['batch_size'] for entry in items],
                [entry['throughput_samples_per_sec'] for entry in items],
                label=f'{run_name} / {runtime_backend} / {execution_mode} / {precision}',
                **plot_style,
            )
        axis.set_xlabel('Batch Size')
        axis.set_ylabel('Throughput (samples/s)')
        axis.set_title(f'Cross-Run Throughput vs Batch Size (threads={num_threads})')
        axis.grid(True, alpha=0.3)
        place_legend_outside_right(fig, axis)
        throughput_plot = output_dir / f'throughput_vs_batch_threads_{num_threads}.jpg'
        fig.savefig(throughput_plot, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_files.append(throughput_plot)

    return plot_files


def render_latency_report(results_payload: Dict[str, Any]) -> str:
    lines = [
        '# Latency Report',
        '',
        f"- Device: `{results_payload['device']}`",
        f"- Runtime backends: `{results_payload.get('runtime_backends', ['pytorch'])}`",
        f"- Execution modes: `{results_payload['execution_modes']}`",
        f"- Precision profiles: `{results_payload['precision_profiles']}`",
        f"- Batch sizes: `{results_payload['batch_sizes']}`",
        f"- Thread counts: `{results_payload['thread_counts']}`",
        '',
    ]
    lines.extend(_render_hardware_summary(results_payload))
    cpu_thread_summaries = results_payload.get('cpu_thread_scaling_summaries', {})
    if cpu_thread_summaries:
        lines.extend([
            '## CPU Thread Scaling Highlights',
            '',
        ])
        for run_key, summary in cpu_thread_summaries.items():
            best = summary['best_throughput']
            lines.append(f"### {run_key}")
            lines.append('')
            lines.append(f"- Runtime backend: `{summary.get('runtime_backend', 'pytorch')}`")
            lines.append(f"- Execution mode: `{summary['execution_mode']}`")
            lines.append(
                f"- Best throughput config: threads=`{best['thread_count']}`, batch=`{best['batch_size']}`, precision=`{best['precision']}`, throughput=`{best['throughput_samples_per_sec']:.3f}` samples/s, p50=`{best['p50_latency_ms']:.3f}` ms"
            )
            min_bs1 = summary.get('min_bs1_latency')
            if min_bs1 is not None:
                lines.append(
                    f"- Lowest batch-1 p50 latency: threads=`{min_bs1['thread_count']}`, precision=`{min_bs1['precision']}`, p50=`{min_bs1['p50_latency_ms']:.3f}` ms, throughput=`{min_bs1['throughput_samples_per_sec']:.3f}` samples/s"
                )
            lines.append('')

    lines.extend([
        '## Run References',
        '',
    ])
    run_references = results_payload.get('run_references', {})
    for run_name, reference in run_references.items():
        lines.append(f"### {run_name}")
        lines.append('')
        lines.append(f"- Run dir: `{reference['run_dir']}`")
        if reference.get('model_flow_markdown_path'):
            lines.append(f"- Model flow: `{reference['model_flow_markdown_path']}`")
        complexity = reference.get('model_complexity')
        if complexity and complexity.get('summary'):
            summary = complexity['summary']
            lines.append(f"- Model complexity JSON: `{complexity['json_path']}`")
            if complexity.get('markdown_path'):
                lines.append(f"- Model complexity Markdown: `{complexity['markdown_path']}`")
            lines.append(f"- Trainable parameters: `{summary.get('trainable_parameters_string', '-')}`")
            lines.append(f"- MACs / sample: `{summary.get('macs_per_sample_string', '-')}`")
            lines.append(f"- FLOPs / sample estimate: `{summary.get('flops_per_sample_estimate_string', '-')}`")
        lines.append('')

    lines.extend([
        '## Results',
        '',
        '| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ])
    for item in results_payload['results']:
        lines.append(
            f"| `{item['run_name']}` | `{item.get('runtime_backend', 'pytorch')}` | `{item['execution_mode']}` | `{item['requested_precision_profile']}` | {item['batch_size']} | {item['num_threads']} | {item['status']} | "
            f"{item.get('graph_prep_time_ms', '-')} | {item.get('p50_latency_ms', '-')} | {item.get('p95_latency_ms', '-')} | {item.get('p99_latency_ms', '-')} | {item.get('throughput_samples_per_sec', '-')} | {item.get('skip_reason', '-') or '-'} |"
        )
    return '\n'.join(lines) + '\n'


def save_latency_results(results_payload: Dict[str, Any], output_dir: Path, aggregate: bool = False) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'latency_results.json'
    with open(json_path, 'w', encoding='utf-8') as output_file:
        json.dump(results_payload, output_file, indent=2, ensure_ascii=False)

    csv_path = _save_latency_csv(results_payload, output_dir)

    samples_path = _save_latency_samples(results_payload['results'], output_dir)
    report_path = output_dir / 'LATENCY_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as output_file:
        output_file.write(render_latency_report(results_payload))

    hardware_path = output_dir / 'hardware_manifest.json'
    manifests = [item['hardware_manifest'] for item in results_payload['results'] if item.get('hardware_manifest')]
    if manifests:
        with open(hardware_path, 'w', encoding='utf-8') as output_file:
            json.dump(manifests[0] if len(manifests) == 1 else manifests, output_file, indent=2, ensure_ascii=False)

    plots_dir = output_dir / 'plots'
    plot_files = _plot_aggregate_latency(results_payload['results'], plots_dir) if aggregate else _plot_run_latency(results_payload['results'], plots_dir)
    return {
        'json_path': str(json_path),
        'csv_path': str(csv_path),
        'samples_path': str(samples_path),
        'report_path': str(report_path),
        'hardware_manifest_path': str(hardware_path),
        'plot_files': [str(path) for path in plot_files],
    }


def benchmark_latency_programmatic(
    exp_dir=None,
    run_dir=None,
    run_dirs=None,
    runs=None,
    device='cpu',
    runtime_backends: Optional[str] = None,
    execution_modes: Optional[str] = None,
    precision_profiles: Optional[str] = None,
    batch_sizes: Optional[str] = None,
    batch_antennas: Optional[str] = None,
    batch_rbgs: Optional[str] = None,
    thread_counts: Optional[str] = None,
    warmup_iters: int = 20,
    measure_iters: int = 50,
    output_dir=None,
) -> Dict[str, Any]:
    exp_dir, run_dir, run_dirs, runs = normalize_latency_selection(
        exp_dir=exp_dir,
        run_dir=run_dir,
        run_dirs=run_dirs,
        runs=runs,
    )
    resolved_device = resolve_latency_device(device)
    benchmark_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    target_dirs = resolve_run_selection(exp_dir=exp_dir, run_dir=run_dir, run_dirs=run_dirs, runs=runs)
    resolved_batch_sizes = resolve_batch_sizes(batch_sizes, batch_antennas=batch_antennas, batch_rbgs=batch_rbgs)
    resolved_thread_counts = parse_csv_ints(thread_counts, default_thread_counts(resolved_device.type))
    resolved_runtime_backends = parse_runtime_backends(resolved_device.type, runtime_backends)
    resolved_execution_modes = parse_execution_modes(resolved_device.type, execution_modes)
    resolved_precisions = parse_precision_profiles(resolved_device.type, precision_profiles)

    aggregate_requested = exp_dir is not None or len(target_dirs) > 1 or output_dir is not None
    aggregate_output_dir = None
    if aggregate_requested:
        aggregate_output_dir = resolve_latency_output_dir(
            exp_dir=Path(exp_dir) if exp_dir else None,
            run_dirs=target_dirs,
            explicit_output=output_dir,
            device_type=resolved_device.type,
            benchmark_id=benchmark_id,
        )

    tasks = build_latency_task_matrix(
        run_dirs=target_dirs,
        device=resolved_device,
        runtime_backends=resolved_runtime_backends,
        execution_modes=resolved_execution_modes,
        precision_profiles=resolved_precisions,
        batch_sizes=resolved_batch_sizes,
        thread_counts=resolved_thread_counts,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    )
    print(f'Benchmark tasks: {len(tasks)} total')

    per_run_results: Dict[str, List[Dict[str, Any]]] = {run_dir.name: [] for run_dir in target_dirs}
    aggregate_results: List[Dict[str, Any]] = []
    for task_index, task in enumerate(tasks, start=1):
        print(
            f"[{task_index}/{len(tasks)}] Benchmarking run={Path(task.run_dir).name} "
            f"device={task.device} backend={task.runtime_backend} mode={task.execution_mode} precision={task.precision} batch={task.batch_size} threads={task.num_threads}"
        )
        result = execute_latency_task(task)
        if result.get('status') == 'ok':
            print(
                f"  -> done: prep={result['graph_prep_time_ms']:.3f} ms, p50={result['p50_latency_ms']:.3f} ms, "
                f"throughput={result['throughput_samples_per_sec']:.3f} samples/s"
            )
        else:
            print(f"  -> skipped: {result.get('skip_reason', 'unknown reason')}")
        per_run_results[Path(task.run_dir).name].append(result)
        aggregate_results.append(result)

    per_run_artifacts: Dict[str, Dict[str, Any]] = {}
    for target_dir in target_dirs:
        run_output_dir = target_dir / 'latency' / f"{benchmark_id}_{resolved_device.type}"
        payload = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_id': benchmark_id,
            'device': str(resolved_device),
            'runtime_backends': resolved_runtime_backends,
            'execution_modes': resolved_execution_modes,
            'precision_profiles': resolved_precisions,
            'batch_sizes': resolved_batch_sizes,
            'thread_counts': resolved_thread_counts if resolved_device.type == 'cpu' else [1],
            'run_name': target_dir.name,
            'results': per_run_results[target_dir.name],
            'run_references': _build_run_references(per_run_results[target_dir.name]),
            'cpu_thread_scaling_summaries': _build_cpu_thread_scaling_summary(per_run_results[target_dir.name]),
        }
        per_run_artifacts[target_dir.name] = save_latency_results(payload, run_output_dir, aggregate=False)

    aggregate_artifacts = None
    if aggregate_output_dir is not None:
        aggregate_payload = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_id': benchmark_id,
            'device': str(resolved_device),
            'runtime_backends': resolved_runtime_backends,
            'execution_modes': resolved_execution_modes,
            'precision_profiles': resolved_precisions,
            'batch_sizes': resolved_batch_sizes,
            'thread_counts': resolved_thread_counts if resolved_device.type == 'cpu' else [1],
            'run_names': [run_dir.name for run_dir in target_dirs],
            'results': aggregate_results,
            'run_references': _build_run_references(aggregate_results),
            'cpu_thread_scaling_summaries': _build_cpu_thread_scaling_summary(aggregate_results),
        }
        aggregate_artifacts = save_latency_results(aggregate_payload, aggregate_output_dir, aggregate=True)

    return {
        'device': str(resolved_device),
        'runtime_backends': resolved_runtime_backends,
        'execution_modes': resolved_execution_modes,
        'precision_profiles': resolved_precisions,
        'batch_sizes': resolved_batch_sizes,
        'thread_counts': resolved_thread_counts if resolved_device.type == 'cpu' else [1],
        'per_run_artifacts': per_run_artifacts,
        'aggregate_artifacts': aggregate_artifacts,
    }
