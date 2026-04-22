# CPU oneDNN Benchmark Tutorial

## Scope

This note explains the three CPU execution paths currently modeled in the latency benchmark:

1. PyTorch eager + oneDNN
2. TorchScript JIT static graph + oneDNN
3. `torch.compile` graph lowering + oneDNN plus non-oneDNN fused kernels when the compiler decides to generate them

The goal is to make the runtime stack explicit so that you can reason about what is actually being measured when benchmarking a trained `.pth` model on CPU.

## What The Benchmark Loads

The benchmark does **not** load a native oneDNN graph artifact.

It loads a normal training run and reconstructs a normal PyTorch model:

1. Read the run directory.
2. Load `model.pth` and `config.yaml`.
3. Recover `model_spec`, `training_spec`, `metadata`, and `component_specs`.
4. Recreate the PyTorch module from `model_spec`.
5. Load the checkpoint `state_dict` back into that module.
6. Build a dummy input matching the run's actual task/model I/O contract.

So the benchmark input is always a PyTorch module plus a PyTorch tensor input. There is no direct deserialization of oneDNN primitives or oneDNN graph IR.

## What “Using oneDNN” Means Here

In this project, “using oneDNN on CPU” means:

- Python code calls PyTorch APIs such as `model(x)`.
- PyTorch's CPU backend decides whether a given operator can use an mkldnn/oneDNN kernel.
- If supported, PyTorch dispatches that operator to the underlying oneDNN implementation.

So the benchmark is measuring **PyTorch CPU inference with oneDNN-backed operator dispatch**, not a handwritten oneDNN C++ application.

This distinction matters:

- You are **not** manually creating oneDNN `engine`, `stream`, `memory`, and `primitive` objects.
- You are **not** serializing a oneDNN graph or executing a standalone oneDNN binary.
- You **are** measuring the path most people would use first for a Python CPU deployment workflow.

## Path 1: PyTorch Eager + oneDNN

### High-level flow

1. Python calls `model(dummy_input)`.
2. PyTorch executes the `forward()` method eagerly.
3. Each operator is dispatched one by one.
4. For supported CPU ops, PyTorch uses mkldnn/oneDNN kernels internally.
5. Unsupported or unfused parts still run through normal PyTorch CPU kernels.

### What is being “interpreted”

The Python `forward()` function is still the top-level execution model.

That does **not** mean the math itself is interpreted Python. The Python layer orchestrates execution, but the heavy tensor ops run inside compiled C++ kernels. When mkldnn is enabled and the op is supported, those kernels are backed by oneDNN.

So, more precisely:

- Python control flow is eager and dynamic.
- Tensor ops run as compiled backend kernels.
- oneDNN is used at the operator-kernel dispatch layer, not as a Python interpreter.

### What the benchmark measures here

The benchmark loads the model, keeps it as a normal PyTorch eager module, and times repeated `forward()` calls under `torch.inference_mode()`.

This is the simplest and most faithful baseline for “PyTorch CPU deployment with mkldnn enabled”.

## Path 2: TorchScript JIT Static Graph + oneDNN

### High-level flow

1. Start from the same restored PyTorch module.
2. Build a representative dummy input.
3. Run `torch.jit.trace(model, dummy_input)`.
4. Run `torch.jit.optimize_for_inference(...)`.
5. Optionally freeze the graph with `torch.jit.freeze(...)` when possible.
6. Execute the resulting TorchScript module repeatedly.

### What JIT means here

In this benchmark, the “JIT mode” is TorchScript tracing, not Python bytecode JIT execution.

The important effect is that the model is converted from a dynamic Python module into a more static graph representation. That graph is then executed by TorchScript runtime rather than by re-entering normal Python eager execution for each forward pass.

### What is being executed at runtime

The runtime is not interpreting Python source anymore for each call. Instead, it runs the traced TorchScript graph.

However, the graph nodes still lower to backend kernels. For supported CPU operators, those backend kernels can still be oneDNN kernels.

So the runtime layering becomes:

- TorchScript graph runtime at the top
- operator dispatch inside PyTorch runtime beneath it
- oneDNN kernels for the supported CPU ops beneath that

### Why this can help

JIT can reduce some Python overhead and make the execution graph more static, which may help scheduling and graph-level optimization. It can also make benchmarking cleaner if your workload is shape-stable.

### Limitations

- `trace` only captures the path exercised by the example input.
- Highly dynamic Python control flow is not faithfully preserved by tracing.
- Some improvements may come from reduced Python overhead rather than from oneDNN itself.

## Path 3: `torch.compile` Graph Lowering + oneDNN + extra fused kernels

### High-level flow

1. Start from the same restored PyTorch module.
2. Call `torch.compile(model, mode='reduce-overhead')`.
3. The first executions trigger graph capture and compilation.
4. PyTorch's compile stack lowers captured graph regions to compiled code.
5. At runtime, the compiled regions execute instead of the original eager Python path.

### What `torch.compile` is actually doing

`torch.compile` is not the same thing as TorchScript JIT.

In current PyTorch, this path typically involves components such as:

- TorchDynamo: captures Python-level frame execution into graph regions
- AOTAutograd machinery: more relevant for training, but part of the broader compile stack
- Inductor backend: lowers graphs to generated kernels / code for execution

On CPU, compiled execution may:

- still call into oneDNN-backed kernels for some ops
- generate fused kernels for elementwise and other subgraphs
- reduce framework overhead and improve memory traffic patterns

So this path is the most likely one to include improvements that are **not purely oneDNN kernel substitutions**.

### What is being executed at runtime

This is no longer ordinary eager Python dispatch on every forward pass.

Instead, compiled graph regions are executed through generated code and backend runtime pieces. Some sub-ops may still be served by oneDNN. Other subgraphs may become compiler-generated fused kernels that are not themselves “oneDNN primitives” in the direct sense.

That is why this path is best understood as:

- compiled PyTorch graph execution on CPU
- oneDNN where useful and selected by lower layers
- plus possible extra compiler-generated fusion beyond oneDNN operator granularity

### Why this path is different from the first two

Compared with eager and TorchScript, `torch.compile` can change the execution boundary itself. Instead of just dispatching op-by-op, it can compile larger graph regions, which means:

- fewer framework transitions
- potentially better fusion
- potentially fewer temporary tensors
- latency changes that cannot be attributed only to oneDNN kernel choice

## Does The Benchmark Measure Compilation Time

The benchmark separates **graph preparation time** from **steady-state latency**:

- eager: graph prep time is `0`
- JIT: graph prep time is the trace + optimize + freeze preparation time
- compile: graph prep time is the `torch.compile(...)` preparation time before the repeated benchmark loop

The repeated latency measurements are intended to reflect steady-state execution after the mode has been prepared.

## Is Anything Converted To A Standalone Binary

Not in the current workflow.

The benchmark does **not**:

- export to a oneDNN binary format
- produce a standalone CPU executable
- emit a persistent native artifact for oneDNN runtime

What you get is a runtime-transformed PyTorch execution object inside the current Python process.

## Python PyTorch API vs oneDNN C++ API

They are not the same interface.

### Python PyTorch API

You write things like:

- `model(x)`
- `torch.inference_mode()`
- `torch.autocast(...)`
- `torch.jit.trace(...)`
- `torch.compile(...)`

### oneDNN C++ API

You write things like:

- engine
- stream
- memory descriptors
- primitives
- reorders
- explicit graph / primitive execution setup

So Python PyTorch and oneDNN C++ are not symmetric APIs. PyTorch is the high-level framework API. oneDNN is the low-level CPU backend library.

## Why Compare All Three

Because the latency difference can come from different layers:

- eager vs JIT: often reveals Python/runtime graph overhead effects
- JIT vs compile: often reveals deeper graph-lowering and fusion effects
- within each path, precision and thread count reveal backend behavior and CPU scaling effects

If two models have similar FLOPs but very different latency under `compile`, that can indicate graph-structure and fusion effects rather than just raw arithmetic complexity.

## Practical Interpretation

When you look at benchmark results, interpret them like this:

- eager: baseline PyTorch CPU deployment with oneDNN dispatch where available
- JIT: static-graph PyTorch runtime with oneDNN-backed ops where available
- compile: compiled graph execution that may mix oneDNN usage with additional compiler-generated fusion

These are not three identical “oneDNN modes”. They are three progressively more transformed execution stacks with different amounts of graph stabilization and compiler involvement.

## Current Benchmark Implementation In This Repo

The benchmark now supports an `execution_mode` dimension in addition to device, precision, batch size, and CPU thread count.

Current CPU defaults:

- execution modes: `eager,jit,compile`
- precision profiles: `fp32,bf16`
- batch sizes: `1,2,4,8,16,32,64,128`
- thread counts: `1,2,4,8,all-physical`

Current CUDA default remains conservative and does not yet expand into these three execution modes by default.

## Recommended Study Order

1. Benchmark one run in `eager` only and understand the baseline.
2. Add `jit` and compare whether batch-1 latency changes mostly from reduced runtime overhead.
3. Add `compile` and compare whether higher-batch throughput improves more strongly.
4. Then repeat the comparison across different thread counts.

That order makes it easier to separate backend-kernel effects from graph-lowering effects.