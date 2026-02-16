"""
============================================================================
 02_compare_tanh.py  –  Custom tanh vs PyTorch Built-in tanh (cuDNN)
============================================================================

 WHAT THIS PROGRAM DOES
 ----------------------
 Compares two ways to compute tanh on a GPU tensor using PyTorch:

   1. Custom tanh    – manual formula:  (exp(2x) - 1) / (exp(2x) + 1)
   2. Built-in tanh  – torch.tanh(),  which calls cuDNN under the hood

 WHY COMPARE THESE?
 -------------------
 When you call torch.tanh(tensor) on a CUDA tensor, PyTorch dispatches to
 cuDNN's optimized activation kernel.  This script shows that the built-in
 path is faster AND more numerically stable than a naive formula.

 HOW PyTorch USES cuDNN
 -----------------------
 The call chain looks like:

   torch.tanh(x)                          # Python API
     → aten::tanh(x)                      # C++ dispatcher
       → at::native::tanh_cuda(x)         # CUDA backend
         → cudnnActivationForward(...)     # cuDNN library call
           with CUDNN_ACTIVATION_TANH

 So torch.tanh() on CUDA is essentially the same cuDNN call we made
 in 01_tanh.cu, but wrapped in PyTorch's autograd + dispatch system.

 CUSTOM TANH FORMULA
 --------------------
 The mathematical identity:

   tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)

 This is correct algebraically, but has a problem on computers:
   - For x > ~44,  e^(2x) overflows to inf  (float32 max ≈ 3.4e38)
   - inf / inf = NaN  →  wrong answer!

 The built-in torch.tanh avoids this because cuDNN uses a numerically
 stable implementation (likely tanhf() in hardware, not the exp formula).

 TIMING: CUDA EVENTS vs time.perf_counter()
 -------------------------------------------
 We use torch.cuda.Event for GPU timing instead of time.perf_counter():

   - CUDA events measure GPU execution time directly
   - time.perf_counter() measures wall-clock CPU time, which includes
     Python overhead, kernel launch latency, etc.
   - For GPU benchmarks, CUDA events are the gold standard

 Usage:
   start.record()       # Insert timestamp in GPU stream
   << do work >>
   end.record()         # Insert another timestamp
   torch.cuda.synchronize()  # Wait for GPU to finish
   elapsed = start.elapsed_time(end)  # Milliseconds (float)

 COMPILE / RUN
 -------------
 python 02_compare_tanh.py

 Requires: PyTorch with CUDA support (pip install torch)
============================================================================
"""

import torch

# ──────────────────────────────────────────────────────────────────────────
#  BENCHMARK PARAMETERS
# ──────────────────────────────────────────────────────────────────────────
#  Tensor shape matches 01_tanh.cu for apples-to-apples comparison.
#
#  Warmup runs:  Trigger PyTorch's CUDA caching allocator, JIT compilation
#                of any fused kernels, and GPU clock ramp-up.
#  Benchmark runs:  Timed iterations — we average for stable results.
# ──────────────────────────────────────────────────────────────────────────
BATCH_SIZE     = 256
CHANNELS       = 32
HEIGHT         = 224
WIDTH          = 224
WARMUP_RUNS    = 10
BENCHMARK_RUNS = 100


def custom_tanh(x):
    """
    Manual tanh using the exponential formula:

        tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)

    WARNING: This overflows for |x| > ~44 because e^(2*44) > float32 max.
    For those values, e2x = inf, and inf/inf = NaN.

    We compute exp(2x) ONCE and reuse it to avoid redundant work.
    """
    e2x = torch.exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)


def benchmark(func, input_tensor, label):
    """
    Benchmark a function using CUDA events for accurate GPU timing.

    CUDA events are inserted directly into the GPU command stream,
    so they measure actual GPU execution time — not CPU-side overhead.

    Returns:
        output (Tensor): The function's output (from the last run)
        avg_ms (float):  Average time per run in milliseconds
        min_ms (float):  Best (minimum) time across all runs
    """
    # ── Warmup ──────────────────────────────────────────────────────
    # Run the function several times without timing.
    # This ensures:
    #   1. PyTorch's CUDA caching allocator has pre-allocated buffers
    #   2. Any lazy kernel compilation (JIT / torch.compile) is done
    #   3. GPU clocks have ramped up from idle
    for _ in range(WARMUP_RUNS):
        output = func(input_tensor)
    torch.cuda.synchronize()

    # ── Timed runs ──────────────────────────────────────────────────
    # We time each iteration individually with CUDA events to get
    # both average and best-case times.
    times = []
    for _ in range(BENCHMARK_RUNS):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        start.record()
        output = func(input_tensor)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))   # milliseconds

    avg_ms = sum(times) / len(times)
    min_ms = min(times)

    print(f"  {label + ':':<20s}  avg = {avg_ms:.3f} ms,  best = {min_ms:.3f} ms")

    return output, avg_ms, min_ms


def main():
    # ── Device setup ────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 62)
    print("  Custom tanh vs PyTorch Built-in tanh (cuDNN)")
    print("=" * 62)
    print()
    print(f"  GPU:            {gpu_name}")
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  CUDA (PyTorch): {torch.version.cuda}")
    print(f"  cuDNN:          {torch.backends.cudnn.version()}")
    print()

    # ── Create input tensor ─────────────────────────────────────────
    # Shape: (N, C, H, W) = (256, 32, 224, 224)
    # Total elements: 411,041,792  (same as 01_tanh.cu)
    # Values in [-1, +1] — the interesting range for tanh
    #
    # torch.rand() produces values in [0, 1).
    # Multiply by 2 and subtract 1 to get [-1, +1).
    input_tensor = torch.rand(
        (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
        device=device,
        dtype=torch.float32
    ) * 2.0 - 1.0

    total_elements = input_tensor.numel()
    tensor_bytes   = total_elements * input_tensor.element_size()

    print(f"  Tensor shape:   {BATCH_SIZE} x {CHANNELS} x {HEIGHT} x {WIDTH}")
    print(f"  Total elements: {total_elements:,}  ({total_elements / 1e6:.2f} million)")
    print(f"  Memory:         {tensor_bytes / (1024**2):.2f} MB")
    print(f"  Warmup runs:    {WARMUP_RUNS}")
    print(f"  Benchmark runs: {BENCHMARK_RUNS}")
    print()

    # ── Run benchmarks ──────────────────────────────────────────────
    print("  Timing:")
    custom_output,  custom_avg,  custom_min  = benchmark(custom_tanh,  input_tensor, "Custom tanh")
    builtin_output, builtin_avg, builtin_min = benchmark(torch.tanh,   input_tensor, "Built-in tanh")
    print()

    # ── Compute effective memory bandwidth ──────────────────────────
    # tanh is memory-bound: reads N floats, writes N floats.
    #
    #   bandwidth = (bytes_read + bytes_written) / time
    #             = 2 * tensor_bytes / avg_time_seconds
    #
    # Result in GB/s:
    #   = (2 * tensor_bytes) / (avg_ms * 1e-3) / 1e9
    #   = (2 * tensor_bytes) / (avg_ms * 1e6)
    total_bytes = 2.0 * tensor_bytes
    custom_bw   = total_bytes / (custom_avg  * 1e6)    # GB/s
    builtin_bw  = total_bytes / (builtin_avg * 1e6)

    # ── Verify correctness ──────────────────────────────────────────
    # Compare custom vs built-in.
    # Also check for NaN — custom_tanh can produce NaN on overflow.
    max_diff   = torch.max(torch.abs(custom_output - builtin_output)).item()
    custom_nan = torch.isnan(custom_output).any().item()

    # Compare both against a CPU reference
    # (move to CPU, compute tanh there, compare)
    cpu_ref         = torch.tanh(input_tensor.cpu()).to(device)
    custom_vs_cpu   = torch.max(torch.abs(custom_output  - cpu_ref)).item()
    builtin_vs_cpu  = torch.max(torch.abs(builtin_output - cpu_ref)).item()

    # ── Speedup ─────────────────────────────────────────────────────
    speedup = custom_avg / builtin_avg

    # ── Print results table ─────────────────────────────────────────
    print("  " + "-" * 58)
    print(f"  {'':24s}  {'Custom':>12s}  {'Built-in':>12s}")
    print("  " + "-" * 58)
    print(f"  {'Average time':24s}  {custom_avg:10.3f} ms  {builtin_avg:10.3f} ms")
    print(f"  {'Best time':24s}  {custom_min:10.3f} ms  {builtin_min:10.3f} ms")
    print(f"  {'Eff. bandwidth':24s}  {custom_bw:9.2f} GB/s  {builtin_bw:9.2f} GB/s")
    print(f"  {'Speedup vs custom':24s}  {'1.00x':>12s}  {speedup:10.2f}x")
    print(f"  {'Max error vs CPU':24s}  {custom_vs_cpu:12.2e}  {builtin_vs_cpu:12.2e}")
    print(f"  {'Contains NaN':24s}  {'YES':>12s}" if custom_nan else
          f"  {'Contains NaN':24s}  {'no':>12s}  {'no':>12s}")
    print(f"  {'Max diff (custom vs built-in)':30s}  {max_diff:.2e}")
    print("  " + "-" * 58)
    print()

    # ── Observations ────────────────────────────────────────────────
    if speedup > 1.05:
        print(f"  → Built-in tanh is {speedup:.1f}x faster (uses cuDNN internally)")
    elif speedup > 0.95:
        print(f"  → Performance is roughly equal ({speedup:.2f}x)")
    else:
        print(f"  → Custom tanh is faster ({1/speedup:.2f}x) — unusual!")

    if custom_nan:
        print("  → WARNING: Custom tanh produced NaN values (exp overflow)")
        print("    This happens for |x| > ~44 where exp(2x) > float32 max")

    print()


if __name__ == "__main__":
    main()
