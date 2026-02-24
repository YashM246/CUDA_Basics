"""
==============================================================================
 01_vec_add.py  –  Vector Addition in Triton vs PyTorch
==============================================================================

 WHAT THIS PROGRAM DOES
 ----------------------
 Demonstrates Triton's programming model by implementing vector addition:
   output[i] = x[i] + y[i]

 Compares performance between:
   1. Triton kernel (custom GPU kernel)
   2. PyTorch built-in (x + y, uses highly optimized CUDA kernels)

 TRITON VS CUDA: KEY DIFFERENCES
 --------------------------------
 CUDA: You write at the THREAD level (scalar program)
   - Each thread processes ONE element
   - You manually compute thread indices (blockIdx.x * blockDim.x + threadIdx.x)
   - You handle boundary checks with if statements

   __global__ void add_kernel(float* x, float* y, float* out, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           out[idx] = x[idx] + y[idx];  // Scalar operation
       }
   }

 Triton: You write at the BLOCK level (block program)
   - Each program instance processes a TILE of elements
   - Compiler generates thread-level code for you
   - Boundary checks use masks (no divergent branches)

   @triton.jit
   def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.load(y_ptr + offsets, mask=mask)
       tl.store(out_ptr + offsets, x + y, mask=mask)

 WHY USE TRITON?
 ---------------
 1. Less code: ~10 lines vs ~50+ lines of CUDA (with error checking)
 2. Python-based: No separate compilation step
 3. Performance: Matches hand-tuned CUDA for memory-bound operations
 4. Automatic optimization: Compiler handles coalescing, vectorization

 KEY TRITON CONCEPTS
 -------------------
 1. @triton.jit decorator: JIT-compiles Python to GPU code

 2. tl.program_id(axis): Which tile is this program processing?
    - Similar to blockIdx.x in CUDA
    - axis=0 for 1D grids (like this example)

 3. tl.arange(0, BLOCK_SIZE): Creates a range [0, 1, ..., BLOCK_SIZE-1]
    - Returns a tile (vector of values), not a scalar

 4. Pointer arithmetic: x_ptr + offsets
    - offsets is a VECTOR, so this creates multiple pointers
    - [x_ptr + 0, x_ptr + 1, x_ptr + 2, ...]

 5. tl.load / tl.store: Vectorized memory operations
    - Loads/stores entire tiles at once
    - Compiler generates coalesced memory accesses
    - mask parameter: load/store only valid elements (out-of-bounds → 0)

 6. tl.constexpr: Compile-time constant
    - Required for shapes (BLOCK_SIZE used in tl.arange)
    - Enables loop unrolling and other compile-time optimizations

 7. Grid launch syntax: kernel[grid](args)
    - grid = (num_blocks,) defines how many program instances to launch
    - Similar to CUDA <<<blocks, threads>>> syntax

 MASKING VS IF STATEMENTS
 -------------------------
 CUDA:
   if (idx < n) { ... }  // Causes warp divergence if some threads
                         // take the branch and others don't

 Triton:
   mask = offsets < n
   tl.load(ptr + offsets, mask=mask)  // No divergence — compiler
                                       // generates predicated instructions

 Predicated loads/stores are more efficient on GPUs because all threads
 in a warp execute the same instruction (some results are just discarded).

 BANDWIDTH CALCULATION
 ---------------------
 Vector addition is memory-bound: performance limited by DRAM bandwidth,
 not compute throughput.

 Total bytes transferred per element:
   - Read x[i]       : 4 bytes (float32)
   - Read y[i]       : 4 bytes
   - Write output[i] : 4 bytes
   - Total           : 12 bytes per element

 Effective bandwidth (GB/s) = (total bytes) / (time in seconds)
                            = (3 * n_elements * 4 bytes) / (ms * 1e-3) / 1e9
                            = (12 * n_elements) / (ms * 1e6)

 Modern GPUs (RTX 4070, A100) have theoretical bandwidth of 400-2000 GB/s.
 Achieving 80-90% of peak is considered excellent.

 EXPECTED RESULTS
 ----------------
 Triton and PyTorch should have nearly identical performance because:
   - Both are memory-bound (same DRAM bandwidth limit)
   - Both use coalesced memory access patterns
   - Vector addition is too simple to benefit from custom kernels

 You'll see a difference in more complex operations (fused kernels,
 custom memory access patterns).

 RUN
 ---
 python 01_vec_add.py

 This will:
   1. Verify correctness (Triton == PyTorch)
   2. Run benchmarks across sizes 2^12 to 2^27
   3. Generate a plot: vector-add-performance.png

 REQUIREMENTS
 ------------
 pip install torch triton
==============================================================================
"""

import torch
import triton
import triton.language as tl


# ===========================================================================
#  TRITON KERNEL
# ===========================================================================
@triton.jit
def add_kernel(
    x_ptr,                    # Pointer to first input vector
    y_ptr,                    # Pointer to second input vector
    output_ptr,               # Pointer to output vector
    n_elements,               # Size of the vector
    BLOCK_SIZE: tl.constexpr  # Number of elements each program processes
):
    """
    Vector addition kernel: output[i] = x[i] + y[i]

    Each program instance processes a tile of BLOCK_SIZE elements.
    The compiler generates threads within each tile automatically.

    Example: n_elements=1024, BLOCK_SIZE=256
      - Launch 4 programs (tiles)
      - Program 0 processes elements [0:256)
      - Program 1 processes elements [256:512)
      - Program 2 processes elements [512:768)
      - Program 3 processes elements [768:1024)
    """

    # -----------------------------------------------------------------------
    #  STEP 1: Identify which tile this program is processing
    # -----------------------------------------------------------------------
    # tl.program_id(axis=0) returns the program index along axis 0.
    # For a 1D grid, this is analogous to blockIdx.x in CUDA.
    #
    # If grid = (4,), then pid ∈ {0, 1, 2, 3}
    pid = tl.program_id(axis=0)

    # -----------------------------------------------------------------------
    #  STEP 2: Compute which elements this program handles
    # -----------------------------------------------------------------------
    # Each program processes a contiguous block of BLOCK_SIZE elements.
    #
    # block_start: First element index for this program
    # offsets: All element indices this program will process
    #
    # Example: pid=1, BLOCK_SIZE=256
    #   block_start = 1 * 256 = 256
    #   offsets = [256, 257, 258, ..., 511]
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # -----------------------------------------------------------------------
    #  STEP 3: Create a mask for boundary checking
    # -----------------------------------------------------------------------
    # If n_elements is not a multiple of BLOCK_SIZE, the last program will
    # access out-of-bounds elements. We use a mask to handle this.
    #
    # mask[i] = True if offsets[i] < n_elements, else False
    #
    # Example: n_elements=1000, BLOCK_SIZE=256, pid=3
    #   offsets = [768, 769, ..., 1023]
    #   mask = [True, True, ..., True (up to 999), False, False, ...]
    #
    # This is equivalent to the CUDA pattern:
    #   if (idx < n_elements) { ... }
    #
    # But masking is better for GPUs because it avoids warp divergence.
    # All threads execute the load/store instruction; out-of-bounds ones
    # are simply ignored (predicated execution).
    mask = offsets < n_elements

    # -----------------------------------------------------------------------
    #  STEP 4: Load tiles from global memory
    # -----------------------------------------------------------------------
    # tl.load(ptr, mask): Loads a tile of data from DRAM.
    #
    # x_ptr + offsets creates a vector of pointers:
    #   [x_ptr + 256, x_ptr + 257, ..., x_ptr + 511]
    #
    # tl.load reads all these locations in parallel (compiler generates
    # coalesced memory access: adjacent threads → adjacent addresses).
    #
    # mask=mask: Out-of-bounds loads return 0 (safe default).
    #
    # Result: x and y are tiles (vectors) of BLOCK_SIZE elements.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # -----------------------------------------------------------------------
    #  STEP 5: Compute element-wise addition
    # -----------------------------------------------------------------------
    # This operates on the ENTIRE tile at once (vectorized operation).
    # The compiler generates efficient code (e.g., SIMD instructions).
    output = x + y

    # -----------------------------------------------------------------------
    #  STEP 6: Store result back to global memory
    # -----------------------------------------------------------------------
    # tl.store(ptr, value, mask): Writes a tile to DRAM.
    #
    # mask=mask: Out-of-bounds stores are ignored (don't write garbage).
    #
    # Compiler ensures coalesced writes for maximum memory bandwidth.
    tl.store(output_ptr + offsets, output, mask=mask)


# ===========================================================================
#  HOST FUNCTION (Python wrapper for Triton kernel)
# ===========================================================================
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Triton-accelerated vector addition.

    Args:
        x: First input tensor (must be on CUDA)
        y: Second input tensor (must be on CUDA)

    Returns:
        output: x + y (computed on GPU using Triton kernel)

    Notes:
        - Tensors must be on the same CUDA device
        - Kernel runs asynchronously; synchronize if needed
    """
    # Allocate output tensor (uninitialized, will be filled by kernel)
    output = torch.empty_like(x)

    # Verify all tensors are on GPU
    assert x.is_cuda and y.is_cuda and output.is_cuda, \
        "All tensors must be on CUDA device"

    n_elements = output.numel()

    # -----------------------------------------------------------------------
    #  GRID DEFINITION
    # -----------------------------------------------------------------------
    # The grid specifies how many program instances (tiles) to launch.
    #
    # triton.cdiv(n, d) = ceiling division = (n + d - 1) // d
    #
    # Example: n_elements=10000, BLOCK_SIZE=1024
    #   grid = (triton.cdiv(10000, 1024),) = (10,)
    #   Launches 10 program instances
    #
    # The lambda is called with meta-parameters (keyword args passed to
    # the kernel). This allows the grid size to depend on BLOCK_SIZE.
    #
    # Equivalent CUDA syntax:
    #   int blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    #   add_kernel<<<blocks, BLOCK_SIZE>>>(...);
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # -----------------------------------------------------------------------
    #  KERNEL LAUNCH
    # -----------------------------------------------------------------------
    # Syntax: kernel[grid](positional_args, keyword_args)
    #
    # - grid: Defined above (returns tuple of grid dimensions)
    # - x, y, output: PyTorch tensors → automatically converted to pointers
    # - BLOCK_SIZE=1024: Meta-parameter (tl.constexpr in kernel)
    #
    # This compiles the kernel (if not cached) and launches it on the GPU.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    # IMPORTANT: Kernel runs asynchronously. Returning here does NOT mean
    # the computation is done. Use torch.cuda.synchronize() if you need
    # to wait for completion before accessing output on CPU.
    return output


# ===========================================================================
#  VERIFICATION
# ===========================================================================
def verify_correctness():
    """
    Verify that Triton kernel produces the same results as PyTorch.
    """
    print("=" * 62)
    print("  Vector Addition: Triton vs PyTorch")
    print("=" * 62)
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires a GPU.")
        return False

    print("  Device:", torch.cuda.get_device_name(0))
    print()

    # Create test inputs
    torch.manual_seed(0)
    size = 98765  # Intentionally not a power of 2 (tests masking)
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    print(f"  Testing with {size:,} elements (not a power of 2)")
    print()

    # Compute with PyTorch (reference)
    torch_output = x + y

    # Compute with Triton
    triton_output = add(x, y)

    # Wait for GPU to finish
    torch.cuda.synchronize()

    # Compare results
    max_diff = torch.max(torch.abs(triton_output - torch_output)).item()
    match = torch.allclose(triton_output, torch_output, rtol=1e-5, atol=1e-5)

    print(f"  Triton result matches PyTorch: {match}")
    print(f"  Maximum difference: {max_diff:.2e}")
    print()

    if match:
        print("  ✓ Verification PASSED")
    else:
        print("  ✗ Verification FAILED")
        print("  First 10 elements:")
        print(f"    Triton:  {triton_output[:10].tolist()}")
        print(f"    PyTorch: {torch_output[:10].tolist()}")

    print()
    return match


# ===========================================================================
#  BENCHMARK
# ===========================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument name to use as X-axis
        x_vals=[2**i for i in range(12, 28, 1)],  # X-axis values: 2^12 to 2^27
        x_log=True,  # Logarithmic X-axis
        line_arg='provider',  # Argument name for different lines
        line_vals=['triton', 'torch'],  # Line values
        line_names=['Triton', 'PyTorch'],  # Legend labels
        styles=[('blue', '-'), ('green', '-')],  # Line colors/styles
        ylabel='GB/s',  # Y-axis label
        plot_name='vector-add-performance',  # Output filename
        args={},  # Fixed arguments (none here)
    ))
def benchmark(size, provider):
    """
    Benchmark Triton vs PyTorch for vector addition.

    Args:
        size: Number of elements in the vectors
        provider: 'triton' or 'torch'

    Returns:
        (median_gbps, min_gbps, max_gbps): Bandwidth in GB/s

    Notes:
        - Bandwidth = (bytes read + bytes written) / time
        - Vector add: read x (4B) + read y (4B) + write out (4B) = 12B per element
        - Higher is better (closer to GPU's theoretical peak bandwidth)
    """
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)

    # Measure time at 50th, 20th, and 80th percentiles
    # (Median, best 20%, worst 20%)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        # PyTorch built-in (dispatches to highly optimized CUDA kernel)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x + y,
            quantiles=quantiles
        )
    elif provider == 'triton':
        # Custom Triton kernel
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y),
            quantiles=quantiles
        )

    # -----------------------------------------------------------------------
    #  BANDWIDTH CALCULATION
    # -----------------------------------------------------------------------
    # Vector addition reads 2 tensors and writes 1 tensor:
    #   - Read x: n_elements * 4 bytes (float32)
    #   - Read y: n_elements * 4 bytes
    #   - Write output: n_elements * 4 bytes
    #   - Total: 3 * n_elements * 4 bytes = 12 * n_elements bytes
    #
    # Bandwidth (GB/s) = total_bytes / time_seconds / 1e9
    #                  = (3 * n_elements * 4) / (ms * 1e-3) / 1e9
    #                  = 12 * n_elements / (ms * 1e6)
    #
    # Simplified: 3 * n_elements * element_size / ms * 1e-6
    def gbps(ms):
        return 3 * x.numel() * x.element_size() / ms * 1e-6

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ===========================================================================
#  MAIN
# ===========================================================================
if __name__ == '__main__':
    # Verify correctness first
    if verify_correctness():
        print("=" * 62)
        print("  Running performance benchmarks...")
        print("=" * 62)
        print()
        print("  This will test sizes from 2^12 (4K) to 2^27 (134M) elements")
        print("  and generate a plot: vector-add-performance.png")
        print()

        # Run benchmarks and save plot
        benchmark.run(print_data=True, show_plots=True)
    else:
        print("Skipping benchmarks due to verification failure.")
