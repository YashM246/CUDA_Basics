/*
 * Vector Addition: CPU vs GPU Benchmark
 * ======================================
 *
 * What this program demonstrates:
 *   Adds two vectors (arrays) element-by-element, comparing
 *   a sequential CPU implementation against a parallel GPU kernel.
 *   Benchmarks both to show the GPU speedup.
 *
 * Why vector addition?
 *   It's the "Hello World" of GPU computing. Each element is
 *   independent (no data dependencies), making it perfectly
 *   parallel. Every thread computes exactly one addition.
 *
 * ===========================================
 * The Math:
 * ===========================================
 *
 *   Given two vectors A and B of length N:
 *     C[i] = A[i] + B[i]   for i = 0, 1, ..., N-1
 *
 *   Example:
 *     A = [1, 2, 3, 4, 5]
 *     B = [6, 7, 8, 9, 10]
 *     C = [7, 9, 11, 13, 15]
 *
 * ===========================================
 * Program Flow:
 * ===========================================
 *
 *   1. Allocate host (CPU) memory for vectors A, B, C
 *   2. Fill A and B with random floats in [0, 1]
 *   3. Allocate device (GPU) memory
 *   4. Copy A and B from host → device
 *   5. Run warmup iterations (CPU + GPU)
 *   6. Benchmark CPU: 20 runs, average time
 *   7. Benchmark GPU: 20 runs, average time
 *   8. Copy GPU result back, verify correctness
 *   9. Free all memory
 *
 * ===========================================
 * Grid/Block Configuration:
 * ===========================================
 *
 *   BLOCK_SIZE = 256 threads per block
 *   num_blocks = ceil(N / BLOCK_SIZE)
 *            = (10,000,000 + 255) / 256
 *            = 39,063 blocks
 *
 *   Total threads launched = 39,063 × 256 = 10,000,128
 *   (128 extra threads will be idle due to bounds check)
 *
 * Compile: nvcc 01_vector_add_v1.cu -o 01_vector_add_v1.exe
 * Run:     .\01_vector_add_v1.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>           // For fabs() - floating-point absolute value
#include <cuda_runtime.h>   // CUDA runtime API (cudaMalloc, cudaMemcpy, etc.)

// On Windows, we use QueryPerformanceCounter for high-resolution timing.
// clock_gettime(CLOCK_MONOTONIC) is POSIX-only and won't compile on Windows.
#ifdef _WIN32
#include <windows.h>
#endif

// ===========================================
// Constants (preprocessor macros)
// ===========================================
// N: Total number of elements in each vector
//    10 million elements × 4 bytes (float) = ~40 MB per vector
//    3 vectors (A, B, C) = ~120 MB on GPU
#define N 10000000

// BLOCK_SIZE: Number of threads per block
// 256 is a common choice because:
//   - It's a multiple of 32 (warp size), so no partial warps
//   - 256 = 8 warps per block, good occupancy
//   - Not too large (max is 1024), leaving room for registers/shared memory
#define BLOCK_SIZE 256

/*
 * CPU Vector Addition (Sequential)
 *
 * A simple for-loop that adds elements one at a time.
 * This is the baseline we compare the GPU against.
 *
 * Parameters:
 *   a, b - Input vectors (pointers to float arrays on the host)
 *   c    - Output vector (pointer to float array on the host)
 *   n    - Number of elements
 *
 * Note on pointers:
 *   Arrays in C decay to pointers when passed to functions.
 *   a, b, and c point to the first elements of the original arrays.
 *   Writing c[i] modifies the caller's array directly (pass by pointer).
 *
 * Time complexity: O(n) - must process every element sequentially
 */
void vector_add_cpu(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
}

/*
 * GPU Vector Addition Kernel (Parallel)
 *
 * Each thread computes ONE element: c[i] = a[i] + b[i]
 * With N threads running in parallel, the entire addition
 * happens simultaneously (in theory - limited by hardware).
 *
 * Parameters:
 *   a, b - Input vectors (pointers to float arrays in GPU global memory)
 *   c    - Output vector (pointer to float array in GPU global memory)
 *   n    - Number of elements (needed for bounds checking)
 *
 * The __global__ qualifier means:
 *   - This function runs on the GPU
 *   - It's called from the CPU using <<<grid, block>>> syntax
 *   - Must return void
 *
 * Index calculation:
 *   i = blockIdx.x * blockDim.x + threadIdx.x
 *
 *   Example: Thread 2 in Block 3 with blockDim = 256
 *     i = 3 * 256 + 2 = 770
 *     This thread computes c[770] = a[770] + b[770]
 *
 * Bounds check (i < n):
 *   We launch ceil(N/256) blocks, which may create more threads
 *   than elements. Extra threads must NOT access out-of-bounds memory.
 *   Example: N=10,000,000 but we launch 10,000,128 threads.
 *   The last 128 threads (i >= 10,000,000) skip the addition.
 */
__global__ void vector_add_gpu(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] + b[i];
    }
}

/*
 * Initialize a vector with random float values in [0, 1]
 *
 * rand() returns an int in [0, RAND_MAX].
 * Dividing by RAND_MAX normalizes to [0.0, 1.0].
 * The (float) cast converts the integer result to float
 * before the division, ensuring floating-point division.
 *
 * Without the cast: rand() / RAND_MAX = integer division = always 0
 * With the cast:    (float)rand() / RAND_MAX = float division = [0.0, 1.0]
 */
void init_vector(float *vec, int n){
    for(int i = 0; i < n; i++){
        vec[i] = (float)rand() / RAND_MAX;
    }
}

/*
 * High-resolution timer function
 *
 * Returns the current time in seconds as a double.
 * Used to measure elapsed time: elapsed = end - start.
 *
 * Platform-specific implementation:
 *   Windows: QueryPerformanceCounter (sub-microsecond precision)
 *   Linux/Mac: clock_gettime with CLOCK_MONOTONIC (nanosecond precision)
 *
 * CLOCK_MONOTONIC: A clock that never jumps backward and isn't
 * affected by system time changes. Ideal for measuring intervals.
 */
double get_time(){
#ifdef _WIN32
    // Windows high-resolution timer
    // QueryPerformanceFrequency: ticks per second (constant on a given machine)
    // QueryPerformanceCounter: current tick count
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    // POSIX high-resolution timer (Linux, Mac)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

int main(){
    // ===========================================
    // Variable declarations
    // ===========================================

    // Host (CPU) pointers - prefixed with h_
    //   h_a, h_b:    Input vectors
    //   h_c_cpu:     Output from CPU computation
    //   h_c_gpu:     Output copied back from GPU (for verification)
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;

    // Device (GPU) pointers - prefixed with d_
    //   These point to GPU global memory (VRAM), NOT accessible from CPU
    float *d_a, *d_b, *d_c;

    // Total bytes needed per vector: 10,000,000 × 4 bytes = 40,000,000 bytes (~38 MB)
    // size_t is an unsigned integer type guaranteed to hold any array size
    size_t size = N * sizeof(float);

    // ===========================================
    // Step 1: Allocate host (CPU) memory
    // ===========================================
    // malloc allocates memory on the heap (RAM)
    // Returns void*, cast to float*
    // We need 4 host arrays: two inputs, two outputs (CPU result + GPU result copy)
    h_a     = (float*)malloc(size);
    h_b     = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);   // Will hold GPU result copied back to CPU

    // ===========================================
    // Step 2: Initialize input vectors with random data
    // ===========================================
    // srand seeds the random number generator
    // time(NULL) gives current time in seconds, so each run produces different values
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // ===========================================
    // Step 3: Allocate device (GPU) memory
    // ===========================================
    // cudaMalloc allocates memory in GPU VRAM
    // First argument: address of the pointer (&d_a is a float**)
    //   cudaMalloc writes the allocated GPU address into d_a
    // Second argument: number of bytes to allocate
    //
    // After this, d_a points to GPU memory - you CANNOT dereference it on the CPU
    // (e.g., d_a[0] from CPU code would crash)
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // ===========================================
    // Step 4: Copy input data from host → device
    // ===========================================
    // cudaMemcpy(destination, source, bytes, direction)
    //
    // cudaMemcpyHostToDevice: copies from CPU RAM to GPU VRAM
    // This is a synchronous call - CPU waits until the copy finishes
    //
    // We only copy inputs (A and B). C will be written by the GPU kernel.
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // ===========================================
    // Step 5: Calculate grid dimensions
    // ===========================================
    // We need one thread per element: N = 10,000,000 threads total
    // Each block has BLOCK_SIZE (256) threads
    // Number of blocks = ceil(N / BLOCK_SIZE)
    //
    // Integer ceiling division trick:
    //   ceil(a/b) = (a + b - 1) / b   (for positive integers)
    //   (10,000,000 + 255) / 256 = 10,000,255 / 256 = 39,063 blocks
    //
    // This ensures we have enough threads even when N isn't divisible by BLOCK_SIZE
    // The extra threads are handled by the bounds check (if i < n) in the kernel
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ===========================================
    // Step 6: Warmup runs
    // ===========================================
    // Why warmup?
    //   - GPU: First kernel launch has overhead from context initialization,
    //     JIT compilation of PTX to native GPU code, and memory page setup.
    //     Subsequent launches reuse the compiled kernel (cached).
    //   - CPU: First runs may cause cache misses (data not yet in L1/L2/L3).
    //     After warmup, data is in CPU cache, giving more realistic timings.
    //   - OS: The scheduler may need to allocate resources on first run.
    //
    // Without warmup, the first timed run would be artificially slow,
    // skewing the benchmark results.
    printf("Performing WarmUp Runs...\n");
    for(int i = 0; i < 3; i++){
        vector_add_cpu(h_a, h_b, h_c_cpu, N);

        // <<<num_blocks, BLOCK_SIZE>>> is the kernel launch configuration:
        //   num_blocks = 39,063 blocks in the grid (1D)
        //   BLOCK_SIZE = 256 threads per block (1D)
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);

        // Kernel launches are asynchronous - CPU continues immediately.
        // cudaDeviceSynchronize() blocks the CPU until the GPU finishes.
        // Required here so we don't start the next iteration before GPU is done.
        cudaDeviceSynchronize();
    }

    // ===========================================
    // Step 7: Benchmark CPU implementation
    // ===========================================
    // Run 20 iterations and average the time to reduce variance
    // caused by OS scheduling, cache effects, and other processes.
    printf("Benchmarking CPU Implementation...\n");
    double cpu_total_time = 0.0;
    for(int i = 0; i < 20; i++){
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;   // Accumulate elapsed seconds
    }
    double cpu_avg_time = cpu_total_time / 20.0;    // Average time in seconds

    // ===========================================
    // Step 8: Benchmark GPU implementation
    // ===========================================
    // Same approach: 20 iterations, average time.
    // cudaDeviceSynchronize() is INSIDE the timing region because:
    //   - Without it, get_time() would measure only the kernel LAUNCH time
    //     (microseconds), not the actual execution time
    //   - We need to wait for the GPU to finish to measure real execution time
    printf("Benchmarking GPU Implementation...\n");
    double gpu_total_time = 0.0;
    for(int i = 0; i < 20; i++){
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();    // Wait for GPU to finish before stopping timer
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // ===========================================
    // Step 9: Print benchmark results
    // ===========================================
    // Multiply by 1000 to convert seconds → milliseconds
    // Speedup = CPU time / GPU time (how many times faster the GPU is)
    //
    // Typical results (varies by hardware):
    //   CPU: ~15-25 ms (sequential, one core)
    //   GPU: ~0.5-2 ms (parallel, thousands of cores)
    //   Speedup: ~10-50x depending on GPU
    printf("\nResults:\n");
    printf("  Vector size:    %d elements\n", N);
    printf("  Block size:     %d threads\n", BLOCK_SIZE);
    printf("  Grid size:      %d blocks\n", num_blocks);
    printf("  CPU Avg Time:   %.3f ms\n", cpu_avg_time * 1000.0);
    printf("  GPU Avg Time:   %.3f ms\n", gpu_avg_time * 1000.0);
    printf("  Speedup:        %.2fx\n", cpu_avg_time / gpu_avg_time);

    // ===========================================
    // Step 10: Verify GPU results match CPU
    // ===========================================
    // Copy GPU output (d_c) back to host (h_c_gpu) for comparison
    // cudaMemcpyDeviceToHost: copies from GPU VRAM to CPU RAM
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    // Compare every element of CPU result vs GPU result
    // fabs() = floating-point absolute value (from math.h)
    // We use a tolerance (1e-5) instead of exact equality because:
    //   - Floating-point arithmetic isn't perfectly precise
    //   - CPU and GPU may use different floating-point rounding modes
    //   - Order of operations can differ, causing tiny rounding differences
    //   Example: (0.1 + 0.2) might be 0.30000001 on one and 0.29999999 on the other
    bool correct = true;
    for(int i = 0; i < N; i++){
        if(fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5){
            printf("  Mismatch at index %d: CPU=%.7f, GPU=%.7f\n",
                   i, h_c_cpu[i], h_c_gpu[i]);
            correct = false;
            break;
        }
    }
    printf("  Verification:   %s\n", correct ? "PASS" : "FAIL");

    // ===========================================
    // Step 11: Free all allocated memory
    // ===========================================
    // Always free memory to avoid leaks.
    // free()     - releases CPU (host) memory allocated with malloc
    // cudaFree() - releases GPU (device) memory allocated with cudaMalloc
    //
    // Order doesn't matter, but freeing device before host is conventional
    // (mirrors the allocation order in reverse)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    return 0;
}

/*
 * Why is the GPU faster?
 *
 *   CPU approach: One core processes 10 million additions sequentially.
 *     Total work = 10,000,000 additions × ~1 ns each ≈ 10-20 ms
 *
 *   GPU approach: Thousands of cores process additions in parallel.
 *     10,000,000 additions spread across ~5000+ CUDA cores
 *     Each thread does just ONE addition
 *     Most of the time is spent on memory transfers, not computation
 *
 * Where does the GPU time go?
 *
 *   The GPU benchmark time includes:
 *     1. Kernel launch overhead (~5-10 μs)
 *     2. Reading A[i] and B[i] from global memory (slow - ~400 cycles)
 *     3. The actual addition (fast - 1 cycle)
 *     4. Writing C[i] to global memory (slow - ~400 cycles)
 *
 *   Vector addition is "memory-bound" - the bottleneck is reading/writing
 *   data, NOT the computation. The GPU's high memory bandwidth (e.g.,
 *   504 GB/s on RTX 4070) is what provides the speedup, not its compute power.
 *
 * Limitations of this benchmark:
 *
 *   - get_time() includes kernel launch overhead and synchronization
 *   - CUDA events (cudaEventRecord) would give more precise GPU timing
 *   - The GPU time does NOT include the cudaMemcpy transfers
 *     (copying data to/from GPU). In a real application, these transfers
 *     can dominate total time for simple kernels like this one.
 *   - A fair comparison would include transfer time, which would
 *     reduce the apparent speedup significantly.
 *
 * Possible improvements (see later versions):
 *   - Use CUDA events for GPU timing
 *   - Include memory transfer time in the benchmark
 *   - Use pinned (page-locked) host memory for faster transfers
 *   - Use CUDA streams for overlapping transfers and computation
 *   - Error checking on all CUDA API calls
 */
