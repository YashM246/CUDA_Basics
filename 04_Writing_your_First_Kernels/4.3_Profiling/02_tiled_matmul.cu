/*
 * Tiled Matrix Multiplication with Shared Memory
 * ===============================================
 *
 * This program demonstrates the OPTIMIZED approach to matrix multiplication
 * using shared memory tiling - a fundamental GPU optimization technique.
 *
 * ===========================================
 * Why is Naive Matrix Multiplication Slow?
 * ===========================================
 *
 * In naive matrix multiplication, each thread computes one element of C:
 *   C[row][col] = sum of A[row][k] * B[k][col] for all k
 *
 * The problem: Global memory access is SLOW (~400-800 cycles latency)
 *
 * For a 1024x1024 matrix:
 *   - Each thread reads 1024 elements from A and 1024 from B
 *   - Total threads: 1024 * 1024 = 1,048,576
 *   - Total global memory reads: 1,048,576 * 2048 = 2 BILLION reads!
 *
 * But notice: Many threads read the SAME data!
 *   - All threads in row 0 read A[0][0], A[0][1], A[0][2], ...
 *   - All threads in col 0 read B[0][0], B[1][0], B[2][0], ...
 *
 * ===========================================
 * The Tiling Solution
 * ===========================================
 *
 * Instead of each thread loading its own data from slow global memory,
 * we COOPERATIVELY load data into fast SHARED MEMORY, then compute.
 *
 * Shared memory is:
 *   - On-chip (very fast, ~5 cycles vs ~400 for global)
 *   - Shared among all threads in a block
 *   - Limited in size (~48KB per block on modern GPUs)
 *
 * The algorithm:
 *   1. Divide matrices A, B, C into TILE_SIZE x TILE_SIZE tiles
 *   2. Each thread block computes one tile of C
 *   3. For each tile of C, we need multiple tiles from A and B
 *   4. Load tiles into shared memory, compute partial sums, repeat
 *
 * Visual representation (TILE_SIZE = 2 for simplicity):
 *
 *        Matrix A              Matrix B              Matrix C
 *     [a00 a01|a02 a03]    [b00 b01|b02 b03]    [c00 c01|c02 c03]
 *     [a10 a11|a12 a13]    [b10 b11|b12 b13]    [c10 c11|c12 c13]
 *     [------+-------]  x  [------+-------]  =  [------+-------]
 *     [a20 a21|a22 a23]    [b20 b21|b22 b23]    [c20 c21|c22 c23]
 *     [a30 a31|a32 a33]    [b30 b31|b32 b33]    [c30 c31|c32 c33]
 *
 * To compute C's top-left tile [c00,c01,c10,c11]:
 *   Phase 1: Load A's [a00,a01,a10,a11] and B's [b00,b01,b10,b11] into shared mem
 *            Compute partial: c_partial = A_tile * B_tile
 *   Phase 2: Load A's [a02,a03,a12,a13] and B's [b20,b21,b30,b31] into shared mem
 *            Accumulate: c_partial += A_tile * B_tile
 *   Final:   Write c_partial to C
 *
 * ===========================================
 * Memory Access Improvement
 * ===========================================
 *
 * With tiling (TILE_SIZE = 16, matrix 1024x1024):
 *   - Each tile is loaded once into shared memory
 *   - 16 threads share each loaded tile
 *   - Global reads reduced by factor of ~16 (TILE_SIZE)
 *   - From 2 billion reads down to ~125 million!
 *
 * ===========================================
 * Performance Expectations
 * ===========================================
 *
 * On RTX 4070, you should see:
 *   - Naive: ~50-100 GFLOPS
 *   - Tiled: ~500-1000 GFLOPS (5-10x faster!)
 *   - cuBLAS: ~2000+ GFLOPS (even more optimizations)
 *
 * Compile: nvcc 02_tiled_matmul.cu -o 02_tiled_matmul.exe
 * Run:     .\02_tiled_matmul.exe
 * Profile: nsys profile .\02_tiled_matmul.exe
 */

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/*
 * ===========================================
 * TILE_SIZE Selection
 * ===========================================
 *
 * Why 16?
 *   - Shared memory per tile: 16 * 16 * 4 bytes = 1 KB
 *   - Two tiles (A and B): 2 KB per block
 *   - Leaves room for more blocks to run concurrently
 *   - 16x16 = 256 threads per block (good occupancy)
 *
 * Other common choices:
 *   - 8:  Less shared memory, but more global memory traffic
 *   - 32: More reuse, but uses 8 KB shared memory, fewer concurrent blocks
 *
 * The optimal value depends on your GPU and matrix size.
 */
#define TILE_SIZE 16

/*
 * ===========================================
 * High-precision timer (Windows compatible)
 * ===========================================
 */
double get_time() {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/*
 * ===========================================
 * Tiled Matrix Multiplication Kernel
 * ===========================================
 *
 * Computes: C = A × B
 *   - A is M × K
 *   - B is K × N
 *   - C is M × N
 *
 * Each thread block computes one TILE_SIZE × TILE_SIZE tile of C.
 * Each thread computes one element of that tile.
 */
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int M, int N, int K) {
    /*
     * Shared memory declaration
     * -------------------------
     * __shared__ means this memory is:
     *   - Allocated once per block (not per thread)
     *   - Visible to all threads in the block
     *   - Very fast (~100x faster than global memory)
     *   - Limited in size (check with cudaDeviceGetAttribute)
     *
     * We allocate two tiles: one for A, one for B
     */
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    /*
     * Thread and block indices (shorthand for readability)
     *
     * bx, by: Which tile of C this block is computing
     * tx, ty: Which element within the tile this thread handles
     */
    int bx = blockIdx.x;   // Block's x position in grid
    int by = blockIdx.y;   // Block's y position in grid
    int tx = threadIdx.x;  // Thread's x position in block
    int ty = threadIdx.y;  // Thread's y position in block

    /*
     * Calculate global row and column for this thread
     *
     * row: Which row of C (and A) this thread is responsible for
     * col: Which column of C (and B) this thread is responsible for
     *
     * Example with TILE_SIZE=16:
     *   Block (1, 2) means we're computing C's tile at row-tiles=2, col-tiles=1
     *   Thread (5, 3) within that block computes:
     *     row = 2 * 16 + 3 = 35
     *     col = 1 * 16 + 5 = 21
     *   So this thread computes C[35][21]
     */
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    /*
     * Accumulator for the dot product
     *
     * This will hold the running sum as we iterate through tiles.
     * We use a register variable (very fast) for this.
     */
    float sum = 0.0f;

    /*
     * Number of tiles we need to process
     *
     * We're computing C[row][col] = sum over k of A[row][k] * B[k][col]
     * K is the shared dimension, so we need ceil(K / TILE_SIZE) tiles.
     *
     * (K + TILE_SIZE - 1) / TILE_SIZE is integer ceiling division.
     */
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    /*
     * Main loop: iterate through all tiles along the K dimension
     *
     * For each iteration:
     *   1. Collaboratively load one tile of A and one tile of B into shared memory
     *   2. Synchronize (make sure all loads are complete)
     *   3. Compute partial dot products using the shared memory tiles
     *   4. Synchronize (make sure all computation is done before loading next tile)
     */
    for (int tile = 0; tile < numTiles; ++tile) {
        /*
         * ===========================================
         * Phase 1: Load tile of A into shared memory
         * ===========================================
         *
         * Which element of A does this thread load?
         *   - Row: same as our output row (row)
         *   - Column: tile * TILE_SIZE + tx (column within current tile)
         *
         * A is stored in row-major order:
         *   A[i][j] is at memory location A[i * K + j]
         *
         * Bounds checking:
         *   - row < M: Don't read past bottom of A
         *   - tile * TILE_SIZE + tx < K: Don't read past right edge of A
         *
         * If out of bounds, load 0.0 (won't affect the sum)
         */
        int aCol = tile * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            sharedA[ty][tx] = A[row * K + aCol];
        } else {
            sharedA[ty][tx] = 0.0f;
        }

        /*
         * ===========================================
         * Phase 2: Load tile of B into shared memory
         * ===========================================
         *
         * Which element of B does this thread load?
         *   - Row: tile * TILE_SIZE + ty (row within current tile)
         *   - Column: same as our output column (col)
         *
         * B is stored in row-major order:
         *   B[i][j] is at memory location B[i * N + j]
         *
         * Bounds checking:
         *   - tile * TILE_SIZE + ty < K: Don't read past bottom of B
         *   - col < N: Don't read past right edge of B
         */
        int bRow = tile * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            sharedB[ty][tx] = B[bRow * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }

        /*
         * ===========================================
         * Synchronization barrier #1
         * ===========================================
         *
         * __syncthreads() blocks until ALL threads in the block reach this point.
         *
         * Why is this CRITICAL?
         *   Without it, some threads might start reading from shared memory
         *   before other threads have finished writing their data!
         *
         * Example race condition without sync:
         *   Thread 0 loads sharedA[0][0], starts computing
         *   Thread 255 is still loading sharedA[15][15]
         *   Thread 0 reads sharedA[0][15] - GARBAGE! Not loaded yet!
         */
        __syncthreads();

        /*
         * ===========================================
         * Phase 3: Compute partial dot product
         * ===========================================
         *
         * Now all threads have loaded their data into shared memory.
         * Each thread computes its partial sum using the tile data.
         *
         * For element C[row][col], we need:
         *   sum += A[row][tile*TILE_SIZE + 0] * B[tile*TILE_SIZE + 0][col]
         *        + A[row][tile*TILE_SIZE + 1] * B[tile*TILE_SIZE + 1][col]
         *        + ... (TILE_SIZE terms)
         *
         * In shared memory terms:
         *   sum += sharedA[ty][0] * sharedB[0][tx]
         *        + sharedA[ty][1] * sharedB[1][tx]
         *        + ...
         *
         * This is where the MAGIC happens:
         *   - sharedA[ty][k] is read by all threads with same ty (same row)
         *   - sharedB[k][tx] is read by all threads with same tx (same col)
         *   - Data loaded ONCE, used by MANY threads = efficiency!
         */
        #pragma unroll  // Hint to compiler: unroll this small loop for speed
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }

        /*
         * ===========================================
         * Synchronization barrier #2
         * ===========================================
         *
         * Why sync AGAIN?
         *   We're about to load the NEXT tile into shared memory.
         *   Without this sync, fast threads might overwrite sharedA/sharedB
         *   while slow threads are still reading from them!
         *
         * Think of it like a relay race:
         *   - All runners must finish using the baton
         *   - Before anyone can grab a new baton
         */
        __syncthreads();
    }

    /*
     * ===========================================
     * Write final result to global memory
     * ===========================================
     *
     * After processing all tiles, 'sum' contains the complete dot product.
     * Write it to the output matrix C.
     *
     * Bounds check: Only write if this thread's position is within C's dimensions.
     * (Some threads in edge blocks might be outside the matrix)
     */
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/*
 * ===========================================
 * Naive Matrix Multiplication Kernel (for comparison)
 * ===========================================
 *
 * This is the simple, unoptimized version.
 * Each thread loads ALL its data from global memory.
 */
__global__ void matrixMultiplyNaive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/*
 * ===========================================
 * CPU Matrix Multiplication (for verification)
 * ===========================================
 */
void matrixMultiplyCPU(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/*
 * ===========================================
 * Verify GPU result against CPU
 * ===========================================
 */
bool verifyResult(float *cpu, float *gpu, int size, float tolerance = 1e-3f) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpu[i] - gpu[i]) > tolerance) {
            std::cout << "Mismatch at index " << i
                      << ": CPU=" << cpu[i] << ", GPU=" << gpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    nvtxRangePushA("Program Execution");

    /*
     * ===========================================
     * Matrix Dimensions
     * ===========================================
     *
     * A: M × K (1024 × 1024)
     * B: K × N (1024 × 1024)
     * C: M × N (1024 × 1024)
     *
     * Total floating-point operations: 2 * M * N * K
     * (multiply and add for each element = 2 ops)
     */
    const int M = 1024;  // Rows in A and C
    const int N = 1024;  // Columns in B and C
    const int K = 1024;  // Columns in A, Rows in B

    // Calculate sizes in bytes
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::cout << "Matrix Multiplication: (" << M << "x" << K << ") x ("
              << K << "x" << N << ") = (" << M << "x" << N << ")\n";
    std::cout << "TILE_SIZE: " << TILE_SIZE << "\n\n";

    /*
     * ===========================================
     * Host Memory Allocation
     * ===========================================
     */
    nvtxRangePushA("Host Memory Allocation");

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_naive = (float*)malloc(size_C);   // Result from naive kernel
    float *h_C_tiled = (float*)malloc(size_C);   // Result from tiled kernel
    float *h_C_cpu = (float*)malloc(size_C);     // Result from CPU (verification)

    if (!h_A || !h_B || !h_C_naive || !h_C_tiled || !h_C_cpu) {
        std::cerr << "Host memory allocation failed!\n";
        return -1;
    }

    nvtxRangePop();

    /*
     * ===========================================
     * Initialize Matrices
     * ===========================================
     *
     * Using small random values to avoid floating-point overflow
     * and make verification easier.
     */
    nvtxRangePushA("Matrix Initialization");

    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;  // Values 0.00 to 0.99
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    nvtxRangePop();

    /*
     * ===========================================
     * Device Memory Allocation
     * ===========================================
     */
    nvtxRangePushA("Device Memory Allocation");

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    nvtxRangePop();

    /*
     * ===========================================
     * Copy Data to Device
     * ===========================================
     */
    nvtxRangePushA("Host to Device Copy");

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    nvtxRangePop();

    /*
     * ===========================================
     * Kernel Launch Configuration
     * ===========================================
     *
     * blockDim: TILE_SIZE × TILE_SIZE threads per block
     *   - 16 × 16 = 256 threads (good occupancy)
     *   - Matches our shared memory tile size
     *
     * gridDim: Enough blocks to cover the entire output matrix C
     *   - x-direction: ceil(N / TILE_SIZE) blocks for columns
     *   - y-direction: ceil(M / TILE_SIZE) blocks for rows
     */
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "Grid: " << gridDim.x << " x " << gridDim.y << " blocks\n";
    std::cout << "Block: " << blockDim.x << " x " << blockDim.y << " threads\n\n";

    /*
     * ===========================================
     * Benchmarking Configuration
     * ===========================================
     *
     * NUM_RUNS: Number of iterations to average over
     *   - More runs = more stable/accurate timing
     *   - 20 runs is a good balance between accuracy and total runtime
     *
     * We also perform warmup runs for BOTH kernels to ensure:
     *   - JIT compilation is complete
     *   - GPU caches are populated
     *   - Memory pages are allocated
     */
    const int NUM_RUNS = 20;

    /*
     * ===========================================
     * Warmup Runs (for accurate timing)
     * ===========================================
     *
     * The first kernel launch has extra overhead:
     *   - JIT compilation of PTX to machine code
     *   - GPU context initialization
     *   - Memory page setup
     *
     * Run each kernel a few times to "warm up" before timing.
     */
    nvtxRangePushA("Warmup");

    std::cout << "Warming up kernels...\n";

    // Warmup naive kernel (3 runs)
    for (int i = 0; i < 3; i++) {
        matrixMultiplyNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Warmup tiled kernel (3 runs)
    for (int i = 0; i < 3; i++) {
        matrixMultiplyTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    nvtxRangePop();

    /*
     * ===========================================
     * Benchmark: Naive Kernel (Average of NUM_RUNS)
     * ===========================================
     */
    nvtxRangePushA("Naive Kernel Benchmark");

    std::cout << "Benchmarking Naive kernel (" << NUM_RUNS << " runs)...\n";

    double naive_time = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        double start = get_time();
        matrixMultiplyNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        naive_time += get_time() - start;
    }
    naive_time /= NUM_RUNS;  // Calculate average

    // Copy result back (from last run, for verification)
    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);

    nvtxRangePop();

    /*
     * ===========================================
     * Benchmark: Tiled Kernel (Average of NUM_RUNS)
     * ===========================================
     */
    nvtxRangePushA("Tiled Kernel Benchmark");

    std::cout << "Benchmarking Tiled kernel (" << NUM_RUNS << " runs)...\n";

    double tiled_time = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        double start = get_time();
        matrixMultiplyTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        tiled_time += get_time() - start;
    }
    tiled_time /= NUM_RUNS;  // Calculate average

    // Copy result back (from last run, for verification)
    cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost);

    nvtxRangePop();

    /*
     * ===========================================
     * CPU Verification (optional, slow for large matrices)
     * ===========================================
     */
    nvtxRangePushA("CPU Verification");

    std::cout << "Running CPU verification (this may take a moment)...\n";
    double start = get_time();
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, M, N, K);
    double cpu_time = get_time() - start;

    nvtxRangePop();

    /*
     * ===========================================
     * Verify Results
     * ===========================================
     */
    nvtxRangePushA("Result Verification");

    bool naive_correct = verifyResult(h_C_cpu, h_C_naive, M * N);
    bool tiled_correct = verifyResult(h_C_cpu, h_C_tiled, M * N);

    nvtxRangePop();

    /*
     * ===========================================
     * Calculate Performance Metrics
     * ===========================================
     *
     * FLOPS = Floating-point Operations Per Second
     *
     * For matrix multiplication C = A × B:
     *   - Each element of C requires K multiplications and K additions
     *   - Total elements in C: M × N
     *   - Total operations: 2 × M × N × K (the "2" counts both mul and add)
     */
    double ops = 2.0 * M * N * K;
    double naive_gflops = (ops / naive_time) / 1e9;
    double tiled_gflops = (ops / tiled_time) / 1e9;
    double cpu_gflops = (ops / cpu_time) / 1e9;

    /*
     * ===========================================
     * Print Results
     * ===========================================
     */
    std::cout << "\n========== Results (Average of " << NUM_RUNS << " runs) ==========\n";
    std::cout << "CPU:         " << cpu_time * 1000.0 << " ms, "
              << cpu_gflops << " GFLOPS\n";
    std::cout << "GPU (Naive): " << naive_time * 1000.0 << " ms, "
              << naive_gflops << " GFLOPS "
              << (naive_correct ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "GPU (Tiled): " << tiled_time * 1000.0 << " ms, "
              << tiled_gflops << " GFLOPS "
              << (tiled_correct ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "\nSpeedup (Tiled vs Naive): " << naive_time / tiled_time << "x\n";
    std::cout << "Speedup (Tiled vs CPU):   " << cpu_time / tiled_time << "x\n";

    /*
     * ===========================================
     * Cleanup
     * ===========================================
     */
    nvtxRangePushA("Cleanup");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    free(h_C_cpu);

    nvtxRangePop();

    /*
     * ===========================================
     * Error Checking
     * ===========================================
     */
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    nvtxRangePop();  // End "Program Execution"

    return 0;
}

/*
 * ===========================================
 * Further Optimizations (Beyond This Demo)
 * ===========================================
 *
 * 1. Memory Coalescing
 *    - Ensure threads in a warp access consecutive memory addresses
 *    - Current code already does this correctly
 *
 * 2. Bank Conflicts
 *    - Shared memory is divided into 32 banks
 *    - Concurrent access to same bank causes serialization
 *    - Solution: Pad shared memory arrays (e.g., [TILE_SIZE][TILE_SIZE+1])
 *
 * 3. Register Tiling
 *    - Load multiple elements per thread into registers
 *    - Compute larger tiles with fewer shared memory loads
 *
 * 4. Double Buffering
 *    - Load next tile while computing current tile
 *    - Hides memory latency
 *
 * 5. Vectorized Loads
 *    - Use float4 to load 4 floats at once
 *    - Better memory bandwidth utilization
 *
 * 6. Tensor Cores (RTX 20+ series)
 *    - Specialized hardware for matrix operations
 *    - Use WMMA (Warp Matrix Multiply Accumulate) API
 *    - Achieves 10-20x speedup over CUDA cores
 */
