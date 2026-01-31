/*
 * Matrix Multiplication: CPU vs GPU (Naive Implementation)
 * =========================================================
 *
 * What this program demonstrates:
 *   Multiplies two matrices A × B = C using:
 *     1. CPU: Triple nested loop (O(M×N×K) operations)
 *     2. GPU: One thread per output element (naive/global memory approach)
 *
 * Why matrix multiplication?
 *   - Core operation in deep learning (neural network layers are matrix multiplies)
 *   - Compute-intensive: O(M×N×K) multiply-adds for M×N output elements
 *   - High arithmetic intensity makes it ideal for GPU acceleration
 *   - Foundation for understanding optimized algorithms (tiling, shared memory)
 *
 * ===========================================
 * Matrix Multiplication Math:
 * ===========================================
 *
 *   Given matrices:
 *     A: M × K (M rows, K columns)
 *     B: K × N (K rows, N columns)
 *     C: M × N (M rows, N columns)
 *
 *   Each element C[i][j] is the dot product of row i of A and column j of B:
 *
 *     C[i][j] = Σ(l=0 to K-1) A[i][l] * B[l][j]
 *
 *   Visual:
 *                    B (K×N)
 *                 j  →
 *              [  ·  ·  ·  ·  ]
 *              [  ·  ·  ·  ·  ]  l
 *              [  ·  ·  ·  ·  ]  ↓
 *
 *   A (M×K)       C (M×N)
 *   i→ [· · ·]    [  ·  C[i,j]  ·  ]
 *      [· · ·]    [  ·    ·     ·  ]
 *      [· · ·]    [  ·    ·     ·  ]
 *         ↑l
 *
 *   Example: 3×2 @ 2×4 = 3×4
 *
 *     A = [[1, 2],       B = [[7,  8,  9, 10],
 *          [3, 4],            [11, 12, 13, 14]]
 *          [5, 6]]
 *
 *     C[0][0] = 1*7 + 2*11 = 7 + 22 = 29
 *     C[0][1] = 1*8 + 2*12 = 8 + 24 = 32
 *     C[1][0] = 3*7 + 4*11 = 21 + 44 = 65
 *     ...
 *
 *     C = [[ 29,  32,  35,  38],
 *          [ 65,  72,  79,  86],
 *          [101, 112, 123, 134]]
 *
 * ===========================================
 * Row-Major Memory Layout:
 * ===========================================
 *
 *   C/C++ stores 2D arrays in row-major order (rows are contiguous):
 *
 *     Matrix A (3×4):        Flat array:
 *     [a00 a01 a02 a03]      [a00, a01, a02, a03, a10, a11, a12, a13, a20, ...]
 *     [a10 a11 a12 a13]       ↑row 0           ↑row 1
 *     [a20 a21 a22 a23]
 *
 *   To access element A[i][j] in a flat array with 'cols' columns:
 *     flat_index = i * cols + j
 *
 *   For our matrices:
 *     A[i][l] → A[i * K + l]   (A has K columns)
 *     B[l][j] → B[l * N + j]   (B has N columns)
 *     C[i][j] → C[i * N + j]   (C has N columns)
 *
 * ===========================================
 * GPU Thread Mapping:
 * ===========================================
 *
 *   Each thread computes ONE element of C:
 *     Thread (row, col) computes C[row][col]
 *
 *   Grid/block layout for C (M×N output):
 *     Block size: 32 × 32 = 1024 threads (maximum)
 *     Grid size:  ceil(N/32) × ceil(M/32) blocks
 *
 *   Why 32×32?
 *     - Maximizes threads per block (1024 = hardware limit)
 *     - 32 is the warp size, so x-dimension threads access consecutive memory
 *     - Each block covers a 32×32 tile of the output matrix
 *
 * ===========================================
 * Dimensions for this program:
 * ===========================================
 *
 *   A: 1024 × 512  (M × K)
 *   B: 512 × 2048  (K × N)
 *   C: 1024 × 2048 (M × N)
 *
 *   Total operations: 1024 × 2048 × 512 = 1,073,741,824 multiply-adds (~1 billion)
 *   Memory: A=2MB + B=4MB + C=8MB = 14MB on GPU
 *
 * Compile: nvcc 03_matmul.cu -o 03_matmul.exe
 * Run:     .\03_matmul.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>           // For fabs()
#include <cuda_runtime.h>

// Windows-compatible high-resolution timing
#ifdef _WIN32
#include <windows.h>
#endif

// ===========================================
// Matrix Dimensions
// ===========================================
// A (M×K) × B (K×N) = C (M×N)
//
// These dimensions are chosen to demonstrate:
// - Non-square matrices (M ≠ K ≠ N)
// - Large enough for meaningful GPU speedup
// - Small enough to fit in GPU memory comfortably

#define M 1024      // Rows in A and C
#define K 512       // Columns in A, Rows in B (shared/contraction dimension)
#define N 2048      // Columns in B and C

// Block size: 32×32 = 1024 threads (maximum per block)
// Each block computes a 32×32 tile of output matrix C
#define BLOCK_SIZE 32

/*
 * CPU Matrix Multiplication (Triple Nested Loop)
 *
 * This is the textbook O(M×N×K) algorithm:
 *   - Outer two loops iterate over output positions (i, j)
 *   - Inner loop computes the dot product along dimension K
 *
 * Parameters:
 *   A - Input matrix (M × K), stored in row-major order
 *   B - Input matrix (K × N), stored in row-major order
 *   C - Output matrix (M × N), stored in row-major order
 *   m, k, n - Matrix dimensions
 *
 * Memory access pattern:
 *   For each C[i][j]:
 *     - A[i][0..K-1]: reads one row of A (sequential, good cache locality)
 *     - B[0..K-1][j]: reads one column of B (strided, poor cache locality)
 *
 * This is why CPU matmul benefits from blocking/tiling optimizations.
 */
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n){
    for(int i = 0; i < m; i++){              // For each row of C
        for(int j = 0; j < n; j++){          // For each column of C
            float sum = 0.0f;
            for(int l = 0; l < k; l++){      // Dot product along K dimension
                // A[i][l] is at index (i * k + l) in row-major layout
                // B[l][j] is at index (l * n + j) in row-major layout
                sum += A[i * k + l] * B[l * n + j];
            }
            // C[i][j] is at index (i * n + j)
            C[i * n + j] = sum;
        }
    }
}

/*
 * GPU Matrix Multiplication Kernel (Naive/Global Memory)
 *
 * Each thread computes ONE element of the output matrix C.
 * This is called "naive" because every thread reads directly from
 * global memory without any data reuse optimizations.
 *
 * Thread-to-element mapping:
 *   row = blockIdx.y * blockDim.y + threadIdx.y  (which row of C)
 *   col = blockIdx.x * blockDim.x + threadIdx.x  (which column of C)
 *
 * Why y=row and x=col?
 *   - Convention: x varies fastest (consecutive threadIdx.x = consecutive memory)
 *   - Threads in the same warp (consecutive threadIdx.x) access consecutive
 *     columns of C, enabling coalesced writes to C[row][col..col+31]
 *   - For matrix B, threads read B[l][col..col+31] which is coalesced
 *   - For matrix A, all threads in a row read the same A[row][l] (broadcast)
 *
 * Memory access analysis (for one thread):
 *   - Reads K elements from A (one row): A[row][0], A[row][1], ..., A[row][K-1]
 *   - Reads K elements from B (one col): B[0][col], B[1][col], ..., B[K-1][col]
 *   - Writes 1 element to C: C[row][col]
 *   - Total: 2K reads + 1 write per thread
 *
 * For M×N output elements, total global memory accesses:
 *   M × N × (2K + 1) = 1024 × 2048 × (1024 + 1) ≈ 2 billion accesses
 *
 * This is inefficient! Each element of A is read N times (once per column of C).
 * Each element of B is read M times (once per row of C).
 * Tiled algorithms (using shared memory) reduce this dramatically.
 *
 * Arithmetic intensity:
 *   - Compute: K multiply-adds per output element
 *   - Memory: 2K reads per output element
 *   - Ratio: ~0.5 FLOP/byte (very low, memory-bound)
 *   - Optimized versions achieve ~10+ FLOP/byte (compute-bound)
 */
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n){
    // Calculate which element of C this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // Row index in C
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Column index in C

    // Bounds check: some threads may be outside the matrix
    // (when M or N isn't divisible by BLOCK_SIZE)
    if(row < m && col < n){
        float sum = 0.0f;

        // Compute dot product of A's row and B's column
        for(int l = 0; l < k; l++){
            // A[row][l]: element at row 'row', column 'l'
            // B[l][col]: element at row 'l', column 'col'
            sum += A[row * k + l] * B[l * n + col];
        }

        // Write result to C[row][col]
        C[row * n + col] = sum;
    }
}

/*
 * Initialize matrix with random float values in [0, 1]
 *
 * Uses flat indexing since matrices are stored as 1D arrays.
 */
void init_matrix(float *mat, int rows, int cols){
    for(int i = 0; i < rows * cols; i++){
        mat[i] = (float)rand() / RAND_MAX;
    }
}

/*
 * High-resolution timer (platform-independent)
 * Returns current time in seconds.
 */
double get_time(){
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

int main(){
    // ===========================================
    // Variable declarations
    // ===========================================

    // Host (CPU) pointers
    float *h_A, *h_B;           // Input matrices
    float *h_C_cpu, *h_C_gpu;   // Output matrices (CPU result, GPU result)

    // Device (GPU) pointers
    float *d_A, *d_B, *d_C;

    // Memory sizes in bytes
    size_t size_A = M * K * sizeof(float);   // 1024 × 512 × 4 = 2 MB
    size_t size_B = K * N * sizeof(float);   // 512 × 2048 × 4 = 4 MB
    size_t size_C = M * N * sizeof(float);   // 1024 × 2048 × 4 = 8 MB

    // Total operations for performance calculation
    // Each output element requires K multiply-adds = 2K FLOPs
    // Total: M × N × 2K FLOPs
    long long total_flops = 2LL * M * N * K;

    // ===========================================
    // Print problem configuration
    // ===========================================
    printf("=== Matrix Multiplication Benchmark (Naive) ===\n\n");
    printf("Matrix dimensions:\n");
    printf("  A: %d x %d (%.2f MB)\n", M, K, size_A / (1024.0 * 1024.0));
    printf("  B: %d x %d (%.2f MB)\n", K, N, size_B / (1024.0 * 1024.0));
    printf("  C: %d x %d (%.2f MB)\n", M, N, size_C / (1024.0 * 1024.0));
    printf("  Total GPU memory: %.2f MB\n\n", (size_A + size_B + size_C) / (1024.0 * 1024.0));
    printf("Computation:\n");
    printf("  Operations: %lld multiply-adds (%.2f GFLOPs)\n",
           total_flops / 2, total_flops / 1e9);
    printf("  Each C element: %d multiply-adds\n\n", K);

    // ===========================================
    // Step 1: Allocate host memory
    // ===========================================
    h_A     = (float*)malloc(size_A);
    h_B     = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    if(!h_A || !h_B || !h_C_cpu || !h_C_gpu){
        printf("Error: Host memory allocation failed\n");
        return 1;
    }

    // ===========================================
    // Step 2: Initialize input matrices
    // ===========================================
    srand(42);  // Fixed seed for reproducibility
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // ===========================================
    // Step 3: Allocate device memory
    // ===========================================
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // ===========================================
    // Step 4: Copy input data to device
    // ===========================================
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // ===========================================
    // Step 5: Configure grid and block dimensions
    // ===========================================
    // Block: 32×32 threads (covers a 32×32 tile of output C)
    // Grid: enough blocks to cover the entire M×N output
    //
    // Grid dimensions:
    //   x-dimension (columns): ceil(N / 32) = ceil(2048 / 32) = 64 blocks
    //   y-dimension (rows):    ceil(M / 32) = ceil(1024 / 32) = 32 blocks
    //   Total: 64 × 32 = 2,048 blocks
    //
    // Total threads: 2,048 blocks × 1,024 threads = 2,097,152 threads
    // Output elements: 1024 × 2048 = 2,097,152 (perfect match when divisible)

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);   // 32 × 32 = 1024 threads
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,   // Columns: ceil(2048/32) = 64
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE    // Rows: ceil(1024/32) = 32
    );

    printf("Kernel configuration:\n");
    printf("  Block size: %d x %d = %d threads\n",
           BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * BLOCK_SIZE);
    printf("  Grid size:  %d x %d = %d blocks\n",
           grid.x, grid.y, grid.x * grid.y);
    printf("  Total threads: %d\n\n", grid.x * grid.y * BLOCK_SIZE * BLOCK_SIZE);

    // ===========================================
    // Step 6: Warmup runs
    // ===========================================
    // Matrix multiplication is compute-heavy, so warmup is important:
    // - GPU: JIT compilation, kernel caching, memory page setup
    // - CPU: Instruction cache warmup, branch predictor training
    printf("Performing warmup runs...\n");

    // CPU warmup (just 1 run since it's slow)
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);

    // GPU warmup (3 runs)
    for(int i = 0; i < 3; i++){
        matmul_gpu<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    // ===========================================
    // Step 7: Benchmark CPU (fewer iterations since it's slow)
    // ===========================================
    // CPU matmul is O(M×N×K) = O(1 billion) operations
    // At ~3 GHz, this takes several seconds per run
    printf("Benchmarking CPU (3 runs)...\n");

    double cpu_total = 0.0;
    int cpu_runs = 3;   // Fewer runs since CPU is much slower
    for(int i = 0; i < cpu_runs; i++){
        double start = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end = get_time();
        cpu_total += end - start;
    }
    double cpu_avg = cpu_total / cpu_runs;
    double cpu_gflops = (total_flops / cpu_avg) / 1e9;

    // ===========================================
    // Step 8: Benchmark GPU
    // ===========================================
    printf("Benchmarking GPU (20 runs)...\n");

    double gpu_total = 0.0;
    int gpu_runs = 20;
    for(int i = 0; i < gpu_runs; i++){
        double start = get_time();
        matmul_gpu<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_total += end - start;
    }
    double gpu_avg = gpu_total / gpu_runs;
    double gpu_gflops = (total_flops / gpu_avg) / 1e9;

    // ===========================================
    // Step 9: Print benchmark results
    // ===========================================
    printf("\n=== Results ===\n");
    printf("  CPU:\n");
    printf("    Time:       %.3f ms\n", cpu_avg * 1000.0);
    printf("    Throughput: %.2f GFLOPS\n", cpu_gflops);
    printf("  GPU (Naive):\n");
    printf("    Time:       %.3f ms\n", gpu_avg * 1000.0);
    printf("    Throughput: %.2f GFLOPS\n", gpu_gflops);
    printf("\n");
    printf("  Speedup (CPU vs GPU): %.2fx\n", cpu_avg / gpu_avg);
    printf("  GPU efficiency: %.1f%% of theoretical peak\n",
           (gpu_gflops / 29000.0) * 100.0);  // RTX 4070 ~29 TFLOPS FP32

    // ===========================================
    // Step 10: Copy GPU result back and verify
    // ===========================================
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results
    // Use relative tolerance since values can be large
    bool correct = true;
    int error_count = 0;
    float max_error = 0.0f;
    for(int i = 0; i < M * N; i++){
        float diff = fabs(h_C_cpu[i] - h_C_gpu[i]);
        float rel_error = diff / (fabs(h_C_cpu[i]) + 1e-6f);

        if(diff > max_error) max_error = diff;

        // Allow small relative error (floating-point accumulation differs)
        if(rel_error > 1e-4f){
            if(error_count < 5){
                printf("  Mismatch at [%d]: CPU=%.6f, GPU=%.6f, diff=%.6f\n",
                       i, h_C_cpu[i], h_C_gpu[i], diff);
            }
            error_count++;
            correct = false;
        }
    }

    printf("\n=== Verification ===\n");
    printf("  Max absolute error: %.6f\n", max_error);
    printf("  Errors found: %d / %d elements\n", error_count, M * N);
    printf("  Result: %s\n", correct ? "PASS" : "FAIL");

    // ===========================================
    // Step 11: Cleanup
    // ===========================================
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}

/*
 * ===========================================
 * Why is this "Naive" implementation slow?
 * ===========================================
 *
 * Problem: Excessive global memory access
 *
 *   Each thread computing C[i][j] reads:
 *     - K elements from row i of A
 *     - K elements from column j of B
 *
 *   For the entire C matrix (M×N elements):
 *     - Total reads of A: M × N × K = M×K read N times each
 *     - Total reads of B: M × N × K = K×N read M times each
 *
 *   Data reuse opportunity (not exploited here):
 *     - All threads in the same row of C read the same row of A
 *     - All threads in the same column of C read the same column of B
 *     - If we could share this data within a block... shared memory!
 *
 * ===========================================
 * The Tiled/Shared Memory Approach (next optimization):
 * ===========================================
 *
 *   Instead of each thread reading all K elements:
 *     1. Divide the dot product into tiles of size TILE_SIZE
 *     2. All threads in a block cooperatively load tiles of A and B
 *        into shared memory (fast, on-chip)
 *     3. Compute partial dot products from shared memory
 *     4. Repeat for all tiles, accumulating the sum
 *
 *   Memory access reduction:
 *     - Naive: M×N×K reads from global memory
 *     - Tiled: M×N×K/TILE reads from global, rest from shared
 *
 *   Typical speedup: 5-20x over naive, depending on matrix size
 *
 * ===========================================
 * GEMM Terminology (used in cuBLAS and deep learning):
 * ===========================================
 *
 *   GEMM = General Matrix-Matrix Multiplication
 *   Full formula: C = α × A × B + β × C
 *
 *   Variants by precision:
 *     SGEMM - Single precision (float32) - what we're doing
 *     DGEMM - Double precision (float64)
 *     HGEMM - Half precision (float16) - used in AI training
 *
 *   cuBLAS (NVIDIA's optimized library) achieves near-peak GPU performance
 *   through advanced tiling, tensor cores, and memory optimization.
 *   Our naive implementation: ~1-5% of cuBLAS performance
 *   Tiled implementation: ~10-30% of cuBLAS
 *   Advanced implementation: ~50-80% of cuBLAS
 *
 * ===========================================
 * Performance expectations:
 * ===========================================
 *
 *   RTX 4070 theoretical peak: ~29 TFLOPS (FP32)
 *   Naive matmul typically achieves: 100-500 GFLOPS (0.3-1.7% efficiency)
 *   Tiled matmul can achieve: 1-5 TFLOPS (3-17% efficiency)
 *   cuBLAS achieves: 20+ TFLOPS (70%+ efficiency)
 *
 *   The naive kernel is heavily memory-bound because:
 *   - Arithmetic intensity: 2 FLOPs per 8 bytes loaded = 0.25 FLOP/byte
 *   - GPU needs ~10 FLOP/byte to be compute-bound
 *   - We're limited by memory bandwidth (~500 GB/s), not compute
 */
