/*
 * cuBLAS vs cuBLASLt vs Naive Kernel - Performance Benchmark
 * ===========================================================
 *
 * This program benchmarks 5 different matrix multiplication approaches
 * on the SAME matrices and compares their performance and accuracy:
 *
 *   1. cuBLAS SGEMM    (FP32)  - Standard cuBLAS, single precision
 *   2. cuBLASLt Matmul (FP32)  - Lightweight cuBLAS, single precision
 *   3. cuBLAS HGEMM    (FP16)  - Standard cuBLAS, half precision
 *   4. cuBLASLt Matmul (FP16)  - Lightweight cuBLAS, half precision
 *   5. Naive CUDA kernel (FP32) - Simple triple-loop kernel (our baseline)
 *
 * ===========================================
 * Why Benchmark?
 * ===========================================
 *
 * Writing your own CUDA kernel for matmul is educational, but in practice
 * the NVIDIA libraries are MUCH faster. This program proves it with numbers.
 *
 * Expected result hierarchy (fastest to slowest):
 *   cuBLAS/cuBLASLt FP16 > cuBLAS/cuBLASLt FP32 >> Naive kernel
 *
 * cuBLAS and cuBLASLt should be similar speed (same underlying algorithms).
 * FP16 is faster because:
 *   - Half the memory bandwidth (16-bit vs 32-bit per element)
 *   - Tensor Cores process FP16 much faster than FP32
 *   - Trade-off: lower precision (about 3-4 decimal digits vs 7 for FP32)
 *
 * The naive kernel is 10-50x slower because it doesn't use:
 *   - Shared memory tiling
 *   - Memory coalescing optimization
 *   - Tensor Cores
 *   - Register-level optimizations
 *
 * ===========================================
 * GFLOPS - How We Measure Performance
 * ===========================================
 *
 * GFLOPS = Giga Floating-Point Operations Per Second
 *
 * For matrix multiply C = A * B where A is M*K and B is K*N:
 *   - Each element of C requires K multiplies + K adds = 2K FLOPs
 *   - C has M*N elements
 *   - Total FLOPs = 2 * M * N * K
 *
 * GFLOPS = (2 * M * N * K) / (time_in_seconds * 10^9)
 * TFLOPS = GFLOPS / 1000
 *
 * Example: RTX 4070 theoretical peaks:
 *   FP32: ~29 TFLOPS
 *   FP16: ~58 TFLOPS (with Tensor Cores)
 *
 * The closer your GFLOPS gets to these peaks, the better your implementation.
 *
 * ===========================================
 * CUDA Event Timing
 * ===========================================
 *
 * CPU timers (clock(), std::chrono) measure wall-clock time and include
 * kernel launch overhead. CUDA events are more accurate for GPU timing:
 *
 *   cudaEventRecord(start)   → GPU timestamps the start
 *   kernel<<<...>>>()        → GPU does work
 *   cudaEventRecord(stop)    → GPU timestamps the end
 *   cudaEventSynchronize(stop) → CPU waits for stop event
 *   cudaEventElapsedTime()   → Returns GPU time in milliseconds
 *
 * This measures ONLY the GPU execution time, not CPU overhead.
 *
 * ===========================================
 * Warmup Runs - Why They Matter
 * ===========================================
 *
 * The first kernel launch is always slower due to:
 *   1. GPU clock ramping (GPU boosts from idle to full speed)
 *   2. JIT compilation (PTX → machine code on first launch)
 *   3. Memory page allocation (first-touch penalty)
 *   4. L2 cache cold start
 *
 * Running a few "warmup" iterations before timing gives more
 * representative results. We use 3 warmup + 20 benchmark runs.
 *
 * Compile: nvcc 03_compare_cublas_cublaslt.cu -o 03_compare_cublas_cublaslt.exe -lcublas -lcublasLt
 * Run:     .\03_compare_cublas_cublaslt.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>          /* cuBLAS standard API (cublasSgemm, cublasHgemm)    */
#include <cublasLt.h>           /* cuBLASLt API (cublasLtMatmul, descriptors)        */
#include <cuda_fp16.h>          /* FP16 half type (__float2half, __half2float)        */

/*
 * C++ headers used for specific features:
 *   <vector>      - Dynamic arrays (std::vector) for host matrices
 *   <functional>  - std::function for passing lambdas to benchmark functions
 *   <random>      - Random number generation (better than rand())
 *   <numeric>     - std::accumulate for averaging benchmark times
 */
#include <vector>
#include <functional>
#include <random>
#include <numeric>

/*
 * ===========================================
 * Error Checking Macros
 * ===========================================
 *
 * Wrapped in do { } while(0) so they work safely in if/else chains:
 *
 *   if (condition)
 *       CHECK_CUDA(call);   // Without do/while, the ; breaks the if/else
 *   else
 *       something();
 *
 * With just { }, the semicolon after CHECK_CUDA(call); creates an empty
 * statement that terminates the if, making the else orphaned. do/while
 * absorbs the semicolon correctly.
 */
#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                    __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/*
 * ===========================================
 * Matrix Dimensions
 * ===========================================
 *
 * A (M x K) * B (K x N) = C (M x N)
 *
 * Using large matrices (4096x1024 * 1024x4096) to see meaningful
 * performance differences. Small matrices don't stress the GPU enough
 * and most of the time is spent on launch overhead.
 *
 * Total FLOPs = 2 * M * N * K = 2 * 4096 * 4096 * 1024 = 34,359,738,368
 *             = ~34.4 GFLOPs per matmul
 */
const int M = 4096;
const int K = 1024;
const int N = 4096;

/*
 * ===========================================
 * Naive Matrix Multiply Kernel
 * ===========================================
 *
 * The simplest possible GPU matmul: each thread computes one element of C.
 *
 * Thread (row, col) computes:
 *   C[row][col] = sum over k: A[row][k] * B[k][col]
 *
 * This works in ROW-MAJOR order (same as C/C++), so no column-major
 * tricks needed. This is intentional - we want a simple, correct
 * reference to verify cuBLAS results against.
 *
 * Why is this slow?
 *   - Each thread reads an entire row of A and column of B from GLOBAL memory
 *   - No shared memory tiling → redundant reads (same row/col read by many threads)
 *   - B access pattern is stride-N (column access) → bad memory coalescing
 *   - No Tensor Core usage
 *
 * For a 4096x4096 matrix, each element needs 1024 multiply-adds.
 * There are 4096*4096 = 16M elements to compute = 16M threads total.
 */
__global__ void naiveMatrixMultiply(const float *A, const float *B, float *C,
                                     int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

/*
 * ===========================================
 * Initialize Matrix with Random Values
 * ===========================================
 *
 * Uses C++ <random> instead of rand() for better quality random numbers.
 * Values in range [-0.5, 0.5] to keep products small and avoid FP16 overflow.
 *
 * FP16 range: [-65504, 65504]. If we used values like [-100, 100],
 * the dot product sums could exceed FP16 range and produce inf.
 * With [-0.5, 0.5], the maximum possible sum for K=1024 is about 256,
 * well within FP16 range.
 *
 * std::random_device - Hardware random seed (non-deterministic)
 * std::mt19937       - Mersenne Twister PRNG (fast, high-quality)
 * std::uniform_real_distribution - Uniform distribution over [a, b]
 */
void initializeMatrix(std::vector<float> &matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(dis(gen));
    }
}

/*
 * ===========================================
 * Verify Results Against Reference
 * ===========================================
 *
 * Compares two result matrices element-by-element.
 *
 * Uses ABSOLUTE error: |expected[i] - actual[i]|
 * Not relative error (which would be |expected - actual| / |expected|).
 *
 * Why absolute error?
 *   - Simple and works well when values are in a known range
 *   - Relative error breaks when expected ≈ 0 (division by near-zero)
 *   - For our random [-0.5, 0.5] inputs, absolute error is sufficient
 *
 * Tolerances:
 *   FP32: 1e-2 (generous because different accumulation order → different rounding)
 *   FP16: 5e-1 (very loose because FP16 has only ~3 decimal digits of precision)
 *
 * Why don't FP32 results match exactly?
 *   The naive kernel accumulates left-to-right: a0*b0 + a1*b1 + a2*b2 + ...
 *   cuBLAS may use tiled accumulation, different order, or FMA instructions.
 *   Floating-point addition is NOT associative: (a+b)+c != a+(b+c)
 *   Different summation order → different rounding errors.
 */
bool verifyResults(const std::vector<float> &expected,
                   const std::vector<float> &actual,
                   float tolerance = 1e-2f) {
    if (expected.size() != actual.size()) return false;

    for (size_t i = 0; i < expected.size(); ++i) {
        float abs_error = fabsf(expected[i] - actual[i]);
        if (abs_error > tolerance) {
            fprintf(stderr, "  Mismatch at index %zu: expected %.6f, got %.6f, "
                    "abs error: %.6f\n", i, expected[i], actual[i], abs_error);
            return false;
        }
    }
    return true;
}

/*
 * ===========================================
 * CUDA Event-Based Timing
 * ===========================================
 *
 * Measures GPU execution time for a single run of the given function.
 *
 * Why use std::function<void()>?
 * -----------------------------
 * std::function is a C++ "type-erased callable wrapper". It can hold:
 *   - Regular functions:  time_kernel(myFunction)
 *   - Lambdas:            time_kernel([&]() { cublasSgemm(...); })
 *   - Function objects:   time_kernel(myFunctor)
 *
 * The [&]() { ... } syntax is a LAMBDA (anonymous function):
 *   [&]  = capture all variables from outer scope by reference
 *   ()   = no parameters
 *   { }  = function body
 *
 * This lets us pass any GPU operation to the timing function without
 * writing a separate timing wrapper for each cuBLAS call.
 *
 * Returns: elapsed time in milliseconds (float).
 */
float time_kernel(std::function<void()> kernel_func) {
    cudaEvent_t start, stop;
    float elapsed_time;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel_func();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_time;
}

/*
 * ===========================================
 * Benchmark with Warmup
 * ===========================================
 *
 * Runs the kernel multiple times and returns the average time.
 *
 * Pattern:
 *   1. Run warmup_runs times (results discarded)
 *   2. Run benchmark_runs times (timed with CUDA events)
 *   3. Return average of benchmark times
 *
 * std::accumulate sums all elements in the vector:
 *   accumulate(begin, end, initial_value) → initial_value + sum of elements
 */
float benchmark_kernel(std::function<void()> kernel_func,
                       int warmup_runs, int benchmark_runs) {
    /* Warmup: let GPU ramp up clocks and cache data */
    for (int i = 0; i < warmup_runs; ++i) {
        kernel_func();
    }

    /* Timed benchmark runs */
    std::vector<float> times;
    for (int i = 0; i < benchmark_runs; ++i) {
        float time = time_kernel(kernel_func);
        times.push_back(time);
    }

    /* Average time across all benchmark runs */
    float avg_time = std::accumulate(times.begin(), times.end(), 0.0f) / benchmark_runs;
    return avg_time;
}

/*
 * ===========================================
 * Compute GFLOPS
 * ===========================================
 *
 * GFLOPS = (2 * M * N * K) / (time_ms * 10^6)
 *
 * The factor of 2 accounts for one multiply + one add per element
 * in the dot product (these are "fused" in FMA but still counted).
 */
float compute_gflops(int m, int n, int k, float time_ms) {
    double flops = 2.0 * m * n * k;
    return (float)(flops / (time_ms * 1e6));
}

int main() {
    printf("==========================================================\n");
    printf(" cuBLAS vs cuBLASLt vs Naive Kernel - Performance Benchmark\n");
    printf("==========================================================\n\n");
    printf("Matrix dimensions: A (%dx%d) * B (%dx%d) = C (%dx%d)\n", M, K, K, N, M, N);
    printf("Total FLOPs per matmul: %.2f GFLOP\n\n",
           (2.0 * M * N * K) / 1e9);

    /*
     * ===========================================
     * Host Memory Allocation
     * ===========================================
     *
     * Using std::vector for automatic memory management.
     * vector(size) allocates and zero-initializes the array.
     * .data() returns the raw pointer (needed for CUDA memcpy).
     */
    std::vector<float> h_A(M * K), h_B(K * N);
    std::vector<float> h_C_cublas_fp32(M * N);
    std::vector<float> h_C_cublasLt_fp32(M * N);
    std::vector<float> h_C_cublas_fp16(M * N);
    std::vector<float> h_C_cublasLt_fp16(M * N);
    std::vector<float> h_C_naive(M * N);

    /* Initialize with random values in [-0.5, 0.5] */
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    /*
     * ===========================================
     * Device Memory Allocation
     * ===========================================
     *
     * We need separate FP32 and FP16 buffers because:
     *   - cuBLAS SGEMM/cuBLASLt FP32 operate on float*
     *   - cuBLAS HGEMM/cuBLASLt FP16 operate on half*
     *   - The naive kernel operates on float* (reuses FP32 buffers)
     *
     * Note: d_C is shared between FP32 operations and the naive kernel.
     * We copy results out after each benchmark before the next one overwrites.
     */
    float *d_A, *d_B, *d_C;
    half *d_A_half, *d_B_half, *d_C_half;

    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A_half, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_half, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_half, M * N * sizeof(half)));

    /* Copy FP32 data to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    /*
     * Convert float → half on host, then copy to device.
     *
     * __float2half() converts one float to half.
     * Precision loss occurs here (7 decimal digits → 3 decimal digits).
     * Values outside [-65504, 65504] become inf in FP16.
     */
    std::vector<half> h_A_half(M * K), h_B_half(K * N);
    for (int i = 0; i < M * K; ++i) h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_half[i] = __float2half(h_B[i]);

    CHECK_CUDA(cudaMemcpy(d_A_half, h_A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_half, h_B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    /*
     * ===========================================
     * Create Library Handles
     * ===========================================
     *
     * cuBLAS and cuBLASLt use DIFFERENT handle types:
     *   cublasHandle_t   → for cublasSgemm, cublasHgemm, etc.
     *   cublasLtHandle_t → for cublasLtMatmul, etc.
     *
     * You CANNOT mix them. Each library needs its own handle.
     */
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    cublasLtHandle_t cublasLt_handle;
    CHECK_CUBLAS(cublasLtCreate(&cublasLt_handle));

    /* Scalars for GEMM: C = alpha*A*B + beta*C */
    float alpha = 1.0f, beta = 0.0f;
    half alpha_half = __float2half(1.0f), beta_half = __float2half(0.0f);

    const int warmup_runs = 3;
    const int benchmark_runs = 20;

    printf("Benchmark config: %d warmup runs, %d timed runs (averaged)\n\n",
           warmup_runs, benchmark_runs);

    /*
     * ===========================================
     * Benchmark 1: cuBLAS SGEMM (FP32)
     * ===========================================
     *
     * cublasSgemm = the simplest way to do GPU matmul.
     * One function call with all parameters inline.
     *
     * Row-major swap trick (see 01_sgemm_hgemm_cublas.cu):
     *   - Swap A↔B and m↔n
     *   - Pass d_B as cuBLAS's A, d_A as cuBLAS's B
     *   - cuBLAS computes B^T * A^T = (A*B)^T in column-major
     *   - Reading column-major (A*B)^T as row-major gives us A*B ✓
     *
     * cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
     * With swap:  m=N, n=M, k=K, A=d_B, lda=N, B=d_A, ldb=K, C=d_C, ldc=N
     */
    printf("Benchmarking cuBLAS SGEMM (FP32)...\n");

    float cublas_fp32_time = benchmark_kernel([&]() {
        CHECK_CUBLAS(cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,             /* m, n, k (swapped: N first, M second) */
            &alpha,
            d_B, N,              /* cuBLAS's A = our B, lda = N */
            d_A, K,              /* cuBLAS's B = our A, ldb = K */
            &beta,
            d_C, N               /* C output, ldc = N */
        ));
    }, warmup_runs, benchmark_runs);

    CHECK_CUDA(cudaMemcpy(h_C_cublas_fp32.data(), d_C, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float cublas_fp32_gflops = compute_gflops(M, N, K, cublas_fp32_time);
    printf("  Average time: %.3f ms  |  %.1f GFLOPS  (%.2f TFLOPS)\n\n",
           cublas_fp32_time, cublas_fp32_gflops, cublas_fp32_gflops / 1000.0f);

    /*
     * ===========================================
     * Benchmark 2: cuBLASLt Matmul (FP32)
     * ===========================================
     *
     * Same computation as cuBLAS SGEMM, but using the descriptor-based API.
     * More setup code, but allows finer control over algorithms and fusions.
     *
     * Steps:
     *   1. Create matmul descriptor (compute type + scale type)
     *   2. Create matrix layout descriptors (dimensions + data type)
     *   3. Call cublasLtMatmul
     *
     * Layout descriptors use the same swap trick:
     *   Adesc: our A (M×K) → column-major view → rows=K, cols=M, ld=K
     *   Bdesc: our B (K×N) → column-major view → rows=N, cols=K, ld=N
     *   Cdesc: output (M×N) → column-major view → rows=N, cols=M, ld=N
     *
     * In cublasLtMatmul, we swap: cuBLAS's A = our d_B, cuBLAS's B = our d_A
     */
    printf("Benchmarking cuBLASLt Matmul (FP32)...\n");

    cublasLtMatmulDesc_t matmulDesc_fp32 = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp32, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    /* Layout for our A matrix (passed as cuBLAS's B) */
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, K, M, K));
    /* Layout for our B matrix (passed as cuBLAS's A) */
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, K, N));
    /* Layout for output C/D matrix */
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, N, M, N));

    /*
     * Note: We don't explicitly set TRANSA/TRANSB attributes here.
     * The default is CUBLAS_OP_N (no transpose) for both, which is what
     * we want since the swap trick already handles row/column-major.
     * See 02_matmul_cublaslt.cu for the explicit version.
     */

    float cublasLt_fp32_time = benchmark_kernel([&]() {
        CHECK_CUBLAS(cublasLtMatmul(
            cublasLt_handle,            /* handle                        */
            matmulDesc_fp32,            /* computation descriptor        */
            &alpha,                     /* alpha = 1.0                   */
            d_B, Bdesc,                 /* cuBLAS's A = our B + layout   */
            d_A, Adesc,                 /* cuBLAS's B = our A + layout   */
            &beta,                      /* beta = 0.0                    */
            d_C, Cdesc,                 /* C (input for beta*C, ignored) */
            d_C, Cdesc,                 /* D (OUTPUT, same as C, ok since beta=0) */
            NULL,                       /* algo: NULL = auto-select      */
            NULL,                       /* workspace: none               */
            0,                          /* workspaceSize: 0              */
            0                           /* stream: default               */
        ));
    }, warmup_runs, benchmark_runs);

    CHECK_CUDA(cudaMemcpy(h_C_cublasLt_fp32.data(), d_C, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float cublasLt_fp32_gflops = compute_gflops(M, N, K, cublasLt_fp32_time);
    printf("  Average time: %.3f ms  |  %.1f GFLOPS  (%.2f TFLOPS)\n\n",
           cublasLt_fp32_time, cublasLt_fp32_gflops, cublasLt_fp32_gflops / 1000.0f);

    /*
     * ===========================================
     * Benchmark 3: cuBLAS HGEMM (FP16)
     * ===========================================
     *
     * cublasHgemm = Half-precision GEMM.
     * Same API as cublasSgemm but takes half* pointers and half alpha/beta.
     *
     * FP16 benefits:
     *   - 2x less memory bandwidth (16-bit vs 32-bit per element)
     *   - Tensor Cores can process FP16 at 2x+ the rate of FP32
     *   - Lower precision: ~3.3 decimal digits vs ~7.2 for FP32
     *
     * Modern GPUs (Volta+) have dedicated Tensor Cores that perform
     * 4x4 matrix multiply-accumulate in hardware. cuBLAS automatically
     * uses them when data type and matrix sizes are compatible.
     */
    printf("Benchmarking cuBLAS HGEMM (FP16)...\n");

    float cublas_fp16_time = benchmark_kernel([&]() {
        CHECK_CUBLAS(cublasHgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha_half,
            d_B_half, N,
            d_A_half, K,
            &beta_half,
            d_C_half, N
        ));
    }, warmup_runs, benchmark_runs);

    /* Convert FP16 results back to FP32 for verification */
    std::vector<half> h_C_half(M * N);
    CHECK_CUDA(cudaMemcpy(h_C_half.data(), d_C_half, M * N * sizeof(half),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; ++i) {
        h_C_cublas_fp16[i] = __half2float(h_C_half[i]);
    }

    float cublas_fp16_gflops = compute_gflops(M, N, K, cublas_fp16_time);
    printf("  Average time: %.3f ms  |  %.1f GFLOPS  (%.2f TFLOPS)\n\n",
           cublas_fp16_time, cublas_fp16_gflops, cublas_fp16_gflops / 1000.0f);

    /*
     * ===========================================
     * Benchmark 4: cuBLASLt Matmul (FP16)
     * ===========================================
     *
     * cuBLASLt with FP16 data and FP16 compute.
     * Same descriptor pattern as FP32 but with CUDA_R_16F and CUBLAS_COMPUTE_16F.
     *
     * CUBLAS_COMPUTE_16F: Accumulates dot products in FP16.
     * For better accuracy, you could use CUBLAS_COMPUTE_32F with FP16 data
     * (mixed precision), but we use pure FP16 here to match cublasHgemm.
     */
    printf("Benchmarking cuBLASLt Matmul (FP16)...\n");

    cublasLtMatmulDesc_t matmulDesc_fp16 = NULL;
    cublasLtMatrixLayout_t Adesc_half = NULL, Bdesc_half = NULL, Cdesc_half = NULL;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp16, CUBLAS_COMPUTE_16F, CUDA_R_16F));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc_half, CUDA_R_16F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc_half, CUDA_R_16F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc_half, CUDA_R_16F, N, M, N));

    float cublasLt_fp16_time = benchmark_kernel([&]() {
        CHECK_CUBLAS(cublasLtMatmul(
            cublasLt_handle,
            matmulDesc_fp16,
            &alpha_half,
            d_B_half, Bdesc_half,       /* cuBLAS's A = our B */
            d_A_half, Adesc_half,       /* cuBLAS's B = our A */
            &beta_half,
            d_C_half, Cdesc_half,       /* C (input)  */
            d_C_half, Cdesc_half,       /* D (OUTPUT) */
            NULL, NULL, 0, 0            /* algo, workspace, size, stream */
        ));
    }, warmup_runs, benchmark_runs);

    CHECK_CUDA(cudaMemcpy(h_C_half.data(), d_C_half, M * N * sizeof(half),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; ++i) {
        h_C_cublasLt_fp16[i] = __half2float(h_C_half[i]);
    }

    float cublasLt_fp16_gflops = compute_gflops(M, N, K, cublasLt_fp16_time);
    printf("  Average time: %.3f ms  |  %.1f GFLOPS  (%.2f TFLOPS)\n\n",
           cublasLt_fp16_time, cublasLt_fp16_gflops, cublasLt_fp16_gflops / 1000.0f);

    /*
     * ===========================================
     * Benchmark 5: Naive CUDA Kernel (FP32)
     * ===========================================
     *
     * Our handwritten kernel for comparison. Uses 32x32 thread blocks.
     *
     * Grid/block calculation:
     *   threads_per_block = (32, 32) = 1024 threads per block (maximum)
     *   grid_size = ceil(N/32) x ceil(M/32) blocks to cover entire matrix
     *
     * Why 32x32?
     *   - 32 threads = 1 warp. Using 32 in x-dimension ensures
     *     threads in the same warp access consecutive memory.
     *   - 32*32 = 1024 = maximum threads per block on most GPUs.
     *   - Maximizes occupancy for simple kernels.
     */
    printf("Benchmarking Naive CUDA Kernel (FP32)...\n");

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    float naive_cuda_time = benchmark_kernel([&]() {
        naiveMatrixMultiply<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        /* Check for kernel launch errors (invalid config, etc.) */
        CHECK_CUDA(cudaGetLastError());
    }, warmup_runs, benchmark_runs);

    CHECK_CUDA(cudaMemcpy(h_C_naive.data(), d_C, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float naive_gflops = compute_gflops(M, N, K, naive_cuda_time);
    printf("  Average time: %.3f ms  |  %.1f GFLOPS  (%.2f TFLOPS)\n\n",
           naive_cuda_time, naive_gflops, naive_gflops / 1000.0f);

    /*
     * ===========================================
     * Results Summary Table
     * ===========================================
     */
    printf("==========================================================\n");
    printf("                    PERFORMANCE SUMMARY\n");
    printf("==========================================================\n");
    printf("%-24s %10s %10s %10s\n", "Method", "Time (ms)", "GFLOPS", "Speedup");
    printf("----------------------------------------------------------\n");
    printf("%-24s %10.3f %10.1f %10.1fx\n",
           "cuBLAS FP32",  cublas_fp32_time,  cublas_fp32_gflops,
           naive_cuda_time / cublas_fp32_time);
    printf("%-24s %10.3f %10.1f %10.1fx\n",
           "cuBLASLt FP32", cublasLt_fp32_time, cublasLt_fp32_gflops,
           naive_cuda_time / cublasLt_fp32_time);
    printf("%-24s %10.3f %10.1f %10.1fx\n",
           "cuBLAS FP16",  cublas_fp16_time,  cublas_fp16_gflops,
           naive_cuda_time / cublas_fp16_time);
    printf("%-24s %10.3f %10.1f %10.1fx\n",
           "cuBLASLt FP16", cublasLt_fp16_time, cublasLt_fp16_gflops,
           naive_cuda_time / cublasLt_fp16_time);
    printf("%-24s %10.3f %10.1f %10s\n",
           "Naive Kernel FP32", naive_cuda_time, naive_gflops, "1.0x");
    printf("==========================================================\n\n");

    /*
     * ===========================================
     * Verification
     * ===========================================
     *
     * Compare all results against the naive kernel (our reference).
     * The naive kernel is simple enough to trust as correct.
     *
     * Tolerance values:
     *   FP32 (1e-2): Different accumulation orders cause small rounding
     *                 differences. For K=1024, errors up to ~0.01 are normal.
     *   FP16 (5e-1): FP16 has only 10 mantissa bits (vs 23 for FP32).
     *                 For large dot products, errors up to ~0.5 are expected.
     */
    printf("Verification (against naive kernel reference):\n");

    bool cublas_fp32_ok   = verifyResults(h_C_naive, h_C_cublas_fp32,   1e-2f);
    bool cublasLt_fp32_ok = verifyResults(h_C_naive, h_C_cublasLt_fp32, 1e-2f);
    bool cublas_fp16_ok   = verifyResults(h_C_naive, h_C_cublas_fp16,   5e-1f);
    bool cublasLt_fp16_ok = verifyResults(h_C_naive, h_C_cublasLt_fp16, 5e-1f);

    printf("  cuBLAS FP32:  %s (tolerance 1e-2)\n",
           cublas_fp32_ok ? "MATCH" : "MISMATCH");
    printf("  cuBLASLt FP32: %s (tolerance 1e-2)\n",
           cublasLt_fp32_ok ? "MATCH" : "MISMATCH");
    printf("  cuBLAS FP16:  %s (tolerance 5e-1)\n",
           cublas_fp16_ok ? "MATCH" : "MISMATCH");
    printf("  cuBLASLt FP16: %s (tolerance 5e-1)\n",
           cublasLt_fp16_ok ? "MATCH" : "MISMATCH");

    /*
     * Compute max absolute error for FP16 results.
     * This shows the worst-case precision loss from using half precision.
     */
    float max_err_cublas_fp16 = 0.0f;
    float max_err_cublasLt_fp16 = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float err = fabsf(h_C_naive[i] - h_C_cublas_fp16[i]);
        if (err > max_err_cublas_fp16) max_err_cublas_fp16 = err;
    }
    for (int i = 0; i < M * N; ++i) {
        float err = fabsf(h_C_naive[i] - h_C_cublasLt_fp16[i]);
        if (err > max_err_cublasLt_fp16) max_err_cublasLt_fp16 = err;
    }
    printf("\n  Max absolute error (FP16 vs naive FP32):\n");
    printf("    cuBLAS FP16:  %.6f\n", max_err_cublas_fp16);
    printf("    cuBLASLt FP16: %.6f\n", max_err_cublasLt_fp16);

    /*
     * ===========================================
     * Cleanup (Reverse Order of Creation)
     * ===========================================
     */
    printf("\nCleaning up...\n");

    /* Destroy FP32 cuBLASLt descriptors */
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));

    /* Destroy FP16 cuBLASLt descriptors */
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc_half));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc_half));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc_half));

    /* Destroy handles */
    CHECK_CUBLAS(cublasLtDestroy(cublasLt_handle));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    /* Free device memory */
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_half));
    CHECK_CUDA(cudaFree(d_B_half));
    CHECK_CUDA(cudaFree(d_C_half));

    printf("Done!\n");
    return 0;
}

/*
 * ===========================================
 * Expected Output (RTX 4070, times will vary)
 * ===========================================
 *
 * ==========================================================
 *  cuBLAS vs cuBLASLt vs Naive Kernel - Performance Benchmark
 * ==========================================================
 *
 * Matrix dimensions: A (4096x1024) * B (1024x4096) = C (4096x4096)
 * Total FLOPs per matmul: 34.36 GFLOP
 *
 * Benchmark config: 3 warmup runs, 20 timed runs (averaged)
 *
 * Benchmarking cuBLAS SGEMM (FP32)...
 *   Average time: 1.xxx ms  |  xxxxx.x GFLOPS  (xx.xx TFLOPS)
 *
 * Benchmarking cuBLASLt Matmul (FP32)...
 *   Average time: 1.xxx ms  |  xxxxx.x GFLOPS  (xx.xx TFLOPS)
 *
 * Benchmarking cuBLAS HGEMM (FP16)...
 *   Average time: 0.xxx ms  |  xxxxx.x GFLOPS  (xx.xx TFLOPS)
 *
 * Benchmarking cuBLASLt Matmul (FP16)...
 *   Average time: 0.xxx ms  |  xxxxx.x GFLOPS  (xx.xx TFLOPS)
 *
 * Benchmarking Naive CUDA Kernel (FP32)...
 *   Average time: xx.xxx ms  |  xxxx.x GFLOPS  (x.xx TFLOPS)
 *
 * ==========================================================
 *                     PERFORMANCE SUMMARY
 * ==========================================================
 * Method                    Time (ms)     GFLOPS    Speedup
 * ----------------------------------------------------------
 * cuBLAS FP32                   1.xxx    xxxxx.x      xx.x
 * cuBLASLt FP32                 1.xxx    xxxxx.x      xx.x
 * cuBLAS FP16                   0.xxx    xxxxx.x      xx.x
 * cuBLASLt FP16                 0.xxx    xxxxx.x      xx.x
 * Naive Kernel FP32            xx.xxx     xxxx.x       1.0x
 * ==========================================================
 *
 * Verification (against naive kernel reference):
 *   cuBLAS FP32:  MATCH (tolerance 1e-2)
 *   cuBLASLt FP32: MATCH (tolerance 1e-2)
 *   cuBLAS FP16:  MATCH (tolerance 5e-1)
 *   cuBLASLt FP16: MATCH (tolerance 5e-1)
 *
 *   Max absolute error (FP16 vs naive FP32):
 *     cuBLAS FP16:  x.xxxxxx
 *     cuBLASLt FP16: x.xxxxxx
 *
 * ===========================================
 * Key Takeaways
 * ===========================================
 *
 * 1. LIBRARY vs HANDWRITTEN:
 *    cuBLAS/cuBLASLt are 10-50x faster than a naive kernel.
 *    They use shared memory tiling, Tensor Cores, optimal memory
 *    access patterns, and architecture-specific optimizations.
 *
 * 2. cuBLAS vs cuBLASLt:
 *    Near-identical performance for basic GEMM. cuBLASLt shines when
 *    you need epilogue fusions (GEMM+bias+ReLU), explicit algorithm
 *    selection, or mixed-precision (FP16 data, FP32 accumulation).
 *
 * 3. FP32 vs FP16:
 *    FP16 is ~2x faster due to halved memory bandwidth and Tensor Core
 *    acceleration. The trade-off is lower precision (~3 vs ~7 digits).
 *    For deep learning inference, FP16 is almost always sufficient.
 *
 * 4. GFLOPS AS A METRIC:
 *    GFLOPS = 2*M*N*K / (time * 10^9). Compare against your GPU's
 *    theoretical peak to see how efficient your implementation is.
 *    cuBLAS typically achieves 80-95% of peak throughput.
 *
 * 5. BENCHMARKING BEST PRACTICES:
 *    - Always use warmup runs (GPU clock ramp, JIT, cache warming)
 *    - Average multiple runs (reduce variance from OS scheduling)
 *    - Use CUDA events (not CPU timers) for GPU timing
 *    - Use large matrices (amortize kernel launch overhead)
 */
