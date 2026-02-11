/*
 * cuBLAS vs cuBLASXt - Performance Comparison
 * =============================================
 *
 * This program benchmarks cuBLAS (single GPU, device memory) against
 * cuBLASXt (multi-GPU capable, host memory) on a LARGE matrix multiplication
 * to see how they compare in real-world performance.
 *
 * ===========================================
 * What Are We Comparing?
 * ===========================================
 *
 * cuBLAS:
 *   - Data lives on GPU (you manage cudaMalloc/cudaMemcpy)
 *   - Compute only is timed (data already on device)
 *   - cublasSgemm is ASYNCHRONOUS → must cudaDeviceSynchronize()
 *
 * cuBLASXt:
 *   - Data lives on HOST (CPU memory)
 *   - Each call copies data to GPU, computes, copies back
 *   - cublasXtSgemm is SYNCHRONOUS → blocks CPU until done
 *   - Total time = transfer + compute + transfer back
 *
 * ===========================================
 * Fair Comparison?
 * ===========================================
 *
 * This benchmark is INTENTIONALLY unfair to cuBLASXt because:
 *
 *   cuBLAS timing:   [----compute----]
 *   cuBLASXt timing: [--copy→GPU--][--compute--][--copy→CPU--]
 *
 * cuBLAS has data pre-loaded on GPU before timing starts.
 * cuBLASXt must transfer data every call.
 *
 * This shows the REAL overhead of cuBLASXt's convenience.
 * In practice, if you're doing repeated matmuls on the same data,
 * cuBLAS wins because data stays on GPU. cuBLASXt is better when
 * you have multiple GPUs and very large matrices.
 *
 * ===========================================
 * Synchronous vs Asynchronous
 * ===========================================
 *
 * cublasSgemm (cuBLAS):
 *   - Returns IMMEDIATELY (GPU works in background)
 *   - Must call cudaDeviceSynchronize() before reading results
 *   - Must sync BEFORE stopping the timer!
 *
 * cublasXtSgemm (cuBLASXt):
 *   - BLOCKS until results are written to host memory
 *   - No sync needed (the call itself waits)
 *   - CPU timer start/stop around the call is sufficient
 *
 * ===========================================
 * Why std::chrono Instead of CUDA Events?
 * ===========================================
 *
 * CUDA events only work for GPU-side timing. cuBLASXt manages its
 * own GPU internally, so we can't insert CUDA events into its pipeline.
 * For a fair apples-to-apples comparison, we use CPU timing (std::chrono)
 * for BOTH libraries. The cuBLAS section includes cudaDeviceSynchronize()
 * to ensure the GPU is done before stopping the CPU timer.
 *
 * ===========================================
 * Memory Requirements
 * ===========================================
 *
 * For M=N=K=16384:
 *   Each matrix:  16384 * 16384 * 4 bytes = 1 GB
 *   Host memory:  4 matrices * 1 GB = 4 GB (A, B, C_cublas, C_cublasxt)
 *   Device memory: 3 matrices * 1 GB = 3 GB (d_A, d_B, d_C)
 *
 * Make sure your GPU has at least 3+ GB VRAM and system has 4+ GB free RAM.
 * RTX 4070 (12 GB VRAM) handles this easily.
 *
 * Compile: nvcc 05_compare_cublas_cublasxt.cu -o 05_compare_cublas_cublasxt.exe -lcublas
 * Run:     .\05_compare_cublas_cublasxt.exe
 *
 * Note: This takes a while (~30-60 seconds) due to the large matrix size.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>          /* cuBLAS (cublasSgemm)            */
#include <cublasXt.h>           /* cuBLASXt (cublasXtSgemm)        */

/*
 * C++ headers used for specific features:
 *   <chrono> - High-resolution CPU timing (needed because cuBLASXt
 *              manages GPU internally, so CUDA events can't be used)
 *   <vector> - Dynamic array for storing per-run timings
 */
#include <chrono>
#include <vector>

/*
 * ===========================================
 * Error Checking Macros
 * ===========================================
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
 * Initialize Matrix with Random Values
 * ===========================================
 *
 * Random values in [0, 1). srand() must be called before this.
 */
void initMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

/*
 * ===========================================
 * Compare Results (Relative Error)
 * ===========================================
 *
 * Uses RELATIVE error: |a - b| / max(|a|, |b|)
 *
 * Relative error is better than absolute error for large values.
 * For example, if a=1000.0 and b=1000.1, absolute error is 0.1
 * (seems large) but relative error is 0.0001 (actually very close).
 *
 * Special case: if both values are 0, skip the check (0 == 0).
 * This avoids division by zero when max_val = 0.
 */
bool compareResults(const float *result1, const float *result2,
                    int size, float tolerance) {
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(result1[i] - result2[i]);
        float max_val = fmaxf(fabsf(result1[i]), fabsf(result2[i]));

        /* Skip if both values are zero (avoids division by zero) */
        if (max_val == 0.0f) continue;

        if (diff / max_val > tolerance) {
            fprintf(stderr, "  Mismatch at index %d:\n", i);
            fprintf(stderr, "    cuBLAS: %.6f, cuBLASXt: %.6f\n",
                    result1[i], result2[i]);
            fprintf(stderr, "    Relative error: %.6e\n", diff / max_val);
            return false;
        }
    }
    return true;
}

/*
 * ===========================================
 * Compute GFLOPS
 * ===========================================
 *
 * GFLOPS = (2 * M * N * K) / (time_seconds * 10^9)
 *
 * We use 'double' for M*N*K because with 16384^3 the intermediate
 * value (2 * 16384 * 16384 * 16384) = 8.8 * 10^12, which overflows
 * a 32-bit int (max ~2.1 * 10^9).
 */
double compute_gflops(int m, int n, int k, double time_seconds) {
    double flops = 2.0 * m * n * k;
    return flops / (time_seconds * 1e9);
}

int main() {
    /*
     * ===========================================
     * Matrix Dimensions
     * ===========================================
     *
     * Using 16384x16384 (16K x 16K) matrices to see meaningful
     * differences between cuBLAS and cuBLASXt. At this size:
     *   - Each matrix is 1 GB
     *   - Total FLOPs per matmul = 2 * 16384^3 = 8.8 TFLOP
     *   - Large enough that compute dominates transfer overhead
     */
    int M = 16384;
    int N = 16384;
    int K = 16384;

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    printf("===========================================================\n");
    printf(" cuBLAS vs cuBLASXt Performance Comparison\n");
    printf("===========================================================\n\n");
    printf("Matrix dimensions: A (%dx%d) * B (%dx%d) = C (%dx%d)\n", M, K, K, N, M, N);
    printf("Each matrix size: %.1f GB\n", (double)size_A / (1024.0 * 1024.0 * 1024.0));
    printf("Total FLOPs per matmul: %.2f TFLOP\n\n",
           (2.0 * M * N * K) / 1e12);

    /*
     * ===========================================
     * Host Memory Allocation
     * ===========================================
     *
     * 4 matrices * 1 GB each = 4 GB total host memory.
     *
     * h_C_cublas   - cuBLAS result (copied from device after compute)
     * h_C_cublasxt - cuBLASXt result (written directly by cuBLASXt)
     */
    float *h_A          = (float *)malloc(size_A);
    float *h_B          = (float *)malloc(size_B);
    float *h_C_cublas   = (float *)malloc(size_C);
    float *h_C_cublasxt = (float *)malloc(size_C);

    if (!h_A || !h_B || !h_C_cublas || !h_C_cublasxt) {
        fprintf(stderr, "Failed to allocate host memory (need ~4 GB)\n");
        return EXIT_FAILURE;
    }

    srand((unsigned int)time(NULL));
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    const int num_runs = 5;
    std::vector<double> cublas_times;
    std::vector<double> cublasxt_times;

    /*
     * ===========================================
     * Benchmark 1: cuBLAS (Device Memory)
     * ===========================================
     *
     * Steps:
     *   1. Create handle + allocate device memory
     *   2. Copy data host → device (NOT timed)
     *   3. Warmup run (not timed)
     *   4. Timed runs: cublasSgemm + cudaDeviceSynchronize
     *   5. Copy result device → host
     *   6. Cleanup
     *
     * The cudaDeviceSynchronize() inside the timing loop is CRITICAL.
     * Without it, we'd only measure the time to LAUNCH the kernel
     * (microseconds), not the time to COMPLETE it (milliseconds).
     *
     * cublasSgemm is asynchronous: it returns immediately and the GPU
     * continues working in the background. Sync forces the CPU to wait.
     */
    printf("--- cuBLAS (device memory) ---\n");
    {
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, size_A));
        CHECK_CUDA(cudaMalloc(&d_B, size_B));
        CHECK_CUDA(cudaMalloc(&d_C, size_C));

        CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
        printf("Data copied to device (%.1f GB)\n",
               (double)(size_A + size_B) / (1024.0 * 1024.0 * 1024.0));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        /* Warmup: GPU clock ramp + JIT compile */
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_B, N, d_A, K,
                                 &beta, d_C, N));
        CHECK_CUDA(cudaDeviceSynchronize());

        /* Timed benchmark runs */
        for (int i = 0; i < num_runs; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha, d_B, N, d_A, K,
                                     &beta, d_C, N));
            CHECK_CUDA(cudaDeviceSynchronize());  /* Wait for GPU! */

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            cublas_times.push_back(diff.count());

            printf("  Run %d: %.4f sec  (%.1f GFLOPS)\n",
                   i + 1, diff.count(),
                   compute_gflops(M, N, K, diff.count()));
        }

        CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C, size_C, cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUBLAS(cublasDestroy(handle));
    }

    printf("\n");

    /*
     * ===========================================
     * Benchmark 2: cuBLASXt (Host Memory)
     * ===========================================
     *
     * Steps:
     *   1. Create handle + select device(s)
     *   2. Warmup run (not timed)
     *   3. Timed runs: cublasXtSgemm (includes internal transfers)
     *   4. Cleanup
     *
     * NO cudaDeviceSynchronize needed! cublasXtSgemm is synchronous:
     * it blocks the CPU until the computation is done and results are
     * written back to host memory.
     *
     * The timed portion includes:
     *   - Host → Device data transfer (internally managed)
     *   - GPU GEMM computation
     *   - Device → Host result transfer (internally managed)
     */
    printf("--- cuBLASXt (host memory) ---\n");
    {
        cublasXtHandle_t handle;
        CHECK_CUBLAS(cublasXtCreate(&handle));

        int devices[1] = {0};
        CHECK_CUBLAS(cublasXtDeviceSelect(handle, 1, devices));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        /* Warmup */
        CHECK_CUBLAS(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    N, M, K, &alpha, h_B, N, h_A, K,
                                    &beta, h_C_cublasxt, N));

        /* Timed benchmark runs */
        for (int i = 0; i < num_runs; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            CHECK_CUBLAS(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        N, M, K, &alpha, h_B, N, h_A, K,
                                        &beta, h_C_cublasxt, N));
            /* No sync needed - cublasXtSgemm blocks until done */

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            cublasxt_times.push_back(diff.count());

            printf("  Run %d: %.4f sec  (%.1f GFLOPS)\n",
                   i + 1, diff.count(),
                   compute_gflops(M, N, K, diff.count()));
        }

        CHECK_CUBLAS(cublasXtDestroy(handle));
    }

    /*
     * ===========================================
     * Results Summary
     * ===========================================
     */
    double avg_cublas = 0.0, avg_cublasxt = 0.0;
    for (int i = 0; i < num_runs; ++i) {
        avg_cublas += cublas_times[i];
        avg_cublasxt += cublasxt_times[i];
    }
    avg_cublas /= num_runs;
    avg_cublasxt /= num_runs;

    double cublas_gflops   = compute_gflops(M, N, K, avg_cublas);
    double cublasxt_gflops = compute_gflops(M, N, K, avg_cublasxt);

    printf("\n===========================================================\n");
    printf("                     RESULTS SUMMARY\n");
    printf("===========================================================\n");
    printf("%-20s %12s %12s %10s\n", "Library", "Avg Time(s)", "GFLOPS", "TFLOPS");
    printf("-----------------------------------------------------------\n");
    printf("%-20s %12.4f %12.1f %10.2f\n",
           "cuBLAS", avg_cublas, cublas_gflops, cublas_gflops / 1000.0);
    printf("%-20s %12.4f %12.1f %10.2f\n",
           "cuBLASXt", avg_cublasxt, cublasxt_gflops, cublasxt_gflops / 1000.0);
    printf("-----------------------------------------------------------\n");

    /* Show which is faster and by how much */
    if (avg_cublas < avg_cublasxt) {
        printf("cuBLAS is %.2fx faster (compute only, data pre-loaded)\n",
               avg_cublasxt / avg_cublas);
    } else {
        printf("cuBLASXt is %.2fx faster\n",
               avg_cublas / avg_cublasxt);
    }
    printf("===========================================================\n\n");

    /*
     * ===========================================
     * Verification
     * ===========================================
     *
     * Compare cuBLAS and cuBLASXt results against each other.
     * They should match closely since they perform the same computation.
     * Using relative error tolerance of 1e-4 (0.01%).
     */
    printf("Verification:\n");
    float tolerance = 1e-4f;
    bool match = compareResults(h_C_cublas, h_C_cublasxt, M * N, tolerance);
    printf("  Results %s (relative tolerance: %.0e)\n",
           match ? "MATCH" : "DO NOT MATCH", tolerance);

    /* Cleanup host memory */
    free(h_A);
    free(h_B);
    free(h_C_cublas);
    free(h_C_cublasxt);

    printf("\nDone!\n");
    return 0;
}

/*
 * ===========================================
 * Expected Output (RTX 4070, 16384x16384)
 * ===========================================
 *
 * ===========================================================
 *  cuBLAS vs cuBLASXt Performance Comparison
 * ===========================================================
 *
 * Matrix dimensions: A (16384x16384) * B (16384x16384) = C (16384x16384)
 * Each matrix size: 1.0 GB
 * Total FLOPs per matmul: 8.80 TFLOP
 *
 * --- cuBLAS (device memory) ---
 * Data copied to device (2.0 GB)
 *   Run 1: x.xxxx sec  (xxxxx.x GFLOPS)
 *   Run 2: x.xxxx sec  (xxxxx.x GFLOPS)
 *   ...
 *
 * --- cuBLASXt (host memory) ---
 *   Run 1: x.xxxx sec  (xxxxx.x GFLOPS)
 *   Run 2: x.xxxx sec  (xxxxx.x GFLOPS)
 *   ...
 *
 * ===========================================================
 *                      RESULTS SUMMARY
 * ===========================================================
 * Library                Avg Time(s)       GFLOPS     TFLOPS
 * -----------------------------------------------------------
 * cuBLAS                     x.xxxx       xxxxx.x      xx.xx
 * cuBLASXt                   x.xxxx       xxxxx.x      xx.xx
 * -----------------------------------------------------------
 * cuBLAS is x.xxx faster (compute only, data pre-loaded)
 * ===========================================================
 *
 * Verification:
 *   Results MATCH (relative tolerance: 1e-04)
 *
 * ===========================================
 * Key Takeaways
 * ===========================================
 *
 * 1. cuBLAS IS FASTER (for single GPU):
 *    cuBLAS only does compute. cuBLASXt also transfers data.
 *    For repeated operations on the same data, cuBLAS wins.
 *
 * 2. cuBLASXt IS SIMPLER:
 *    No cudaMalloc, no cudaMemcpy. Just pass host pointers.
 *    Good for one-shot operations or prototyping.
 *
 * 3. FAIR COMPARISON WOULD INCLUDE TRANSFERS:
 *    If you add cudaMemcpy time to cuBLAS, the gap narrows.
 *    cuBLASXt's overhead is mostly the data transfer.
 *
 * 4. cuBLASXt SHINES WITH MULTI-GPU:
 *    On a system with 2+ GPUs, cuBLASXt can split the work.
 *    For matrices too large for one GPU's VRAM, cuBLASXt can
 *    tile and distribute automatically.
 *
 * 5. SYNC MATTERS FOR TIMING:
 *    cublasSgemm is async → must sync before stopping timer.
 *    cublasXtSgemm is sync → no sync needed.
 *    Forgetting sync makes cuBLAS look ~1000x faster (wrong!).
 */
