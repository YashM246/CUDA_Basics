/*
 * cuBLASXt Matrix Multiplication - Multi-GPU GEMM
 * =================================================
 *
 * This program demonstrates cuBLASXt (cuBLAS eXtended) for matrix
 * multiplication across one or more GPUs.
 *
 * ===========================================
 * What is cuBLASXt?
 * ===========================================
 *
 * cuBLASXt is a multi-GPU extension of cuBLAS that:
 *   1. Works on HOST (CPU) memory - no cudaMalloc/cudaMemcpy needed!
 *   2. Automatically splits matrices across multiple GPUs
 *   3. Handles all device memory management internally
 *   4. Tiles the matrices into blocks and distributes work
 *
 * This is the KEY difference from cuBLAS and cuBLASLt:
 *
 *   cuBLAS/cuBLASLt:
 *     You allocate device memory (cudaMalloc)
 *     You copy data to device (cudaMemcpy)
 *     You call the GEMM function (uses device pointers)
 *     You copy results back (cudaMemcpy)
 *
 *   cuBLASXt:
 *     You pass HOST pointers directly!
 *     cuBLASXt handles ALL the GPU memory management internally.
 *     It copies data to GPU(s), computes, and copies results back for you.
 *
 * ===========================================
 * When to Use cuBLASXt?
 * ===========================================
 *
 * cuBLASXt is best for:
 *   - VERY large matrices that benefit from multi-GPU splitting
 *   - Simple code where you don't want to manage device memory
 *   - Multi-GPU systems where you want automatic load balancing
 *
 * cuBLASXt is NOT ideal for:
 *   - Small matrices (overhead of data transfer dominates)
 *   - Single-GPU workloads (cuBLAS/cuBLASLt are faster with less overhead)
 *   - Repeated operations on same data (cuBLAS keeps data on GPU)
 *   - Pipeline/streaming workloads (cuBLASLt with CUDA streams is better)
 *
 * In practice, cuBLASXt is less commonly used than cuBLAS/cuBLASLt.
 * Most deep learning frameworks use cuBLASLt directly.
 *
 * ===========================================
 * cuBLASXt Workflow
 * ===========================================
 *
 * 1. cublasXtCreate(&handle)              - Create cuBLASXt handle
 * 2. cublasXtDeviceSelect(handle, n, devs) - Select which GPU(s) to use
 * 3. cublasXtSgemm(handle, ..., HOST_ptrs) - Run GEMM with HOST pointers
 * 4. cublasXtDestroy(handle)              - Cleanup
 *
 * Optionally:
 *   cublasXtSetBlockDim(handle, blockDim)  - Set tile size for GPU splitting
 *
 * ===========================================
 * How cuBLASXt Splits Work (Block Tiling)
 * ===========================================
 *
 * For multi-GPU, cuBLASXt divides matrices into tiles (blocks):
 *
 *   Matrix A (M x K):           Matrix B (K x N):
 *   +-------+-------+           +-------+-------+
 *   | A_00  | A_01  |           | B_00  | B_01  |
 *   +-------+-------+           +-------+-------+
 *   | A_10  | A_11  |           | B_10  | B_11  |
 *   +-------+-------+           +-------+-------+
 *
 *   GPU 0 computes some tiles of C, GPU 1 computes others.
 *   cuBLASXt handles the scheduling automatically.
 *
 * The default block size is usually fine. You can tune it with:
 *   cublasXtSetBlockDim(handle, blockDim)
 *   - Smaller blocks → more parallelism, more overhead
 *   - Larger blocks → less overhead, may underutilize GPUs
 *   - Default is usually 2048-4096
 *
 * ===========================================
 * cuBLASXt vs cuBLAS vs cuBLASLt
 * ===========================================
 *
 * | Feature          | cuBLAS     | cuBLASLt    | cuBLASXt     |
 * |------------------|------------|-------------|--------------|
 * | Memory           | Device     | Device      | HOST         |
 * | Multi-GPU        | No         | No          | Yes          |
 * | Memory mgmt      | Manual     | Manual      | Automatic    |
 * | API style        | Simple     | Descriptors | Simple       |
 * | Overhead         | Low        | Low         | Higher       |
 * | Best for         | Single GPU | Tuning      | Multi-GPU    |
 *
 * ===========================================
 * Row-Major Swap Trick (Same as cuBLAS)
 * ===========================================
 *
 * cublasXtSgemm has the SAME interface as cublasSgemm, so the
 * same row-major swap trick applies:
 *   - Swap A and B, swap m and n
 *   - See 01_sgemm_hgemm_cublas.cu for the full explanation
 *
 * Compile: nvcc 04_matmul_cublasxt.cu -o 04_matmul_cublasxt.exe -lcublas
 * Run:     .\04_matmul_cublasxt.exe
 *
 * Note: cuBLASXt is part of the cuBLAS library, so -lcublas is sufficient.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublasXt.h>           /* cuBLASXt API (cublasXtSgemm, etc.)  */

/*
 * ===========================================
 * Error Checking Macro
 * ===========================================
 *
 * cuBLASXt uses the same cublasStatus_t as cuBLAS.
 */
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
 * Using 256x256 matrices for a quick demo. In real usage, cuBLASXt
 * shines with MUCH larger matrices (4096+ per dimension) where
 * the multi-GPU tiling overhead is amortized.
 *
 * For small matrices like these, cuBLASXt has more overhead than
 * plain cuBLAS because it manages host↔device transfers internally.
 */
const int M = 256;
const int N = 256;
const int K = 256;

int main() {
    printf("=================================================\n");
    printf(" cuBLASXt Matrix Multiplication Demo\n");
    printf("=================================================\n\n");
    printf("Matrix dimensions: A (%dx%d) * B (%dx%d) = C (%dx%d)\n\n", M, K, K, N, M, N);

    /*
     * ===========================================
     * Host Memory Allocation
     * ===========================================
     *
     * cuBLASXt works with HOST pointers directly!
     * No cudaMalloc needed. This is the main convenience of cuBLASXt.
     *
     * We allocate:
     *   A_host     - Input matrix A (M x K)
     *   B_host     - Input matrix B (K x N)
     *   C_host_cpu - CPU reference result
     *   C_host_gpu - cuBLASXt result (written directly to host memory!)
     */
    float *A_host     = (float *)malloc(M * K * sizeof(float));
    float *B_host     = (float *)malloc(K * N * sizeof(float));
    float *C_host_cpu = (float *)malloc(M * N * sizeof(float));
    float *C_host_gpu = (float *)malloc(M * N * sizeof(float));

    if (!A_host || !B_host || !C_host_cpu || !C_host_gpu) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /*
     * Initialize matrices with random values in [0, 1).
     * srand(time(0)) seeds with current time for different values each run.
     */
    srand((unsigned int)time(NULL));
    for (int i = 0; i < M * K; ++i) A_host[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B_host[i] = (float)rand() / RAND_MAX;

    /*
     * ===========================================
     * CPU Reference Multiplication
     * ===========================================
     *
     * Standard triple-loop matmul in row-major order.
     * Used to verify cuBLASXt results.
     */
    printf("Running CPU matrix multiply...\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A_host[i * K + k] * B_host[k * N + j];
            }
            C_host_cpu[i * N + j] = sum;
        }
    }

    /*
     * ===========================================
     * Step 1: Create cuBLASXt Handle
     * ===========================================
     *
     * cublasXtHandle_t is DIFFERENT from cublasHandle_t:
     *   cublasHandle_t   → single GPU, device memory
     *   cublasXtHandle_t → multi-GPU, host memory
     *
     * You CANNOT pass a cublasXtHandle_t to cublasSgemm (or vice versa).
     */
    cublasXtHandle_t handle;
    CHECK_CUBLAS(cublasXtCreate(&handle));
    printf("cuBLASXt handle created\n");

    /*
     * ===========================================
     * Step 2: Select GPU Device(s)
     * ===========================================
     *
     * cublasXtDeviceSelect(handle, numDevices, deviceArray)
     *
     * Parameters:
     *   handle     - cuBLASXt handle
     *   numDevices - How many GPUs to use
     *   deviceArray - Array of CUDA device IDs
     *
     * For single GPU: devices = {0}, numDevices = 1
     * For multi-GPU:  devices = {0, 1}, numDevices = 2
     *
     * This MUST be called before any computation.
     * cuBLASXt will distribute work across all selected GPUs.
     *
     * To find available GPUs:
     *   int deviceCount;
     *   cudaGetDeviceCount(&deviceCount);
     */
    int devices[1] = {0};       /* Use GPU 0 only (single GPU system) */
    CHECK_CUBLAS(cublasXtDeviceSelect(handle, 1, devices));
    printf("Selected %d GPU(s) for computation\n", 1);

    /*
     * ===========================================
     * Step 3: Execute Matrix Multiplication
     * ===========================================
     *
     * cublasXtSgemm has the SAME signature as cublasSgemm:
     *
     * cublasXtSgemm(handle, transa, transb, m, n, k,
     *               &alpha, A, lda, B, ldb, &beta, C, ldc)
     *
     * The ONLY difference: A, B, C are HOST pointers (not device pointers).
     * cuBLASXt internally:
     *   1. Allocates device memory on selected GPU(s)
     *   2. Copies input data from host to device(s)
     *   3. Runs the GEMM computation
     *   4. Copies results back to host
     *   5. Frees device memory
     *
     * This is why cuBLASXt is slower for repeated operations:
     * it copies data EVERY call. With cuBLAS, you keep data on GPU.
     *
     * The row-major swap trick is the same:
     *   - Pass B_host as cuBLAS's A, A_host as cuBLAS's B
     *   - m=N, n=M, k=K
     */
    printf("\nRunning cuBLASXt SGEMM...\n");

    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUBLAS(cublasXtSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,                    /* m, n, k (swapped for row-major) */
        &alpha,
        B_host, N,                  /* cuBLAS's A = our B, lda = N     */
        A_host, K,                  /* cuBLAS's B = our A, ldb = K     */
        &beta,
        C_host_gpu, N               /* C output (HOST pointer!), ldc=N */
    ));

    /*
     * ===========================================
     * Verification
     * ===========================================
     *
     * Compare cuBLASXt result against CPU reference.
     * Track the maximum absolute error across all elements.
     *
     * Expected: very small errors (< 1e-4) due to different
     * accumulation order between CPU sequential and GPU parallel.
     */
    printf("\nVerification:\n");

    float max_error = 0.0f;
    int mismatch_count = 0;
    const float tolerance = 1e-4f;

    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(C_host_cpu[i] - C_host_gpu[i]);
        if (diff > max_error) {
            max_error = diff;
        }
        if (diff > tolerance) {
            mismatch_count++;
            /* Print first few mismatches for debugging */
            if (mismatch_count <= 5) {
                fprintf(stderr, "  Mismatch at index %d: CPU=%.6f, GPU=%.6f, "
                        "diff=%.6f\n", i, C_host_cpu[i], C_host_gpu[i], diff);
            }
        }
    }

    printf("  Max absolute error: %.8f\n", max_error);
    printf("  Elements exceeding tolerance (%.0e): %d / %d\n",
           tolerance, mismatch_count, M * N);
    printf("  Result: %s\n", (mismatch_count == 0) ? "PASS" : "FAIL");

    /*
     * ===========================================
     * Cleanup
     * ===========================================
     *
     * Destroy handle, then free host memory.
     */
    printf("\nCleaning up...\n");

    CHECK_CUBLAS(cublasXtDestroy(handle));

    free(A_host);
    free(B_host);
    free(C_host_cpu);
    free(C_host_gpu);

    printf("Done!\n");
    return 0;
}

/*
 * ===========================================
 * Expected Output
 * ===========================================
 *
 * =================================================
 *  cuBLASXt Matrix Multiplication Demo
 * =================================================
 *
 * Matrix dimensions: A (256x256) * B (256x256) = C (256x256)
 *
 * Running CPU matrix multiply...
 * cuBLASXt handle created
 * Selected 1 GPU(s) for computation
 *
 * Running cuBLASXt SGEMM...
 *
 * Verification:
 *   Max absolute error: 0.000xxxxx
 *   Elements exceeding tolerance (1e-04): 0 / 65536
 *   Result: PASS
 *
 * Cleaning up...
 * Done!
 *
 * ===========================================
 * Key Takeaways
 * ===========================================
 *
 * 1. HOST POINTERS:
 *    cuBLASXt is the ONLY cuBLAS variant that works with host memory.
 *    No cudaMalloc, no cudaMemcpy. Just pass your CPU arrays directly.
 *    cuBLASXt handles all GPU memory management internally.
 *
 * 2. MULTI-GPU:
 *    cublasXtDeviceSelect() lets you choose multiple GPUs.
 *    cuBLASXt automatically tiles and distributes the work.
 *    For single-GPU, cuBLAS/cuBLASLt are more efficient.
 *
 * 3. SAME API PATTERN:
 *    cublasXtSgemm has the same signature as cublasSgemm.
 *    The only difference is host vs device pointers.
 *    The same row-major swap trick applies.
 *
 * 4. OVERHEAD:
 *    cuBLASXt copies data to GPU every call, so it's slower
 *    than cuBLAS for repeated operations on the same data.
 *    Use cuBLAS/cuBLASLt if data stays on GPU between operations.
 *
 * 5. BLOCK TILING:
 *    cublasXtSetBlockDim() controls how matrices are split.
 *    Default is fine for most cases. Tune only for very large
 *    multi-GPU workloads.
 */
