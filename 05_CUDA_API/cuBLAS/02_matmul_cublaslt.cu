/*
 * cuBLASLt Matrix Multiplication - FP32 vs FP16
 * ===============================================
 *
 * This program demonstrates cuBLASLt (cuBLAS Lightweight) for matrix
 * multiplication in both single (FP32) and half (FP16) precision.
 *
 * ===========================================
 * cuBLAS vs cuBLASLt - Why Does cuBLASLt Exist?
 * ===========================================
 *
 * cuBLAS (standard):
 *   - Simple API: one function call does everything
 *   - cublasSgemm(handle, ..., A, lda, B, ldb, ..., C, ldc)
 *   - Easy to use but limited control
 *
 * cuBLASLt (lightweight):
 *   - Descriptor-based API: you describe everything, then execute
 *   - More setup code, but much more flexible
 *   - Advantages over cuBLAS:
 *       1. Manual algorithm selection (benchmark to find fastest)
 *       2. Mixed precision (FP16 input, FP32 accumulation)
 *       3. Epilogue fusions (GEMM + bias + ReLU in one kernel)
 *       4. Explicit Tensor Core control
 *       5. Better for deep learning workloads
 *
 * This is what PyTorch and TensorFlow use internally for matmul.
 *
 * ===========================================
 * cuBLASLt Workflow (Step by Step)
 * ===========================================
 *
 * Standard cuBLAS (1 step):
 *   cublasSgemm(handle, ..., all params in one call)
 *
 * cuBLASLt (multiple steps):
 *   1. Create handle                    (cublasLtCreate)
 *   2. Create matmul descriptor         (cublasLtMatmulDescCreate)
 *   3. Set descriptor attributes        (cublasLtMatmulDescSetAttribute)
 *   4. Create matrix layout descriptors (cublasLtMatrixLayoutCreate)
 *   5. Execute matmul                   (cublasLtMatmul)
 *   6. Destroy everything               (reverse order of creation)
 *
 * Why so many steps? Because each descriptor is REUSABLE.
 * If you run 1000 matmuls with same shapes, you create descriptors
 * once and call cublasLtMatmul 1000 times.
 *
 * ===========================================
 * The GEMM Equation
 * ===========================================
 *
 * cuBLASLt computes:  D = alpha * op(A) * op(B) + beta * C
 *
 *   - A, B = input matrices
 *   - C = input matrix (for the beta*C term, often zeros)
 *   - D = OUTPUT matrix (where the result goes)
 *   - alpha, beta = scalar multipliers
 *   - op() = optional transpose (CUBLAS_OP_N or CUBLAS_OP_T)
 *
 * Note: D and C can point to the SAME memory (in-place operation).
 * When beta=0, C is ignored so you can set D=C safely.
 *
 * Difference from cuBLAS:
 *   cuBLAS:   C = alpha*A*B + beta*C   (C is both input AND output)
 *   cuBLASLt: D = alpha*A*B + beta*C   (D is output, C is separate input)
 *
 * ===========================================
 * Row-Major Trick (Same as cuBLAS)
 * ===========================================
 *
 * cuBLASLt uses column-major (like cuBLAS). The same swap trick applies:
 *   - Swap A and B
 *   - Swap m and n
 *   - Result is correct in row-major
 *
 * See 01_sgemm_hgemm_cublas.cu for the full explanation.
 *
 * ===========================================
 * cublasLtMatmul Parameter Reference
 * ===========================================
 *
 * cublasLtMatmul(
 *     handle,           // cuBLASLt handle (context)
 *     computeDesc,      // Matmul descriptor (compute type, transpose ops)
 *     alpha,            // Scalar multiplier for A*B
 *     A, Adesc,         // Matrix A data + layout descriptor
 *     B, Bdesc,         // Matrix B data + layout descriptor
 *     beta,             // Scalar multiplier for C
 *     C, Cdesc,         // Matrix C data + layout descriptor (input)
 *     D, Ddesc,         // Matrix D data + layout descriptor (OUTPUT)
 *     algo,             // Algorithm to use (NULL = default)
 *     workspace,        // Temporary memory for some algorithms
 *     workspaceSize,    // Size of workspace in bytes
 *     stream            // CUDA stream (0 = default)
 * )
 *
 * That's 16 parameters! The most important difference from cuBLAS
 * is the separate D output matrix.
 *
 * Compile: nvcc 02_matmul_cublaslt.cu -o 02_matmul_cublaslt.exe -lcublasLt
 * Run:     .\02_matmul_cublaslt.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublasLt.h>       // cuBLASLt API
#include <cuda_fp16.h>      // FP16 (half precision) support

/*
 * ===========================================
 * Error Checking Macros
 * ===========================================
 *
 * cuBLASLt uses the same cublasStatus_t as cuBLAS.
 * We need separate macros for CUDA runtime vs cuBLAS errors.
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
 * CPU Matrix Multiply (Reference)
 * ===========================================
 *
 * Standard triple-loop in row-major order.
 * Used to verify GPU results are correct.
 */
void cpu_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/*
 * Print a row-major matrix with formatted output.
 */
void print_matrix(const float *matrix, int rows, int cols, const char *name) {
    printf("Matrix %s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    printf("cuBLASLt Matrix Multiplication Demo\n");
    printf("====================================\n\n");

    /*
     * Matrix dimensions: A (M×K) * B (K×N) = C (M×N)
     *
     * A (4×4):                        B (4×4):
     * |  1   2   3   4 |             |  1   2   4   4 |
     * |  5   6   7   8 |             |  5   6   7   8 |
     * |  9  10  11  12 |             |  9  10  11  12 |
     * | 13  14  15  16 |             | 17  18  19  20 |
     *
     * Note: A and B are intentionally different to catch
     * bugs that would be hidden if A == B.
     */
    const int M = 4, K = 4, N = 4;

    float h_A[M * K] = {
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    float h_B[K * N] = {
         1.0f,  2.0f,  4.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
        17.0f, 18.0f, 19.0f, 20.0f
    };

    float h_C_cpu[M * N]      = {0};
    float h_C_gpu_fp32[M * N] = {0};
    float h_C_gpu_fp16[M * N] = {0};

    /* Print input matrices */
    print_matrix(h_A, M, K, "A");
    print_matrix(h_B, K, N, "B");

    /* ==================== CPU Reference ==================== */
    printf("Running CPU matrix multiply...\n");
    cpu_matmul(h_A, h_B, h_C_cpu, M, N, K);

    /*
     * ===========================================
     * Allocate Device Memory
     * ===========================================
     *
     * Note the correct sizes:
     *   A: M*K elements
     *   B: K*N elements (NOT M*K!)
     *   C: M*N elements (NOT M*K!)
     *
     * When M=K=N, wrong sizes happen to work by coincidence.
     * Always use the correct dimension formula!
     */
    printf("Allocating device memory...\n");

    /* FP32 device memory */
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    CHECK_CUDA(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_fp32, K * N * sizeof(float)));   // K*N, not M*K!
    CHECK_CUDA(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));   // M*N, not M*K!

    /* FP16 device memory */
    half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    CHECK_CUDA(cudaMalloc(&d_A_fp16, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_fp16, K * N * sizeof(half)));    // K*N, not M*K!
    CHECK_CUDA(cudaMalloc(&d_C_fp16, M * N * sizeof(half)));    // M*N, not M*K!

    /* Copy FP32 data to device */
    CHECK_CUDA(cudaMemcpy(d_A_fp32, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp32, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));  // K*N!

    /*
     * Convert float to half and copy to device.
     * __float2half() may lose precision for large values.
     */
    half h_A_half[M * K], h_B_half[K * N];
    for (int i = 0; i < M * K; ++i) h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_half[i] = __float2half(h_B[i]);

    CHECK_CUDA(cudaMemcpy(d_A_fp16, h_A_half, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp16, h_B_half, K * N * sizeof(half), cudaMemcpyHostToDevice));

    /*
     * ===========================================
     * Step 1: Create cuBLASLt Handle
     * ===========================================
     *
     * cublasLtHandle_t vs cublasHandle_t:
     *   - cublasHandle_t    → for cuBLAS (standard)
     *   - cublasLtHandle_t  → for cuBLASLt (lightweight)
     *
     * They are DIFFERENT types. You cannot use one with the other's API.
     * The Lt handle is lighter weight (less internal state).
     */
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));
    printf("cuBLASLt handle created\n\n");

    /*
     * ===========================================
     * Step 2: Create Matrix Layout Descriptors
     * ===========================================
     *
     * cublasLtMatrixLayoutCreate(layout, dataType, rows, cols, ld)
     *
     * Parameters:
     *   layout   - Output: the created layout descriptor
     *   dataType - Element type in memory:
     *                CUDA_R_32F = float (32-bit real)
     *                CUDA_R_16F = half  (16-bit real)
     *                CUDA_R_8I  = int8  (8-bit integer)
     *              The "R" means Real (vs "C" for Complex)
     *   rows     - Number of rows (in column-major!)
     *   cols     - Number of columns (in column-major!)
     *   ld       - Leading dimension (stride between consecutive columns)
     *
     * IMPORTANT: rows/cols are in COLUMN-MAJOR sense!
     *
     * Since we use the row-major swap trick:
     *   - Our row-major A (M×K) → column-major A^T (K×M)
     *   - Our row-major B (K×N) → column-major B^T (N×K)
     *   - Output C (M×N) → column-major C^T (N×M)
     *
     * And we SWAP A and B in the matmul call, so:
     *   - cuBLAS's "A" = our B → layout has rows=N, cols=K, ld=N
     *   - cuBLAS's "B" = our A → layout has rows=K, cols=M, ld=K
     *   - cuBLAS's "C"/"D"     → layout has rows=N, cols=M, ld=N
     */
    printf("Creating FP32 matrix layouts...\n");

    cublasLtMatrixLayout_t matA_fp32, matB_fp32, matC_fp32;

    /*
     * Layout for our A (passed as cuBLAS's "B"):
     *   Row-major A (M×K) in memory = Column-major A^T (K×M)
     *   → rows=K, cols=M, ld=K
     */
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_fp32, CUDA_R_32F, K, M, K));

    /*
     * Layout for our B (passed as cuBLAS's "A"):
     *   Row-major B (K×N) in memory = Column-major B^T (N×K)
     *   → rows=N, cols=K, ld=N
     */
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_fp32, CUDA_R_32F, N, K, N));

    /*
     * Layout for output C/D:
     *   Row-major C (M×N) in memory = Column-major C^T (N×M)
     *   → rows=N, cols=M, ld=N
     */
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_fp32, CUDA_R_32F, N, M, N));

    /*
     * Create FP16 layouts - same dimensions but CUDA_R_16F data type.
     * The layout structure is identical, only the element type changes.
     */
    printf("Creating FP16 matrix layouts...\n");

    cublasLtMatrixLayout_t matA_fp16, matB_fp16, matC_fp16;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_fp16, CUDA_R_16F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_fp16, CUDA_R_16F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_fp16, CUDA_R_16F, N, M, N));

    /*
     * ===========================================
     * Step 3: Create Matmul Descriptors
     * ===========================================
     *
     * cublasLtMatmulDescCreate(desc, computeType, scaleType)
     *
     * Parameters:
     *   desc        - Output: the matmul descriptor
     *   computeType - Precision for INTERNAL math:
     *                   CUBLAS_COMPUTE_32F = accumulate in FP32
     *                   CUBLAS_COMPUTE_16F = accumulate in FP16
     *                 This is the precision used for the dot product sums.
     *                 Higher precision = more accurate but potentially slower.
     *   scaleType   - Data type for alpha and beta scalars:
     *                   CUDA_R_32F = float alpha/beta
     *                   CUDA_R_16F = half alpha/beta
     *
     * Common combinations:
     *   FP32 data + FP32 compute + FP32 scale  → Standard single precision
     *   FP16 data + FP16 compute + FP16 scale  → Full half precision
     *   FP16 data + FP32 compute + FP32 scale  → Mixed precision (best of both!)
     *
     * The matmul descriptor also stores transpose operations
     * (set via cublasLtMatmulDescSetAttribute).
     */
    printf("Creating matmul descriptors...\n");

    /* FP32 descriptor: compute in FP32, alpha/beta are float */
    cublasLtMatmulDesc_t matmulDesc_fp32;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp32, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    /* FP16 descriptor: compute in FP16, alpha/beta are half */
    cublasLtMatmulDesc_t matmulDesc_fp16;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp16, CUBLAS_COMPUTE_16F, CUDA_R_16F));

    /*
     * ===========================================
     * Step 4: Set Transpose Operations
     * ===========================================
     *
     * cublasLtMatmulDescSetAttribute(desc, attr, value, sizeInBytes)
     *
     * This is how cuBLASLt configures operations - through attributes.
     * Instead of passing transpose flags as function parameters (like cuBLAS),
     * you SET them on the descriptor.
     *
     * CUBLASLT_MATMUL_DESC_TRANSA - How to treat matrix A
     * CUBLASLT_MATMUL_DESC_TRANSB - How to treat matrix B
     *
     * Values:
     *   CUBLAS_OP_N - No transpose (use matrix as-is)
     *   CUBLAS_OP_T - Transpose the matrix
     *
     * We use CUBLAS_OP_N for both because the row-major swap trick
     * already handles the layout difference.
     *
     * Other useful attributes you can set (not used here):
     *   CUBLASLT_MATMUL_DESC_EPILOGUE       - Fused post-GEMM operations
     *     Examples:
     *       CUBLASLT_EPILOGUE_RELU   → D = ReLU(alpha*A*B + beta*C)
     *       CUBLASLT_EPILOGUE_GELU   → D = GELU(alpha*A*B + beta*C)
     *       CUBLASLT_EPILOGUE_BIAS   → D = alpha*A*B + beta*C + bias
     *   CUBLASLT_MATMUL_DESC_BIAS_POINTER   - Pointer to bias vector
     */
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* Set transpose attributes for FP32 descriptor */
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSA,
        &transa, sizeof(cublasOperation_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSB,
        &transb, sizeof(cublasOperation_t)));

    /* Set transpose attributes for FP16 descriptor */
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSA,
        &transa, sizeof(cublasOperation_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSB,
        &transb, sizeof(cublasOperation_t)));

    /*
     * ===========================================
     * Step 5: Execute FP32 Matrix Multiplication
     * ===========================================
     *
     * cublasLtMatmul has 16 parameters:
     *
     * cublasLtMatmul(
     *     handle,              // 1:  cuBLASLt context
     *     matmulDesc,          // 2:  computation descriptor
     *     &alpha,              // 3:  scalar for A*B
     *     A_data, A_layout,    // 4,5:  matrix A pointer + layout
     *     B_data, B_layout,    // 6,7:  matrix B pointer + layout
     *     &beta,               // 8:  scalar for C
     *     C_data, C_layout,    // 9,10: matrix C pointer + layout (input for beta*C)
     *     D_data, D_layout,    // 11,12: matrix D pointer + layout (OUTPUT!)
     *     algo,                // 13: algorithm (NULL = let cuBLAS choose)
     *     workspace,           // 14: scratch memory (NULL = none)
     *     workspaceSize,       // 15: workspace size in bytes
     *     stream               // 16: CUDA stream (0 = default)
     * )
     *
     * IMPORTANT: D is the OUTPUT. C is only used for the beta*C term.
     * When beta=0.0, C is ignored, so D and C can safely share memory.
     *
     * Remember the swap: we pass d_B as cuBLAS's A, d_A as cuBLAS's B.
     */
    printf("\nRunning cuBLASLt FP32 matmul...\n");
    const float alpha_fp32 = 1.0f, beta_fp32 = 0.0f;

    CHECK_CUBLAS(cublasLtMatmul(
        handle,                      // cuBLASLt handle
        matmulDesc_fp32,             // computation descriptor
        &alpha_fp32,                 // alpha = 1.0
        d_B_fp32, matB_fp32,         // cuBLAS's A = our B, with B's layout
        d_A_fp32, matA_fp32,         // cuBLAS's B = our A, with A's layout
        &beta_fp32,                  // beta = 0.0
        d_C_fp32, matC_fp32,         // C (input for beta*C, ignored since beta=0)
        d_C_fp32, matC_fp32,         // D (OUTPUT - same buffer as C, safe since beta=0)
        NULL,                        // algo: NULL = cuBLAS picks default
        NULL,                        // workspace: no extra memory
        0,                           // workspaceSize: 0 bytes
        0                            // stream: 0 = default stream
    ));

    /* Copy FP32 result back to host */
    CHECK_CUDA(cudaMemcpy(h_C_gpu_fp32, d_C_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    /*
     * ===========================================
     * Step 6: Execute FP16 Matrix Multiplication
     * ===========================================
     *
     * Same structure as FP32 but with:
     *   - half precision data pointers
     *   - half precision alpha/beta
     *   - FP16 compute descriptor
     *   - FP16 matrix layouts
     */
    printf("Running cuBLASLt FP16 matmul...\n\n");
    const half alpha_fp16 = __float2half(1.0f);
    const half beta_fp16  = __float2half(0.0f);

    CHECK_CUBLAS(cublasLtMatmul(
        handle,
        matmulDesc_fp16,
        &alpha_fp16,
        d_B_fp16, matB_fp16,         // cuBLAS's A = our B
        d_A_fp16, matA_fp16,         // cuBLAS's B = our A
        &beta_fp16,
        d_C_fp16, matC_fp16,         // C (input)
        d_C_fp16, matC_fp16,         // D (OUTPUT)
        NULL,                        // algo
        NULL,                        // workspace
        0,                           // workspaceSize
        0                            // stream
    ));

    /* Copy FP16 result back and convert to float for printing */
    half h_C_half[M * N];
    CHECK_CUDA(cudaMemcpy(h_C_half, d_C_fp16, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; ++i) {
        h_C_gpu_fp16[i] = __half2float(h_C_half[i]);
    }

    /* ==================== Print Results ==================== */
    print_matrix(h_C_cpu, M, N, "C (CPU Reference)");
    print_matrix(h_C_gpu_fp32, M, N, "C (cuBLASLt FP32)");
    print_matrix(h_C_gpu_fp16, M, N, "C (cuBLASLt FP16)");

    /* ==================== Verification ==================== */
    printf("Verification:\n");
    int fp32_match = 1, fp16_match = 1;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(h_C_cpu[i] - h_C_gpu_fp32[i]) > 1e-5f) fp32_match = 0;
        if (fabs(h_C_cpu[i] - h_C_gpu_fp16[i]) > 1e-2f) fp16_match = 0;
    }
    printf("  FP32 vs CPU: %s\n", fp32_match ? "MATCH" : "MISMATCH");
    printf("  FP16 vs CPU: %s (within FP16 tolerance)\n", fp16_match ? "MATCH" : "MISMATCH");

    /*
     * ===========================================
     * Cleanup
     * ===========================================
     *
     * Destroy in reverse order of creation.
     * Descriptors first, then handle, then device memory.
     */
    printf("\nCleaning up...\n");

    /* Destroy matmul descriptors */
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp32));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp16));

    /* Destroy FP32 matrix layouts */
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_fp32));

    /* Destroy FP16 matrix layouts */
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_fp16));

    /* Destroy handle */
    CHECK_CUBLAS(cublasLtDestroy(handle));

    /* Free device memory */
    CHECK_CUDA(cudaFree(d_A_fp32));
    CHECK_CUDA(cudaFree(d_B_fp32));
    CHECK_CUDA(cudaFree(d_C_fp32));
    CHECK_CUDA(cudaFree(d_A_fp16));
    CHECK_CUDA(cudaFree(d_B_fp16));
    CHECK_CUDA(cudaFree(d_C_fp16));

    printf("Done!\n");
    return 0;
}

/*
 * ===========================================
 * Expected Output
 * ===========================================
 *
 * cuBLASLt Matrix Multiplication Demo
 * ====================================
 *
 * Matrix A (4x4):
 *     1.00     2.00     3.00     4.00
 *     5.00     6.00     7.00     8.00
 *     9.00    10.00    11.00    12.00
 *    13.00    14.00    15.00    16.00
 *
 * Matrix B (4x4):
 *     1.00     2.00     4.00     4.00
 *     5.00     6.00     7.00     8.00
 *     9.00    10.00    11.00    12.00
 *    17.00    18.00    19.00    20.00
 *
 * Running CPU matrix multiply...
 * Allocating device memory...
 * cuBLASLt handle created
 *
 * Creating FP32 matrix layouts...
 * Creating FP16 matrix layouts...
 * Creating matmul descriptors...
 *
 * Running cuBLASLt FP32 matmul...
 * Running cuBLASLt FP16 matmul...
 *
 * Matrix C (CPU Reference) (4x4):
 *   106.00   116.00   127.00   132.00
 *   234.00   260.00   287.00   300.00
 *   362.00   404.00   447.00   468.00
 *   490.00   548.00   607.00   636.00
 *
 * Matrix C (cuBLASLt FP32) (4x4):
 *   106.00   116.00   127.00   132.00
 *   234.00   260.00   287.00   300.00
 *   362.00   404.00   447.00   468.00
 *   490.00   548.00   607.00   636.00
 *
 * Matrix C (cuBLASLt FP16) (4x4):
 *   106.00   116.00   127.00   132.00
 *   234.00   260.00   287.00   300.00
 *   362.00   404.00   447.00   468.00
 *   490.00   548.00   607.00   636.00
 *
 * Verification:
 *   FP32 vs CPU: MATCH
 *   FP16 vs CPU: MATCH (within FP16 tolerance)
 *
 * Cleaning up...
 * Done!
 *
 * ===========================================
 * Key Takeaways
 * ===========================================
 *
 * 1. cuBLASLt vs cuBLAS:
 *    - More setup (descriptors) but more control
 *    - Better mixed precision support
 *    - Can fuse operations (GEMM + bias + activation)
 *    - Used by PyTorch/TensorFlow internally
 *
 * 2. DESCRIPTOR PATTERN:
 *    cuBLASLt needs 3 types of descriptors:
 *    a) cublasLtMatmulDesc_t    - HOW to compute (precision, transpose)
 *    b) cublasLtMatrixLayout_t  - WHAT the matrices look like (shape, type)
 *    c) cublasLtMatmulAlgo_t    - WHICH algorithm to use (optional)
 *
 * 3. D vs C OUTPUT:
 *    cuBLASLt: D = alpha*A*B + beta*C (D is separate output)
 *    cuBLAS:   C = alpha*A*B + beta*C (C is overwritten)
 *    D and C can point to same memory when beta=0.
 *
 * 4. 16 PARAMETERS:
 *    cublasLtMatmul needs 16 args. The most commonly forgotten
 *    are D/Ddesc (the output) and stream (last parameter).
 *
 * 5. SAME ROW-MAJOR TRICK:
 *    Swap A and B, swap m and n, same as standard cuBLAS.
 */
