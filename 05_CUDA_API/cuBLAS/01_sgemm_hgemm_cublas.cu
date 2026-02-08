/*
 * cuBLAS SGEMM vs HGEMM - Single vs Half Precision Matrix Multiplication
 * ========================================================================
 *
 * This program demonstrates two cuBLAS GEMM operations:
 *   1. SGEMM - Single precision (float, 32-bit)
 *   2. HGEMM - Half precision (FP16, 16-bit)
 *
 * Both are compared against a CPU reference implementation.
 *
 * ===========================================
 * What is GEMM?
 * ===========================================
 *
 * GEMM = General Matrix Multiply
 *
 * It computes:  C = alpha * A * B + beta * C
 *
 *   alpha, beta = scalar multipliers
 *   A = M x K matrix
 *   B = K x N matrix
 *   C = M x N matrix (result)
 *
 * When alpha=1.0 and beta=0.0, this simplifies to: C = A * B
 *
 * ===========================================
 * SGEMM vs HGEMM
 * ===========================================
 *
 *   SGEMM = Single precision GEMM (float, 32-bit)
 *     - More precise (7 decimal digits)
 *     - Uses more memory (4 bytes per element)
 *     - Standard for most applications
 *
 *   HGEMM = Half precision GEMM (FP16, 16-bit)
 *     - Less precise (3-4 decimal digits)
 *     - Uses less memory (2 bytes per element)
 *     - 2x faster on Tensor Cores (RTX 4070 has these!)
 *     - Standard in deep learning inference
 *
 * Naming convention:
 *   cublas[S]gemm → S = Single (float)
 *   cublas[D]gemm → D = Double (double)
 *   cublas[H]gemm → H = Half (FP16)
 *
 * ===========================================
 * The Column-Major Problem (CRITICAL!)
 * ===========================================
 *
 * cuBLAS inherited from Fortran BLAS, which uses COLUMN-MAJOR order.
 * C/C++ uses ROW-MAJOR order. This is the #1 source of bugs!
 *
 * Example with a 2x3 matrix:
 *
 *   Visual:     | 1  2  3 |
 *               | 4  5  6 |
 *
 *   Row-major memory (C/C++):     [1, 2, 3, 4, 5, 6]  (read left-to-right, top-to-bottom)
 *   Column-major memory (Fortran): [1, 4, 2, 5, 3, 6]  (read top-to-bottom, left-to-right)
 *
 * THE TRICK: Instead of converting your data to column-major,
 * you can swap A and B and compute B * A instead of A * B.
 *
 * Why does this work?
 * -------------------
 * A row-major matrix in memory is identical to its TRANSPOSE in column-major.
 *
 *   Row-major A (M×K) in memory = Column-major A^T (K×M)
 *   Row-major B (K×N) in memory = Column-major B^T (N×K)
 *
 * We want: C = A * B  (in row-major)
 * cuBLAS sees: C^T = B^T * A^T  (in column-major)
 *
 * Since (A*B)^T = B^T * A^T, this gives us the correct result!
 *
 * So we call cublasSgemm with arguments SWAPPED:
 *   - Pass d_B as the first matrix (cuBLAS's "A")
 *   - Pass d_A as the second matrix (cuBLAS's "B")
 *   - Swap m and n dimensions accordingly
 *
 * ===========================================
 * cublasSgemm Parameter Reference
 * ===========================================
 *
 * cublasSgemm(handle, transa, transb, m, n, k,
 *             &alpha, A, lda, B, ldb, &beta, C, ldc)
 *
 * Parameters (in column-major thinking):
 *   handle  - cuBLAS context
 *   transa  - Transpose A? (CUBLAS_OP_N = no, CUBLAS_OP_T = yes)
 *   transb  - Transpose B? (CUBLAS_OP_N = no, CUBLAS_OP_T = yes)
 *   m       - Rows of op(A) and C
 *   n       - Columns of op(B) and C
 *   k       - Columns of op(A) / Rows of op(B) (shared dimension)
 *   alpha   - Scalar multiplier for A*B
 *   A       - Device pointer to matrix A
 *   lda     - Leading dimension of A (number of rows in column-major storage)
 *   B       - Device pointer to matrix B
 *   ldb     - Leading dimension of B
 *   beta    - Scalar multiplier for C
 *   C       - Device pointer to matrix C (output)
 *   ldc     - Leading dimension of C
 *
 * With the row-major swap trick (B*A instead of A*B):
 *   m   = N  (columns of our row-major result)
 *   n   = M  (rows of our row-major result)
 *   k   = K  (shared dimension)
 *   A   = d_B, lda = N
 *   B   = d_A, ldb = K
 *   C   = d_C, ldc = N
 *
 * ===========================================
 * What is "Leading Dimension" (lda, ldb, ldc)?
 * ===========================================
 *
 * In column-major storage, the leading dimension is the number of ROWS
 * in the matrix (i.e., the stride between consecutive columns in memory).
 *
 * For a column-major M×N matrix stored contiguously: lda = M
 *
 * But with the row-major trick, our row-major matrices are seen as
 * their column-major transposes:
 *   - Row-major A (M×K) → Column-major A^T (K×M) → lda = K? No...
 *   - Since we SWAP and pass d_B as cuBLAS's first matrix:
 *     Row-major B (K×N) → Column-major B^T (N×K) → lda = N
 *   - And d_A as cuBLAS's second matrix:
 *     Row-major A (M×K) → Column-major A^T (K×M) → ldb = K
 *
 * Compile: nvcc 01_sgemm_hgemm_cublas.cu -o 01_sgemm_hgemm_cublas.exe -lcublas
 * Run:     .\01_sgemm_hgemm_cublas.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>      // cuBLAS API (v2 = modern pointer-based API)
#include <cuda_fp16.h>      // FP16 (half precision) support

/*
 * Matrix dimensions
 * A is M×K, B is K×N, C is M×N
 *
 * A (3×2):    B (2×4):         C (3×4):
 * | 1  2 |   | 1  2  3  4 |   | 11  14  17  20 |
 * | 3  4 |   | 5  6  7  8 |   | 23  30  37  44 |
 * | 5  6 |                     | 35  46  57  68 |
 */
#define M 3
#define N 4
#define K 2

/*
 * ===========================================
 * Error Checking Macros
 * ===========================================
 *
 * CUDA Runtime and cuBLAS have DIFFERENT error types:
 *   - CUDA Runtime: cudaError_t, checked against cudaSuccess
 *   - cuBLAS: cublasStatus_t, checked against CUBLAS_STATUS_SUCCESS
 *
 * So we need TWO separate macros.
 *
 * Note: cuBLAS doesn't provide a cublasGetErrorString() like
 * CUDA runtime does, so we just print the numeric status code.
 *
 * Common cuBLAS status codes:
 *   0 = CUBLAS_STATUS_SUCCESS
 *   1 = CUBLAS_STATUS_NOT_INITIALIZED  (forgot cublasCreate?)
 *   7 = CUBLAS_STATUS_INVALID_VALUE    (wrong dimensions/pointers?)
 *  11 = CUBLAS_STATUS_MAPPING_ERROR    (memory access error)
 *  13 = CUBLAS_STATUS_EXECUTION_FAILED (kernel launch failed)
 *  14 = CUBLAS_STATUS_INTERNAL_ERROR   (internal cuBLAS failure)
 */
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error in %s:%d: %d\n", \
                    __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/*
 * Print matrix macro - prints a row-major matrix with formatting.
 *
 * Parameters:
 *   mat  - Pointer to float array
 *   rows - Number of rows
 *   cols - Number of columns
 *
 * Access pattern: mat[i*cols + j]
 *   i*cols skips to the correct row
 *   +j selects the column within that row
 */
#define PRINT_MATRIX(mat, rows, cols) \
    for (int i = 0; i < rows; i++) { \
        for (int j = 0; j < cols; j++) { \
            printf("%8.3f ", (mat)[i * cols + j]); \
        } \
        printf("\n"); \
    }

/*
 * ===========================================
 * CPU Matrix Multiply (Reference)
 * ===========================================
 *
 * Standard triple-loop matrix multiplication.
 * Used to verify the cuBLAS results are correct.
 *
 * C[i][j] = sum over k of A[i][k] * B[k][j]
 */
void cpu_matmul(float *A, float *B, float *C) {
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

int main() {
    printf("cuBLAS SGEMM vs HGEMM Demo\n");
    printf("==========================\n\n");

    /*
     * ===========================================
     * Initialize Matrices (Row-Major)
     * ===========================================
     *
     * A (3×2):             B (2×4):
     * | 1.0  2.0 |         | 1.0  2.0  3.0  4.0 |
     * | 3.0  4.0 |         | 5.0  6.0  7.0  8.0 |
     * | 5.0  6.0 |
     *
     * Memory layout (row-major, how C stores it):
     *   A: [1, 2, 3, 4, 5, 6]
     *   B: [1, 2, 3, 4, 5, 6, 7, 8]
     */
    float A[M * K] = {
        1.0f, 2.0f,    // Row 0
        3.0f, 4.0f,    // Row 1
        5.0f, 6.0f     // Row 2
    };
    float B[K * N] = {
        1.0f, 2.0f, 3.0f, 4.0f,    // Row 0
        5.0f, 6.0f, 7.0f, 8.0f     // Row 1
    };
    float C_cpu[M * N], C_cublas_s[M * N], C_cublas_h[M * N];

    /* ==================== CPU Reference ==================== */
    printf("Running CPU matrix multiply...\n");
    cpu_matmul(A, B, C_cpu);

    /* ==================== cuBLAS Setup ==================== */

    /*
     * cublasHandle_t - Opaque handle holding cuBLAS library context.
     *
     * It stores:
     *   - Which GPU to use
     *   - Stream association
     *   - Math mode settings
     *   - Internal memory allocations
     *
     * You MUST create one before calling any cuBLAS function.
     * One handle can be reused for many operations.
     * Typically create ONE handle per application.
     */
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    printf("cuBLAS handle created\n\n");

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    /* Copy host matrices to device */
    CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    /*
     * ===========================================================
     * SGEMM - Single Precision GEMM (float, 32-bit)
     * ===========================================================
     *
     * Computes: C = alpha * A * B + beta * C
     *
     * With alpha=1.0 and beta=0.0: C = A * B
     *
     * Since our data is in ROW-MAJOR (C/C++ default) but cuBLAS
     * expects COLUMN-MAJOR, we use the swap trick:
     *
     *   Instead of: cublasSgemm(..., d_A, ..., d_B, ...)   // WRONG for row-major
     *   We call:    cublasSgemm(..., d_B, ..., d_A, ...)   // Swap A and B!
     *
     * And we swap m↔n:
     *   m = N = 4  (becomes "rows" in cuBLAS's column-major view)
     *   n = M = 3  (becomes "cols" in cuBLAS's column-major view)
     *   k = K = 2  (shared dimension, unchanged)
     *
     * Parameter mapping for our call:
     *   cublasSgemm(handle,
     *       CUBLAS_OP_N,   // transa: Don't transpose "A" (which is our B)
     *       CUBLAS_OP_N,   // transb: Don't transpose "B" (which is our A)
     *       N,             // m = 4 (rows of cuBLAS's "A" = cols of our B)
     *       M,             // n = 3 (cols of cuBLAS's "B" = rows of our A)
     *       K,             // k = 2 (shared dimension)
     *       &alpha,        // scalar = 1.0
     *       d_B, N,        // cuBLAS's "A" = our B (K×N), lda = N = 4
     *       d_A, K,        // cuBLAS's "B" = our A (M×K), ldb = K = 2
     *       &beta,         // scalar = 0.0
     *       d_C, N         // output C (M×N), ldc = N = 4
     *   );
     */
    printf("Running cuBLAS SGEMM (single precision)...\n");
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,         // "A" = our B, lda = N
        d_A, K,         // "B" = our A, ldb = K
        &beta,
        d_C, N          // C output, ldc = N
    ));

    /* Copy SGEMM result back to host */
    CHECK_CUDA(cudaMemcpy(C_cublas_s, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    /*
     * ===========================================================
     * HGEMM - Half Precision GEMM (FP16, 16-bit)
     * ===========================================================
     *
     * Same operation as SGEMM but using half precision (FP16).
     *
     * What is half precision (FP16)?
     * ------------------------------
     * A 16-bit floating point format:
     *   - 1 sign bit, 5 exponent bits, 10 mantissa bits
     *   - Range: ~6×10^-8 to 65504
     *   - Precision: ~3-4 decimal digits (vs 7 for float)
     *   - Size: 2 bytes (vs 4 bytes for float)
     *
     * Why use FP16?
     *   - 2x less memory → larger batches, larger models
     *   - Tensor Cores (RTX 4070) do FP16 math MUCH faster
     *   - Good enough for deep learning inference
     *   - NOT good enough for scientific computing (too imprecise)
     *
     * The `half` type (from cuda_fp16.h):
     *   - CUDA's FP16 type (also called __half)
     *   - Conversion functions:
     *       __float2half(float)  → convert float to half
     *       __half2float(half)   → convert half back to float
     *   - These work on both host and device (CUDA 12+)
     *
     * Process:
     *   1. Convert float arrays → half arrays (on CPU)
     *   2. Copy half arrays to GPU
     *   3. Run cublasHgemm (FP16 math on GPU)
     *   4. Copy half result back to CPU
     *   5. Convert half result → float for printing/comparison
     */
    printf("Running cuBLAS HGEMM (half precision)...\n\n");

    /* Allocate device memory for half precision */
    half *d_A_h, *d_B_h, *d_C_h;
    CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(half)));

    /*
     * Convert float → half on CPU.
     *
     * __float2half(float value) converts a 32-bit float to 16-bit half.
     * Precision loss happens here! For example:
     *   float: 3.14159265  →  half: 3.140625 (lost some digits)
     *
     * For our small integer values (1-12), conversion is exact.
     */
    half A_h[M * K], B_h[K * N];
    for (int i = 0; i < M * K; i++) {
        A_h[i] = __float2half(A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        B_h[i] = __float2half(B[i]);
    }

    /* Copy half precision data to device */
    CHECK_CUDA(cudaMemcpy(d_A_h, A_h, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_h, B_h, K * N * sizeof(half), cudaMemcpyHostToDevice));

    /*
     * cublasHgemm - same signature as cublasSgemm but with half types.
     * Same row-major swap trick applies (swap A↔B, swap m↔n).
     *
     * Note: alpha and beta must also be __half type for HGEMM.
     */
    __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    CHECK_CUBLAS(cublasHgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        d_B_h, N,       // "A" = our B_h, lda = N
        d_A_h, K,        // "B" = our A_h, ldb = K
        &beta_h,
        d_C_h, N         // C output, ldc = N
    ));

    /*
     * Copy half result back and convert to float for printing.
     * __half2float(half value) converts 16-bit back to 32-bit.
     */
    half C_h[M * N];
    CHECK_CUDA(cudaMemcpy(C_h, d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; i++) {
        C_cublas_h[i] = __half2float(C_h[i]);
    }

    /* ==================== Print Results ==================== */
    printf("Matrix A (%dx%d):\n", M, K);
    PRINT_MATRIX(A, M, K);
    printf("\nMatrix B (%dx%d):\n", K, N);
    PRINT_MATRIX(B, K, N);
    printf("\nCPU Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cpu, M, N);
    printf("\ncuBLAS SGEMM Result (float32) (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_s, M, N);
    printf("\ncuBLAS HGEMM Result (float16) (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_h, M, N);

    /* ==================== Verification ==================== */
    printf("\nVerification:\n");
    int sgemm_match = 1, hgemm_match = 1;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_cpu[i] - C_cublas_s[i]) > 1e-5f) sgemm_match = 0;
        if (fabs(C_cpu[i] - C_cublas_h[i]) > 1e-2f) hgemm_match = 0;  // FP16 has less precision
    }
    printf("  SGEMM vs CPU: %s\n", sgemm_match ? "MATCH" : "MISMATCH");
    printf("  HGEMM vs CPU: %s (within FP16 tolerance)\n", hgemm_match ? "MATCH" : "MISMATCH");

    /* ==================== Cleanup ==================== */
    printf("\nCleaning up...\n");
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_h));
    CHECK_CUDA(cudaFree(d_B_h));
    CHECK_CUDA(cudaFree(d_C_h));
    CHECK_CUBLAS(cublasDestroy(handle));
    printf("Done!\n");

    return 0;
}

/*
 * ===========================================
 * Expected Output
 * ===========================================
 *
 * cuBLAS SGEMM vs HGEMM Demo
 * ==========================
 *
 * Running CPU matrix multiply...
 * cuBLAS handle created
 *
 * Running cuBLAS SGEMM (single precision)...
 * Running cuBLAS HGEMM (half precision)...
 *
 * Matrix A (3x2):
 *    1.000    2.000
 *    3.000    4.000
 *    5.000    6.000
 *
 * Matrix B (2x4):
 *    1.000    2.000    3.000    4.000
 *    5.000    6.000    7.000    8.000
 *
 * CPU Result (3x4):
 *   11.000   14.000   17.000   20.000
 *   23.000   30.000   37.000   44.000
 *   35.000   46.000   57.000   68.000
 *
 * cuBLAS SGEMM Result (float32) (3x4):
 *   11.000   14.000   17.000   20.000
 *   23.000   30.000   37.000   44.000
 *   35.000   46.000   57.000   68.000
 *
 * cuBLAS HGEMM Result (float16) (3x4):
 *   11.000   14.000   17.000   20.000
 *   23.000   30.000   37.000   44.000
 *   35.000   46.000   57.000   68.000
 *
 * Verification:
 *   SGEMM vs CPU: MATCH
 *   HGEMM vs CPU: MATCH (within FP16 tolerance)
 *
 * Cleaning up...
 * Done!
 *
 * ===========================================
 * Key Takeaways
 * ===========================================
 *
 * 1. ROW-MAJOR TRICK:
 *    Swap A and B in cublasSgemm to handle C's row-major layout.
 *    This avoids manual data transposition.
 *
 * 2. SGEMM vs HGEMM:
 *    - SGEMM: 32-bit, precise, standard
 *    - HGEMM: 16-bit, faster on Tensor Cores, slight precision loss
 *    - For deep learning inference, HGEMM is usually preferred
 *
 * 3. HANDLE PATTERN:
 *    cublasCreate → operations → cublasDestroy
 *    One handle per application is typical.
 *
 * 4. ERROR CHECKING:
 *    Always check both CUDA and cuBLAS return codes.
 *    cuBLAS errors are silent without explicit checking.
 *
 * 5. FP16 CONVERSION:
 *    __float2half() and __half2float() convert between precisions.
 *    Precision loss is expected - use wider tolerance for comparisons.
 */
