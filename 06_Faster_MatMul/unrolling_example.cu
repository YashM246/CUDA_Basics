/*
 * ==========================================================================
 *  unrolling_example.cu  –  Loop Unrolling with #pragma unroll
 * ==========================================================================
 *
 *  WHAT THIS PROGRAM DOES
 *  ----------------------
 *  Compares two identical kernels:
 *    1. vectorAddNoUnroll  (no explicit unrolling)
 *    2. vectorAddUnroll    (with #pragma unroll directive)
 *
 *  Both compute the same result, but we measure whether the compiler
 *  produces faster code when explicitly told to unroll the loop.
 *
 *  WHAT IS LOOP UNROLLING?
 *  -------------------------
 *  Loop unrolling replaces iterations with repeated code to:
 *    - Reduce loop overhead (fewer branch instructions)
 *    - Expose instruction-level parallelism (ILP)
 *    - Enable better register allocation
 *
 *  Example:
 *    Original:                Unrolled:
 *    for (int i=0; i<4; i++)  sum += a[0] + b[0];
 *      sum += a[i] + b[i];    sum += a[1] + b[1];
 *                             sum += a[2] + b[2];
 *                             sum += a[3] + b[3];
 *
 *  THE ARTIFICIAL LOOP
 *  --------------------
 *  The kernels below contain an artificial loop:
 *
 *    for (j = 0; j < LOOP_COUNT; j++)
 *        sum += a[tid] + b[tid];
 *
 *  This is NOT realistic — it just adds the same value 100 times.
 *  The purpose is to CREATE a loop that:
 *    1. Is small enough to unroll (LOOP_COUNT = 100)
 *    2. Has independent iterations (no loop-carried dependencies)
 *    3. Can measure unrolling impact in isolation
 *
 *  In real matmul kernels, the loop would iterate over different
 *  array indices (e.g., the inner product k-loop), not repeat the
 *  same operation.
 *
 *  DOES THE COMPILER AUTO-UNROLL?
 *  --------------------------------
 *  Modern compilers often unroll loops automatically, even without
 *  #pragma unroll.  This example tests whether:
 *    1. The compiler already unrolls the loop (speedup ≈ 1.0x)
 *    2. #pragma unroll helps (speedup > 1.0x)
 *
 *  To check if unrolling happened, inspect the PTX assembly:
 *    nvcc -ptx unrolling_example.cu -o - | less
 *
 *  Search for the kernel name (vectorAddNoUnroll / vectorAddUnroll)
 *  and look for repeated instructions instead of loop control flow.
 *
 *  TYPICAL RESULTS
 *  ----------------
 *  On most GPUs, both kernels perform identically because:
 *    - The compiler auto-unrolls small loops with constant bounds
 *    - The loop body is trivial (one FP add, no memory latency)
 *
 *  You'd see a difference in more complex loops where the compiler
 *  can't auto-unroll (e.g., variable bounds, large iteration counts).
 *
 *  COMPILE
 *  --------
 *  nvcc unrolling_example.cu -o unrolling_example
 *
 *  To view PTX and verify unrolling:
 *    nvcc -ptx unrolling_example.cu -o unrolling.ptx
 *    cat unrolling.ptx | grep -A 50 vectorAddNoUnroll
 * ==========================================================================
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Problem size: 10 million elements × 100 iterations each */
#define N                  10000000
#define THREADS_PER_BLOCK  256
#define LOOP_COUNT         100
#define WARMUP_RUNS        5
#define BENCH_RUNS         10

/* -----------------------------------------------------------------------
 *  ERROR-CHECKING MACROS
 * ----------------------------------------------------------------------- */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d : %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* -----------------------------------------------------------------------
 *  KERNEL WITHOUT EXPLICIT UNROLLING
 * -----------------------------------------------------------------------
 *  The compiler MAY auto-unroll this loop, or it may not.  Check the
 *  PTX to see what actually happened.
 * ----------------------------------------------------------------------- */
__global__ void vectorAddNoUnroll(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        float sum = 0.0f;

        /* Artificial loop: adds (a[tid] + b[tid]) 100 times.
         * In a real kernel, the loop would iterate over different
         * array indices (e.g., k-loop in matmul). */
        for (int j = 0; j < LOOP_COUNT; j++) {
            sum += a[tid] + b[tid];
        }

        c[tid] = sum;
    }
}

/* -----------------------------------------------------------------------
 *  KERNEL WITH EXPLICIT UNROLLING
 * -----------------------------------------------------------------------
 *  #pragma unroll tells the compiler: "replace this loop with repeated
 *  instructions."  If LOOP_COUNT=100, this generates 100 copies of
 *  the loop body.
 *
 *  Benefits (if compiler didn't already unroll):
 *    - No loop control overhead (setp, bra instructions)
 *    - Better instruction scheduling (GPU can issue multiple adds/cycle)
 *    - More registers available for intermediate values
 *
 *  Cost:
 *    - Larger code size (100 instructions instead of 1 + loop control)
 *    - May spill registers if the loop is too large
 * ----------------------------------------------------------------------- */
__global__ void vectorAddUnroll(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        float sum = 0.0f;

        #pragma unroll
        for (int j = 0; j < LOOP_COUNT; j++) {
            sum += a[tid] + b[tid];
        }

        c[tid] = sum;
    }
}

/* -----------------------------------------------------------------------
 *  VERIFY RESULTS
 * -----------------------------------------------------------------------
 *  Expected value: (1.0 + 2.0) * LOOP_COUNT = 3.0 * 100 = 300.0
 *
 *  BUG FIX: Original code used abs() which is for integers and would
 *  truncate floats.  Correct: fabsf() for single-precision absolute value.
 * ----------------------------------------------------------------------- */
int verifyResults(float *c, int n, float *max_error) {
    float expected = (1.0f + 2.0f) * LOOP_COUNT;   /* 300.0 */
    *max_error = 0.0f;

    for (int i = 0; i < n; i++) {
        float err = fabsf(c[i] - expected);
        if (err > *max_error) {
            *max_error = err;
        }
        if (err > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f (error = %e)\n",
                    i, c[i], expected, err);
            return 0;
        }
    }
    return 1;
}

/* -----------------------------------------------------------------------
 *  RUN KERNEL AND MEASURE TIME
 * -----------------------------------------------------------------------
 *  Uses CUDA events for accurate GPU-side timing.
 *  Returns elapsed time in milliseconds.
 * ----------------------------------------------------------------------- */
float runKernel(void (*kernel)(float*, float*, float*, int),
                float *d_a, float *d_b, float *d_c, int n,
                const char *kernel_name) {
    int numBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEvent_t start, stop;
    float milliseconds;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaEventRecord(stop));

    /* Check for kernel launch errors (invalid config, etc.) */
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds;
}

/* ======================================================================= */
int main(void) {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    printf("==========================================================\n");
    printf("  Loop Unrolling Benchmark (#pragma unroll)\n");
    printf("==========================================================\n\n");
    printf("Problem size:    %d elements\n", N);
    printf("Loop iterations: %d (per element)\n", LOOP_COUNT);
    printf("Total ops:       %ld (= N * LOOP_COUNT)\n", (long)N * LOOP_COUNT);
    printf("Warmup runs:     %d\n", WARMUP_RUNS);
    printf("Benchmark runs:  %d\n\n", BENCH_RUNS);

    /* -------------------------------------------------------------------
     *  ALLOCATE HOST MEMORY
     * ------------------------------------------------------------------- */
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory (%zu bytes x 3)\n", size);
        return EXIT_FAILURE;
    }

    /* Initialize: a[i] = 1.0, b[i] = 2.0
     * Expected result: c[i] = (1.0 + 2.0) * LOOP_COUNT = 300.0 */
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    /* -------------------------------------------------------------------
     *  ALLOCATE DEVICE MEMORY
     * ------------------------------------------------------------------- */
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* -------------------------------------------------------------------
     *  WARMUP
     * -------------------------------------------------------------------
     *  Run both kernels several times to:
     *    - Bring GPU out of idle power state
     *    - Trigger JIT compilation of PTX → SASS
     *    - Warm up L2 cache
     * ------------------------------------------------------------------- */
    printf("Running warmup...\n");
    for (int i = 0; i < WARMUP_RUNS; i++) {
        runKernel(vectorAddNoUnroll, d_a, d_b, d_c, N, "vectorAddNoUnroll");
        runKernel(vectorAddUnroll,   d_a, d_b, d_c, N, "vectorAddUnroll");
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Warmup complete.\n\n");

    /* -------------------------------------------------------------------
     *  BENCHMARK
     * -------------------------------------------------------------------
     *  BUG FIX: Run each kernel in its own loop to avoid cache
     *  interference.  The original code alternated kernels in the same
     *  loop, which can cause the second kernel to benefit from warm
     *  cache lines left by the first.
     * ------------------------------------------------------------------- */
    printf("Benchmarking...\n");

    float totalTimeNoUnroll = 0.0f;
    for (int i = 0; i < BENCH_RUNS; i++) {
        totalTimeNoUnroll += runKernel(vectorAddNoUnroll, d_a, d_b, d_c, N,
                                       "vectorAddNoUnroll");
    }

    float totalTimeUnroll = 0.0f;
    for (int i = 0; i < BENCH_RUNS; i++) {
        totalTimeUnroll += runKernel(vectorAddUnroll, d_a, d_b, d_c, N,
                                     "vectorAddUnroll");
    }

    float avgTimeNoUnroll = totalTimeNoUnroll / BENCH_RUNS;
    float avgTimeUnroll   = totalTimeUnroll   / BENCH_RUNS;
    float speedup         = avgTimeNoUnroll / avgTimeUnroll;

    /* -------------------------------------------------------------------
     *  VERIFY CORRECTNESS
     * ------------------------------------------------------------------- */
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    float max_error = 0.0f;
    int correct = verifyResults(h_c, N, &max_error);

    /* -------------------------------------------------------------------
     *  PRINT RESULTS
     * ------------------------------------------------------------------- */
    printf("\n");
    printf("==========================================================\n");
    printf("  RESULTS\n");
    printf("==========================================================\n\n");
    printf("  %-30s  %10.3f ms\n", "No unroll (average):", avgTimeNoUnroll);
    printf("  %-30s  %10.3f ms\n", "With #pragma unroll (avg):", avgTimeUnroll);
    printf("  %-30s  %10.2fx\n",   "Speedup:", speedup);
    printf("  %-30s  %s\n",        "Results correct:", correct ? "YES" : "NO");
    printf("  %-30s  %.2e\n",      "Max error:", max_error);
    printf("\n");

    if (speedup > 1.05f) {
        printf("  → #pragma unroll provides a %.1fx speedup.\n", speedup);
        printf("    The compiler did NOT auto-unroll the loop.\n");
    } else if (speedup < 0.95f) {
        printf("  → The unrolled version is SLOWER (%.2fx).\n", speedup);
        printf("    Possible reasons: register spilling, code bloat.\n");
    } else {
        printf("  → Performance is nearly identical (%.2fx).\n", speedup);
        printf("    The compiler likely auto-unrolled the loop already.\n");
        printf("    Check the PTX with: nvcc -ptx unrolling_example.cu -o -\n");
    }
    printf("\n");

    /* -------------------------------------------------------------------
     *  CLEANUP
     * ------------------------------------------------------------------- */
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}
