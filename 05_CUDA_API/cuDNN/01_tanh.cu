/*
 * ==========================================================================
 *  01_tanh.cu  –  cuDNN Activation (tanh) vs Naive CUDA Kernel
 * ==========================================================================
 *
 *  WHAT THIS PROGRAM DOES
 *  ----------------------
 *  Compares two ways to compute tanh activation on a 4-D tensor:
 *    1. A naive CUDA kernel  (one thread per element, calls tanhf())
 *    2. cuDNN's cudnnActivationForward  (NVIDIA's optimized library)
 *
 *  Both results are verified against a CPU reference for correctness.
 *
 *  WHAT IS cuDNN?
 *  ---------------
 *  cuDNN (CUDA Deep Neural Network library) is NVIDIA's GPU-accelerated
 *  library for deep learning primitives:
 *    - Activation functions  (ReLU, tanh, sigmoid, ELU, ...)
 *    - Convolution           (forward, backward-data, backward-filter)
 *    - Pooling               (max, average)
 *    - Batch normalization
 *    - RNNs / LSTMs
 *    - Softmax, Dropout, ...
 *
 *  cuDNN does NOT do matrix multiplication (that's cuBLAS).
 *  cuDNN does NOT manage training loops (that's the framework: PyTorch, etc.).
 *
 *  WHY cuDNN OVER A HAND-WRITTEN KERNEL?
 *  ----------------------------------------
 *  For a simple element-wise op like tanh the difference may be small,
 *  but cuDNN can still win because:
 *    - Internally uses optimized memory access patterns
 *    - May fuse operations or pick specialized kernels per GPU architecture
 *    - Handles all data layouts (NCHW, NHWC) without user effort
 *
 *  For complex ops (convolution), cuDNN is dramatically faster because it
 *  selects from many algorithm variants (Winograd, FFT, implicit GEMM, ...).
 *
 *  cuDNN's DESCRIPTOR-BASED WORKFLOW
 *  ----------------------------------
 *  cuDNN never takes raw dimensions directly.  Instead you build
 *  "descriptor" objects that describe your data, then pass them to functions.
 *
 *  The pattern for every cuDNN operation:
 *
 *    1. cudnnCreate(&handle)          – Create the library context
 *    2. Create descriptors            – Describe tensors, operations, etc.
 *    3. Set descriptor fields         – Fill in dimensions, data types, etc.
 *    4. Call the operation             – e.g. cudnnActivationForward(...)
 *    5. Destroy descriptors           – Clean up (reverse order)
 *    6. cudnnDestroy(handle)          – Release the library context
 *
 *  For activation specifically:
 *
 *    ┌──────────────────────────────────────────────────────────────────┐
 *    │  cudnnHandle_t                                                  │
 *    │    └── cudnnTensorDescriptor_t   (describes input/output shape) │
 *    │    └── cudnnActivationDescriptor_t (describes the activation)   │
 *    │                                                                 │
 *    │  cudnnActivationForward(handle,                                 │
 *    │      activation_desc,                                           │
 *    │      &alpha, input_desc, d_input,     // y = alpha * f(x)      │
 *    │      &beta,  output_desc, d_output);  //     + beta * y_old    │
 *    └──────────────────────────────────────────────────────────────────┘
 *
 *  NCHW TENSOR FORMAT
 *  -------------------
 *  cuDNN uses 4-D tensors in NCHW order (the default):
 *
 *    N = batch size    (number of images)
 *    C = channels      (e.g. 3 for RGB, 32 for a conv layer output)
 *    H = height        (spatial rows)
 *    W = width         (spatial columns)
 *
 *  Memory layout (row-major, last index varies fastest):
 *
 *    element[n][c][h][w]  =  data[ n*(C*H*W) + c*(H*W) + h*W + w ]
 *
 *  ALPHA / BETA SCALING
 *  ---------------------
 *  Most cuDNN functions follow this pattern:
 *
 *    output = alpha * operation(input) + beta * output_old
 *
 *  Common usage:
 *    alpha=1, beta=0  →  output = operation(input)       (overwrite)
 *    alpha=1, beta=1  →  output = operation(input) + old (accumulate)
 *
 *  These are HOST pointers (float*), not device pointers.
 *
 *  COMPILE
 *  --------
 *  nvcc 01_tanh.cu -o 01_tanh -lcudnn
 *
 *  NOTE: cuDNN must be installed separately from the CUDA toolkit.
 *  Download from: https://developer.nvidia.com/cudnn-downloads
 * ==========================================================================
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>      /* for srand(time(NULL)) */

/* -----------------------------------------------------------------------
 *  ERROR-CHECKING MACROS
 * -----------------------------------------------------------------------
 *  Wrapped in do { ... } while(0) so they are safe inside if/else chains:
 *
 *    if (condition)
 *        CHECK_CUDA(someCall());   // Without do-while, the 'else' below
 *    else                          // would bind to the inner 'if' in the
 *        doSomethingElse();        // macro, not the outer 'if'.
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

#define CHECK_CUDNN(call)                                                   \
    do {                                                                    \
        cudnnStatus_t err = call;                                           \
        if (err != CUDNN_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuDNN error at %s:%d : %s\n",                 \
                    __FILE__, __LINE__, cudnnGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* -----------------------------------------------------------------------
 *  NAIVE TANH KERNEL
 * -----------------------------------------------------------------------
 *  One thread per element.  Each thread computes:
 *      output[idx] = tanhf(input[idx])
 *
 *  tanhf() is a CUDA device math function (single-precision tanh).
 *  It is implemented in hardware on NVIDIA GPUs, so it's quite fast.
 *
 *  This is the simplest possible implementation — no shared memory,
 *  no vectorized loads (float4), no loop unrolling.  cuDNN may use
 *  all of those optimizations internally.
 * ----------------------------------------------------------------------- */
__global__ void naiveTanhKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

/* -----------------------------------------------------------------------
 *  INITIALIZE DATA
 * -----------------------------------------------------------------------
 *  Fills array with random floats in [-1.0, +1.0].
 *
 *  Why this range?  tanh maps all of R → (-1, +1), but the interesting
 *  region (where the gradient is non-trivial) is roughly [-2, +2].
 *  Values far outside this range saturate to ±1.
 * ----------------------------------------------------------------------- */
void initializeData(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

/* -----------------------------------------------------------------------
 *  VERIFY RESULTS
 * -----------------------------------------------------------------------
 *  Compares GPU output against CPU reference using absolute error.
 *
 *  tolerance = 1e-5:  tanhf is computed in single precision on both
 *  CPU and GPU, so results should match very closely.  Small differences
 *  can arise from different instruction orderings or FMA usage.
 *
 *  Returns: 1 if all elements match within tolerance, 0 otherwise.
 *  Also sets *max_error to the largest absolute difference found.
 * ----------------------------------------------------------------------- */
int verifyResults(float *cpu_output, float *gpu_output, int size,
                  float tolerance, float *max_error) {
    *max_error = 0.0f;

    for (int i = 0; i < size; ++i) {
        float diff = fabsf(cpu_output[i] - gpu_output[i]);
        if (diff > *max_error) {
            *max_error = diff;
        }
        if (diff > tolerance) {
            fprintf(stderr, "  Mismatch at index %d: CPU = %f, GPU = %f "
                            "(diff = %e)\n",
                    i, cpu_output[i], gpu_output[i], diff);
            return 0;
        }
    }
    return 1;
}

/* ======================================================================= */
int main(void) {
    /* -------------------------------------------------------------------
     *  SEED THE RANDOM NUMBER GENERATOR
     * -------------------------------------------------------------------
     *  Without srand(), rand() uses seed=1 by default, producing the
     *  exact same "random" sequence every run.  Seeding with time(NULL)
     *  gives different data each execution.
     * ------------------------------------------------------------------- */
    srand((unsigned int)time(NULL));

    /* -------------------------------------------------------------------
     *  TENSOR DIMENSIONS
     * -------------------------------------------------------------------
     *  We use a realistic deep-learning tensor size:
     *    N=256 images, C=32 channels, H=224, W=224
     *
     *  Total elements: 256 * 32 * 224 * 224 = 411,041,792
     *  Memory:         411,041,792 * 4 bytes ≈ 1.53 GB
     *
     *  This is large enough to saturate GPU bandwidth and see meaningful
     *  timing differences between the two approaches.
     *
     *  NOTE: We need 3 device buffers (input + 2 outputs) = ~4.6 GB
     *  plus 4 host buffers = ~6.1 GB.  Make sure your GPU has enough
     *  memory (an RTX 4070 with 12 GB is fine).
     * ------------------------------------------------------------------- */
    const int batch_size = 256;
    const int channels   = 32;
    const int height     = 224;
    const int width      = 224;
    const int tensor_size = batch_size * channels * height * width;

    /* Use (size_t) cast to prevent overflow in byte calculations.
     * tensor_size (411M) * 4 = 1.64 billion bytes — fits in size_t
     * but would overflow if computed as int * int on some paths. */
    size_t tensor_bytes = (size_t)tensor_size * sizeof(float);

    printf("==========================================================\n");
    printf("  cuDNN Activation (tanh) vs Naive CUDA Kernel\n");
    printf("==========================================================\n\n");
    printf("Tensor shape (NCHW): %d x %d x %d x %d\n",
           batch_size, channels, height, width);
    printf("Total elements:      %d  (%.2f million)\n",
           tensor_size, tensor_size / 1e6);
    printf("Memory per buffer:   %.2f MB\n\n", tensor_bytes / (1024.0 * 1024.0));

    /* -------------------------------------------------------------------
     *  ALLOCATE HOST MEMORY
     * -------------------------------------------------------------------
     *  4 buffers:
     *    h_input         – random input data
     *    h_output_naive  – results from our hand-written kernel
     *    h_output_cudnn  – results from cuDNN
     *    h_output_cpu    – CPU reference for verification
     * ------------------------------------------------------------------- */
    float *h_input        = (float *)malloc(tensor_bytes);
    float *h_output_naive = (float *)malloc(tensor_bytes);
    float *h_output_cudnn = (float *)malloc(tensor_bytes);
    float *h_output_cpu   = (float *)malloc(tensor_bytes);

    if (!h_input || !h_output_naive || !h_output_cudnn || !h_output_cpu) {
        fprintf(stderr, "Failed to allocate host memory (%.2f MB x 4)\n",
                tensor_bytes / (1024.0 * 1024.0));
        return EXIT_FAILURE;
    }

    /* Fill input with random values in [-1, +1] */
    initializeData(h_input, tensor_size);

    /* -------------------------------------------------------------------
     *  ALLOCATE DEVICE MEMORY
     * -------------------------------------------------------------------
     *  3 device buffers:
     *    d_input         – copy of h_input on GPU
     *    d_output_naive  – destination for naive kernel
     *    d_output_cudnn  – destination for cuDNN
     * ------------------------------------------------------------------- */
    float *d_input, *d_output_naive, *d_output_cudnn;
    CHECK_CUDA(cudaMalloc(&d_input,        tensor_bytes));
    CHECK_CUDA(cudaMalloc(&d_output_naive, tensor_bytes));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, tensor_bytes));

    /* Copy input data from host to device */
    CHECK_CUDA(cudaMemcpy(d_input, h_input, tensor_bytes,
                          cudaMemcpyHostToDevice));

    /* -------------------------------------------------------------------
     *  CREATE CUDA EVENTS FOR TIMING
     * -------------------------------------------------------------------
     *  CUDA events measure GPU-side elapsed time in milliseconds.
     *
     *  Usage pattern:
     *    cudaEventRecord(start)       – insert timestamp in GPU stream
     *    << launch kernel >>
     *    cudaEventRecord(stop)        – insert another timestamp
     *    cudaEventSynchronize(stop)   – wait for stop event to complete
     *    cudaEventElapsedTime(&ms, start, stop)  – compute difference
     *
     *  This measures ONLY GPU execution time, not CPU overhead.
     * ------------------------------------------------------------------- */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* -------------------------------------------------------------------
     *  BENCHMARK PARAMETERS
     * -------------------------------------------------------------------
     *  Warmup runs:     Bring the GPU out of idle power state, trigger
     *                   JIT compilation of PTX, warm up L2 cache.
     *  Benchmark runs:  Timed iterations — we average these for a
     *                   stable measurement.
     * ------------------------------------------------------------------- */
    const int num_warmup    = 10;
    const int num_benchmark = 100;
    float naive_times[100];     /* Store each iteration's time (ms) */
    float cudnn_times[100];

    /* =================================================================
     *  BENCHMARK 1:  NAIVE CUDA KERNEL
     * ================================================================= */
    printf("--- Benchmark: Naive CUDA Kernel ---\n");

    /* Launch configuration:
     *   block = 256 threads per block
     *   grid  = ceil(tensor_size / 256)
     *
     *   For 411M elements: grid = 1,605,632 blocks
     *   Total threads launched: 411,041,792 (one per element)  */
    dim3 block(256);
    dim3 grid((tensor_size + block.x - 1) / block.x);

    /* Warmup — not timed */
    for (int i = 0; i < num_warmup; ++i) {
        naiveTanhKernel<<<grid, block>>>(d_input, d_output_naive, tensor_size);
    }

    /* Check for kernel launch errors.
     * Kernel launches are asynchronous — errors don't appear until
     * we synchronize.  cudaGetLastError() catches launch config
     * errors (e.g. too many threads), while cudaDeviceSynchronize()
     * catches execution errors. */
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Timed benchmark runs */
    for (int i = 0; i < num_benchmark; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        naiveTanhKernel<<<grid, block>>>(d_input, d_output_naive, tensor_size);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&naive_times[i], start, stop));
    }

    /* =================================================================
     *  BENCHMARK 2:  cuDNN ACTIVATION
     * =================================================================
     *
     *  STEP 1: Create the cuDNN handle
     *  --------------------------------
     *  The handle is the library context.  It holds internal state
     *  (memory allocators, stream associations, etc.).
     *
     *  One handle per thread is typical.  Creating a handle is
     *  expensive — do it once, reuse across many operations.
     * ----------------------------------------------------------------- */
    printf("--- Benchmark: cuDNN Activation ---\n");

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    /* -----------------------------------------------------------------
     *  STEP 2: Create and set the TENSOR DESCRIPTOR
     * -----------------------------------------------------------------
     *  Tells cuDNN the shape and layout of our input/output tensor.
     *
     *  cudnnSetTensor4dDescriptor(descriptor, format, dataType, N, C, H, W)
     *
     *    descriptor : the descriptor object to fill
     *    format     : CUDNN_TENSOR_NCHW  (N, C, H, W order — default)
     *                 CUDNN_TENSOR_NHWC  (N, H, W, C — used by TF)
     *    dataType   : CUDNN_DATA_FLOAT   (32-bit float)
     *                 CUDNN_DATA_HALF    (16-bit float)
     *                 CUDNN_DATA_DOUBLE  (64-bit float)
     *    N, C, H, W : the tensor dimensions
     *
     *  We use the SAME descriptor for both input and output because
     *  tanh is element-wise (output shape = input shape).
     * ----------------------------------------------------------------- */
    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NCHW,      /* Memory layout: batch, channel, height, width */
        CUDNN_DATA_FLOAT,       /* Data type: 32-bit float                      */
        batch_size,             /* N = 256                                      */
        channels,               /* C = 32                                       */
        height,                 /* H = 224                                      */
        width                   /* W = 224                                      */
    ));

    /* -----------------------------------------------------------------
     *  STEP 3: Create and set the ACTIVATION DESCRIPTOR
     * -----------------------------------------------------------------
     *  cudnnSetActivationDescriptor(descriptor, mode, nanPropagation, coef)
     *
     *    mode : CUDNN_ACTIVATION_SIGMOID
     *           CUDNN_ACTIVATION_RELU
     *           CUDNN_ACTIVATION_TANH       ← we use this
     *           CUDNN_ACTIVATION_CLIPPED_RELU  (needs coef for ceiling)
     *           CUDNN_ACTIVATION_ELU           (needs coef for alpha)
     *           CUDNN_ACTIVATION_IDENTITY
     *
     *    nanPropagation : CUDNN_PROPAGATE_NAN   – NaN in → NaN out
     *                     CUDNN_NOT_PROPAGATE_NAN – NaN in → 0 out
     *
     *    coef : Used by CLIPPED_RELU (ceiling) and ELU (alpha).
     *           Ignored for TANH, SIGMOID, RELU.  We pass 0.0.
     * ----------------------------------------------------------------- */
    cudnnActivationDescriptor_t activation_descriptor;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    CHECK_CUDNN(cudnnSetActivationDescriptor(
        activation_descriptor,
        CUDNN_ACTIVATION_TANH,      /* tanh activation function          */
        CUDNN_PROPAGATE_NAN,        /* propagate NaN values through      */
        0.0                         /* coef — unused for tanh            */
    ));

    /* -----------------------------------------------------------------
     *  STEP 4: Call cudnnActivationForward
     * -----------------------------------------------------------------
     *  cudnnActivationForward(handle,
     *      activationDesc,
     *      &alpha, xDesc, x,       // input side
     *      &beta,  yDesc, y);      // output side
     *
     *  Computes:  y = alpha * tanh(x) + beta * y_old
     *
     *  With alpha=1, beta=0:  y = tanh(x)   (simple overwrite)
     *
     *  NOTE: alpha and beta are HOST pointers to float values.
     *  This is a common cuDNN/cuBLAS convention — the scaling factors
     *  live on the CPU, not the GPU.
     *
     *  NOTE: We reuse input_descriptor for both xDesc and yDesc because
     *  tanh doesn't change the tensor shape.
     * ----------------------------------------------------------------- */
    float alpha = 1.0f;    /* output = 1.0 * tanh(input) ...  */
    float beta  = 0.0f;    /*        + 0.0 * output_old       */

    /* Warmup — not timed */
    for (int i = 0; i < num_warmup; ++i) {
        CHECK_CUDNN(cudnnActivationForward(
            cudnn,
            activation_descriptor,
            &alpha, input_descriptor, d_input,
            &beta,  input_descriptor, d_output_cudnn
        ));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Timed benchmark runs */
    for (int i = 0; i < num_benchmark; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDNN(cudnnActivationForward(
            cudnn,
            activation_descriptor,
            &alpha, input_descriptor, d_input,
            &beta,  input_descriptor, d_output_cudnn
        ));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&cudnn_times[i], start, stop));
    }

    /* =================================================================
     *  COMPUTE AVERAGE TIMES
     * =================================================================
     *  We average over num_benchmark iterations to reduce variance
     *  from OS scheduling, thermal throttling, etc.
     * ================================================================= */
    float avg_naive = 0.0f;
    float avg_cudnn = 0.0f;
    float min_naive = naive_times[0];
    float min_cudnn = cudnn_times[0];

    for (int i = 0; i < num_benchmark; ++i) {
        avg_naive += naive_times[i];
        avg_cudnn += cudnn_times[i];
        if (naive_times[i] < min_naive) min_naive = naive_times[i];
        if (cudnn_times[i] < min_cudnn) min_cudnn = cudnn_times[i];
    }
    avg_naive /= num_benchmark;
    avg_cudnn /= num_benchmark;

    /* =================================================================
     *  VERIFY CORRECTNESS
     * =================================================================
     *  Compute tanh on the CPU for every element, then compare both
     *  GPU outputs against it.
     * ================================================================= */
    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output_naive, tensor_bytes,
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output_cudnn, tensor_bytes,
                          cudaMemcpyDeviceToHost));

    /* CPU reference computation */
    for (int i = 0; i < tensor_size; ++i) {
        h_output_cpu[i] = tanhf(h_input[i]);
    }

    float tolerance = 1e-5f;
    float naive_max_error = 0.0f;
    float cudnn_max_error = 0.0f;

    int naive_correct = verifyResults(h_output_cpu, h_output_naive,
                                      tensor_size, tolerance, &naive_max_error);
    int cudnn_correct = verifyResults(h_output_cpu, h_output_cudnn,
                                      tensor_size, tolerance, &cudnn_max_error);

    /* =================================================================
     *  COMPUTE EFFECTIVE BANDWIDTH
     * =================================================================
     *  Tanh is memory-bound: it reads N floats, writes N floats, and
     *  does minimal compute per element.  The useful metric is
     *  effective memory bandwidth (GB/s), not GFLOPS.
     *
     *  Bandwidth = (bytes_read + bytes_written) / time_seconds / 10^9
     *            = (2 * tensor_size * 4) / (time_ms * 10^-3) / 10^9
     *            = (2 * tensor_bytes) / (time_ms * 10^6)     [GB/s]
     * ================================================================= */
    double total_bytes = 2.0 * tensor_bytes;    /* read + write */
    double naive_bw = total_bytes / (avg_naive * 1e6);  /* GB/s */
    double cudnn_bw = total_bytes / (avg_cudnn * 1e6);

    /* =================================================================
     *  PRINT RESULTS
     * ================================================================= */
    printf("\n");
    printf("==========================================================\n");
    printf("  RESULTS  (averaged over %d runs, %d warmup)\n", num_benchmark, num_warmup);
    printf("==========================================================\n");
    printf("\n");
    printf("  %-24s  %12s  %12s\n", "", "Naive Kernel", "cuDNN");
    printf("  --------------------------------------------------------\n");
    printf("  %-24s  %10.3f ms  %10.3f ms\n", "Average time",  avg_naive, avg_cudnn);
    printf("  %-24s  %10.3f ms  %10.3f ms\n", "Best time",     min_naive, min_cudnn);
    printf("  %-24s  %9.2f GB/s  %9.2f GB/s\n","Eff. bandwidth", naive_bw,  cudnn_bw);
    printf("  %-24s  %12s  %10.2fx\n",         "Speedup vs naive", "1.00x",
           avg_naive / avg_cudnn);
    printf("  %-24s  %12s  %12s\n", "Correct",
           naive_correct ? "YES" : "NO",
           cudnn_correct ? "YES" : "NO");
    printf("  %-24s  %12.2e  %12.2e\n", "Max error vs CPU",
           naive_max_error, cudnn_max_error);
    printf("  --------------------------------------------------------\n");
    printf("\n");

    /* =================================================================
     *  CLEANUP
     * =================================================================
     *  Order: device resources first, then host.
     *  Within each category, destroy in reverse order of creation.
     *
     *  cuDNN descriptors must be destroyed before the handle:
     *    activation_descriptor → input_descriptor → cudnn handle
     * ================================================================= */

    /* Device memory */
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_naive));
    CHECK_CUDA(cudaFree(d_output_cudnn));

    /* CUDA events */
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    /* cuDNN descriptors (reverse order of creation) */
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activation_descriptor));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor));

    /* cuDNN handle */
    CHECK_CUDNN(cudnnDestroy(cudnn));

    /* Host memory */
    free(h_input);
    free(h_output_naive);
    free(h_output_cudnn);
    free(h_output_cpu);

    return 0;
}
