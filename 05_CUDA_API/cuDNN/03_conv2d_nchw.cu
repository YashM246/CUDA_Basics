/*
 * ==========================================================================
 *  03_conv2d_nchw.cu  –  cuDNN Convolution Forward vs Naive CUDA Kernel
 * ==========================================================================
 *
 *  WHAT THIS PROGRAM DOES
 *  ----------------------
 *  Performs 2D convolution on a small 4x4 image with a 3x3 kernel, using:
 *    1. A naive CUDA kernel  (brute-force, one thread per output pixel)
 *    2. cuDNN's cudnnConvolutionForward  (optimized library call)
 *
 *  Uses small, human-readable values so you can verify the math by hand.
 *  Both outputs are printed and compared.
 *
 *  WHAT IS CONVOLUTION? (in Deep Learning)
 *  ----------------------------------------
 *  A convolution slides a small filter (kernel) over an input image,
 *  computing a weighted sum at each position:
 *
 *    Input (4x4):          Kernel (3x3):       Output (4x4, "same" pad):
 *    ┌──┬──┬──┬──┐         ┌──┬──┬──┐
 *    │ 1│ 2│ 3│ 4│         │ 1│ 2│ 3│          The kernel slides over
 *    ├──┼──┼──┼──┤         ├──┼──┼──┤          every position of the
 *    │ 5│ 6│ 7│ 8│         │ 4│ 5│ 6│          input, computing the
 *    ├──┼──┼──┼──┤         ├──┼──┼──┤          dot product at each
 *    │ 9│10│11│12│         │ 7│ 8│ 9│          location.
 *    ├──┼──┼──┼──┤         └──┴──┴──┘
 *    │13│14│15│16│
 *    └──┴──┴──┴──┘
 *
 *  For position (1,1) — kernel centered on input[1][1] = 6:
 *
 *    1*1 + 2*2 + 3*3       = 14
 *    4*5 + 5*6 + 6*7       = 92
 *    7*9 + 8*10 + 9*11     = 242
 *                      sum = 348   ← output[1][1]
 *
 *  CONVOLUTION vs CROSS-CORRELATION
 *  ----------------------------------
 *  True mathematical convolution FLIPS the kernel (rotates 180°) before
 *  sliding.  Cross-correlation does NOT flip — it uses the kernel as-is.
 *
 *  In deep learning, we ALWAYS use cross-correlation (the weights are
 *  learned anyway, so flipping doesn't matter).  cuDNN calls this
 *  CUDNN_CROSS_CORRELATION.  Despite the name "convolution layer" in
 *  neural networks, it's technically cross-correlation.
 *
 *  "SAME" PADDING
 *  ---------------
 *  Without padding, a 3x3 kernel on a 4x4 input produces a 2x2 output
 *  (the kernel can only fit where all 9 weights overlap the input).
 *
 *  "Same" padding adds zeros around the border so the output size equals
 *  the input size:
 *    padding = kernelSize / 2 = 3 / 2 = 1  (one row/col of zeros on each side)
 *
 *    ┌──┬──┬──┬──┬──┬──┐
 *    │ 0│ 0│ 0│ 0│ 0│ 0│  ← padded input (6x6)
 *    ├──┼──┼──┼──┼──┼──┤
 *    │ 0│ 1│ 2│ 3│ 4│ 0│     The kernel can now center on every
 *    ├──┼──┼──┼──┼──┼──┤     original pixel, including edges.
 *    │ 0│ 5│ 6│ 7│ 8│ 0│
 *    ├──┼──┼──┼──┼──┼──┤     Output: still 4x4 (same as input).
 *    │ 0│ 9│10│11│12│ 0│
 *    ├──┼──┼──┼──┼──┼──┤
 *    │ 0│13│14│15│16│ 0│
 *    ├──┼──┼──┼──┼──┼──┤
 *    │ 0│ 0│ 0│ 0│ 0│ 0│
 *    └──┴──┴──┴──┴──┴──┘
 *
 *  NEW cuDNN CONCEPTS (beyond 01_tanh.cu)
 *  ----------------------------------------
 *
 *  1. cudnnFilterDescriptor_t
 *     Like a tensor descriptor but specifically for convolution filters.
 *     Shape is (K, C, R, S) = (outChannels, inChannels, kernelH, kernelW).
 *
 *  2. cudnnConvolutionDescriptor_t
 *     Describes the convolution operation itself:
 *       - padding (how many zeros to add)
 *       - stride  (how far to move the kernel each step)
 *       - dilation (gaps between kernel elements)
 *       - mode    (CONVOLUTION vs CROSS_CORRELATION)
 *
 *  3. cudnnGetConvolutionForwardAlgorithm_v7
 *     cuDNN has MULTIPLE algorithms for convolution (GEMM, Winograd, FFT,
 *     etc.).  This function benchmarks them all and returns timing results
 *     sorted fastest-first.
 *
 *  4. Workspace memory
 *     Some convolution algorithms need temporary scratch memory.
 *     You query the size with cudnnGetConvolutionForwardWorkspaceSize,
 *     then cudaMalloc it yourself.
 *
 *  5. cudnnConvolutionForward  (13 parameters!)
 *     The main operation.  See annotated call below.
 *
 *  MULTI-CHANNEL CONVOLUTION
 *  --------------------------
 *  In a real CNN, convolution works across channels:
 *
 *    Input:   (N, C_in, H, W)    e.g. (1, 3, 224, 224)  — RGB image
 *    Filter:  (C_out, C_in, R, S) e.g. (64, 3, 3, 3)    — 64 filters
 *    Output:  (N, C_out, H, W)   e.g. (1, 64, 224, 224)  — 64 feature maps
 *
 *  Each output channel is produced by one filter sliding over ALL input
 *  channels and summing the results:
 *
 *    output[n][k][h][w] = SUM over c of (filter[k][c] * input[n][c])
 *
 *  COMPILE
 *  --------
 *  nvcc 03_conv2d_nchw.cu -o 03_conv2d_nchw -lcudnn
 * ==========================================================================
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* -----------------------------------------------------------------------
 *  ERROR-CHECKING MACROS
 * -----------------------------------------------------------------------
 *  Wrapped in do { ... } while(0) for safe use inside if/else chains.
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
 *  NAIVE 2D CONVOLUTION KERNEL
 * -----------------------------------------------------------------------
 *  Performs cross-correlation (no kernel flip) with "same" padding
 *  (zero-padding so output size = input size).
 *
 *  Thread mapping:
 *    x, y       → output pixel position (width, height)
 *    blockIdx.z → encodes BOTH the output channel AND the batch index
 *                 outChannel = blockIdx.z % outChannels
 *                 batchIdx   = blockIdx.z / outChannels
 *
 *  For each output pixel, we loop over:
 *    - All input channels (C_in)
 *    - All kernel rows and columns (R x S)
 *  and accumulate the weighted sum.
 *
 *  NCHW INDEX CALCULATION
 *  -----------------------
 *  For a 4-D tensor in NCHW layout, the linear index is:
 *
 *    index = ((n * C + c) * H + h) * W + w
 *
 *  This comes from the memory layout:
 *    - n selects the batch (stride = C*H*W)
 *    - c selects the channel (stride = H*W)
 *    - h selects the row (stride = W)
 *    - w selects the column (stride = 1)
 *
 *  FILTER INDEX (K, C, R, S):
 *    index = ((k * C_in + c) * R + r) * S + s
 * ----------------------------------------------------------------------- */
__global__ void naiveConv2d(float *input, float *kernel, float *output,
                            int width, int height,
                            int inChannels, int outChannels,
                            int kernelSize, int batchSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   /* output column */
    int y = blockIdx.y * blockDim.y + threadIdx.y;   /* output row    */
    int outChannel = blockIdx.z % outChannels;       /* which filter  */
    int batchIdx   = blockIdx.z / outChannels;       /* which image   */

    if (x < width && y < height && outChannel < outChannels && batchIdx < batchSize) {
        float sum = 0.0f;
        int halfKernel = kernelSize / 2;

        /* Loop over all input channels.
         * Each output pixel aggregates information from EVERY input channel. */
        for (int inChannel = 0; inChannel < inChannels; inChannel++) {

            /* Loop over kernel rows and columns.
             * ky, kx range from -halfKernel to +halfKernel.
             * For a 3x3 kernel: ky, kx ∈ {-1, 0, +1} */
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {

                    /* Compute the input position this kernel element maps to.
                     * "Same" padding: if (ix, iy) is outside the image,
                     * we treat it as zero (skip the accumulation). */
                    int ix = x + kx;
                    int iy = y + ky;

                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        /* NCHW index into input: ((n*C + c)*H + h)*W + w */
                        int inputIdx = ((batchIdx * inChannels + inChannel) * height + iy) * width + ix;

                        /* Filter index: ((k*C_in + c)*R + r)*S + s
                         * r = ky + halfKernel maps [-1,0,1] → [0,1,2]
                         * s = kx + halfKernel maps [-1,0,1] → [0,1,2] */
                        int kernelIdx = ((outChannel * inChannels + inChannel) * kernelSize
                                         + (ky + halfKernel)) * kernelSize + (kx + halfKernel);

                        /* BUG FIX: Original code had `inputIdx * kernel[kernelIdx]`
                         * which multiplied the INDEX (an integer!) by the weight,
                         * instead of the VALUE at that index.
                         * Correct: input[inputIdx] * kernel[kernelIdx] */
                        sum += input[inputIdx] * kernel[kernelIdx];
                    }
                }
            }
        }

        /* BUG FIX: Original code had `height * y` instead of `height + y`
         * in the output index.  This caused:
         *   ((batchIdx * outChannels + outChannel) * height * y) * width + x
         * which is completely wrong.  Correct NCHW formula:
         *   ((n * C + c) * H + h) * W + w */
        int outputIdx = ((batchIdx * outChannels + outChannel) * height + y) * width + x;
        output[outputIdx] = sum;
    }
}

/* ======================================================================= */
int main(void) {
    /* -------------------------------------------------------------------
     *  PROBLEM DIMENSIONS
     * -------------------------------------------------------------------
     *  Small values so we can print and verify the output by hand.
     *
     *    Input:  1 batch, 1 channel, 4x4 pixels   (16 values)
     *    Kernel: 1 output channel, 1 input channel, 3x3  (9 values)
     *    Output: 1 batch, 1 channel, 4x4 pixels   (16 values, same padding)
     * ------------------------------------------------------------------- */
    const int width       = 4;
    const int height      = 4;
    const int kernelSize  = 3;
    const int inChannels  = 1;
    const int outChannels = 1;
    const int batchSize   = 1;

    const int inputSize      = batchSize * inChannels  * height * width;
    const int outputSize     = batchSize * outChannels * height * width;
    const int kernelElements = outChannels * inChannels * kernelSize * kernelSize;

    printf("==========================================================\n");
    printf("  cuDNN Convolution Forward vs Naive CUDA Kernel\n");
    printf("==========================================================\n\n");
    printf("Input shape  (NCHW): %d x %d x %d x %d\n", batchSize, inChannels, height, width);
    printf("Kernel shape (KCRS): %d x %d x %d x %d\n", outChannels, inChannels, kernelSize, kernelSize);
    printf("Output shape (NCHW): %d x %d x %d x %d\n", batchSize, outChannels, height, width);
    printf("Padding: %d  (\"same\" — output size = input size)\n", kernelSize / 2);
    printf("Stride:  1\n\n");

    /* -------------------------------------------------------------------
     *  ALLOCATE HOST MEMORY
     * ------------------------------------------------------------------- */
    float *h_input        = (float *)malloc(inputSize      * sizeof(float));
    float *h_kernel       = (float *)malloc(kernelElements * sizeof(float));
    float *h_output_cudnn = (float *)malloc(outputSize     * sizeof(float));
    float *h_output_naive = (float *)malloc(outputSize     * sizeof(float));

    if (!h_input || !h_kernel || !h_output_cudnn || !h_output_naive) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* -------------------------------------------------------------------
     *  INITIALIZE INPUT AND KERNEL
     * -------------------------------------------------------------------
     *  Using simple sequential values so the output is easy to verify.
     *
     *  Input (4x4):         Kernel (3x3):
     *   1   2   3   4        1  2  3
     *   5   6   7   8        4  5  6
     *   9  10  11  12        7  8  9
     *  13  14  15  16
     *
     *  Example output[1][1] (kernel centered on 6):
     *    1*1 + 2*2 + 3*3 + 5*4 + 6*5 + 7*6 + 9*7 + 10*8 + 11*9
     *    = 1 + 4 + 9 + 20 + 30 + 42 + 63 + 80 + 99
     *    = 348
     * ------------------------------------------------------------------- */
    float input_values[] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    float kernel_values[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    memcpy(h_input,  input_values,  inputSize      * sizeof(float));
    memcpy(h_kernel, kernel_values, kernelElements * sizeof(float));

    /* -------------------------------------------------------------------
     *  ALLOCATE DEVICE MEMORY
     * ------------------------------------------------------------------- */
    float *d_input, *d_kernel, *d_output_cudnn, *d_output_naive;
    CHECK_CUDA(cudaMalloc(&d_input,        inputSize      * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel,       kernelElements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, outputSize     * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, outputSize     * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input,  h_input,  inputSize      * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernelElements * sizeof(float), cudaMemcpyHostToDevice));

    /* =================================================================
     *  cuDNN CONVOLUTION SETUP
     * =================================================================
     *  Convolution requires 4 descriptors:
     *
     *    1. cudnnTensorDescriptor_t  inputDesc   – input tensor shape
     *    2. cudnnFilterDescriptor_t  filterDesc  – kernel/filter shape
     *    3. cudnnConvolutionDescriptor_t convDesc – operation parameters
     *    4. cudnnTensorDescriptor_t  outputDesc  – output tensor shape
     *
     *  Plus algorithm selection and workspace allocation.
     * ================================================================= */

    /* STEP 1: Create the cuDNN handle */
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    /* -----------------------------------------------------------------
     *  STEP 2: CREATE descriptors (must come before SET)
     * -----------------------------------------------------------------
     *  BUG FIX: The original code called cudnnSetTensor4dDescriptor
     *  WITHOUT first calling cudnnCreateTensorDescriptor.  Descriptors
     *  are opaque objects — you must CREATE them to allocate the
     *  internal struct, then SET their values.
     *
     *  Create → Set → Use → Destroy
     * ----------------------------------------------------------------- */
    cudnnTensorDescriptor_t     inputDesc, outputDesc;
    cudnnFilterDescriptor_t     filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    /* -----------------------------------------------------------------
     *  STEP 3: SET descriptor values
     * ----------------------------------------------------------------- */

    /* Input tensor descriptor: (N, C, H, W) = (1, 1, 4, 4) */
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        inputDesc,
        CUDNN_TENSOR_NCHW,      /* Layout                    */
        CUDNN_DATA_FLOAT,       /* Data type                 */
        batchSize,              /* N = 1                     */
        inChannels,             /* C = 1                     */
        height,                 /* H = 4                     */
        width                   /* W = 4                     */
    ));

    /* Output tensor descriptor: (N, K, H, W) = (1, 1, 4, 4)
     * Output height/width = input because we use "same" padding */
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        outputDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        outChannels,
        height,
        width
    ));

    /* -----------------------------------------------------------------
     *  FILTER DESCRIPTOR (NEW — not used in 01_tanh.cu)
     * -----------------------------------------------------------------
     *  cudnnSetFilter4dDescriptor(desc, dataType, format, K, C, H, W)
     *
     *  Unlike a tensor descriptor, filters use (K, C, R, S) ordering:
     *    K = number of output channels (number of filters)
     *    C = number of input channels  (must match input tensor)
     *    R = kernel height
     *    S = kernel width
     *
     *  Why a separate type?  Filters may have special memory layouts
     *  for performance (e.g., pre-transposed for GEMM algorithms).
     *  Using cudnnFilterDescriptor_t lets cuDNN optimize internally.
     * ----------------------------------------------------------------- */
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filterDesc,
        CUDNN_DATA_FLOAT,       /* Data type                      */
        CUDNN_TENSOR_NCHW,      /* Layout (NCHW = KCRS for filter)*/
        outChannels,            /* K = 1 (number of filters)      */
        inChannels,             /* C = 1 (input channels)         */
        kernelSize,             /* R = 3 (kernel height)          */
        kernelSize              /* S = 3 (kernel width)           */
    ));

    /* -----------------------------------------------------------------
     *  CONVOLUTION DESCRIPTOR (NEW — not used in 01_tanh.cu)
     * -----------------------------------------------------------------
     *  cudnnSetConvolution2dDescriptor(desc,
     *      pad_h, pad_w,           // zero-padding on each side
     *      stride_h, stride_w,     // step size of the kernel
     *      dilation_h, dilation_w, // gaps between kernel elements
     *      mode,                   // CONVOLUTION or CROSS_CORRELATION
     *      computeType)            // data type for accumulation
     *
     *  Parameters explained:
     *
     *    Padding (1, 1):
     *      Adds 1 row/col of zeros on each side.  With kernelSize=3,
     *      this gives "same" output size (output H = input H).
     *      Formula: padding = kernelSize / 2
     *
     *    Stride (1, 1):
     *      Move the kernel 1 pixel at a time.  stride=2 would skip
     *      every other position, halving the output size.
     *
     *    Dilation (1, 1):
     *      No gaps between kernel elements.  dilation=2 would space
     *      out kernel weights with zeros between them, increasing
     *      the effective receptive field without more parameters.
     *
     *      Dilation=1 (normal):    Dilation=2 (dilated):
     *        [1 2 3]                [1 0 2 0 3]
     *        [4 5 6]                [0 0 0 0 0]
     *        [7 8 9]                [4 0 5 0 6]
     *                               [0 0 0 0 0]
     *                               [7 0 8 0 9]
     *
     *    Mode:
     *      CUDNN_CROSS_CORRELATION – no kernel flip (standard in DL)
     *      CUDNN_CONVOLUTION      – flips kernel 180° (math definition)
     * ----------------------------------------------------------------- */
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        convDesc,
        kernelSize / 2,         /* pad_h = 1      ("same" padding)     */
        kernelSize / 2,         /* pad_w = 1                           */
        1,                      /* stride_h = 1   (no downsampling)    */
        1,                      /* stride_w = 1                        */
        1,                      /* dilation_h = 1 (no dilation)        */
        1,                      /* dilation_w = 1                      */
        CUDNN_CROSS_CORRELATION,/* Standard in deep learning           */
        CUDNN_DATA_FLOAT        /* Accumulation data type              */
    ));

    /* -----------------------------------------------------------------
     *  STEP 4: ALGORITHM SELECTION (NEW — not used in 01_tanh.cu)
     * -----------------------------------------------------------------
     *  cuDNN offers multiple algorithms for convolution:
     *
     *    | Algorithm              | Strategy          | Memory   |
     *    |------------------------|-------------------|----------|
     *    | IMPLICIT_GEMM          | Im2col + GEMM     | Low      |
     *    | IMPLICIT_PRECOMP_GEMM  | Precomputed im2col| Medium   |
     *    | GEMM                   | Explicit GEMM     | High     |
     *    | WINOGRAD               | Winograd transform| Medium   |
     *    | FFT                    | FFT-based         | High     |
     *
     *  cudnnGetConvolutionForwardAlgorithm_v7 benchmarks them and
     *  returns results SORTED by execution time (fastest first).
     *  So perfResults[0] is always the best algorithm.
     *
     *  Note: The "_v7" suffix is the API version, not cuDNN 7.
     *  This function exists in cuDNN 8/9 as well.
     * ----------------------------------------------------------------- */
    int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        inputDesc,
        filterDesc,
        convDesc,
        outputDesc,
        requestedAlgoCount,     /* How many algorithms to try     */
        &returnedAlgoCount,     /* How many succeeded             */
        perfResults             /* Array of results (sorted)      */
    ));

    /* Print all tested algorithms */
    printf("--- Algorithm Selection ---\n");
    for (int i = 0; i < returnedAlgoCount; i++) {
        printf("  Algo %d: time = %.4f ms, memory = %zu bytes, status = %s\n",
               perfResults[i].algo,
               perfResults[i].time,
               perfResults[i].memory,
               perfResults[i].status == CUDNN_STATUS_SUCCESS ? "OK" : "FAIL");
    }

    /* perfResults[0] is already the fastest (sorted by time).
     * No need for a manual search loop. */
    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    printf("  Selected: algo %d (fastest)\n\n", algo);

    /* -----------------------------------------------------------------
     *  STEP 5: ALLOCATE WORKSPACE (NEW — not used in 01_tanh.cu)
     * -----------------------------------------------------------------
     *  Some algorithms need temporary GPU memory (scratch space).
     *  For example, GEMM-based algorithms need space to store the
     *  "im2col" matrix (the input rearranged into columns).
     *
     *  We ask cuDNN how much workspace the selected algorithm needs,
     *  then allocate it ourselves with cudaMalloc.
     *
     *  If workspaceSize = 0, the algorithm doesn't need extra memory
     *  (e.g., IMPLICIT_GEMM computes im2col on-the-fly).
     * ----------------------------------------------------------------- */
    size_t workspaceSize;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        inputDesc,
        filterDesc,
        convDesc,
        outputDesc,
        algo,
        &workspaceSize
    ));

    printf("Workspace size: %zu bytes\n\n", workspaceSize);

    void *d_workspace = NULL;
    if (workspaceSize > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));
    }

    /* =================================================================
     *  BENCHMARK SETUP
     * ================================================================= */
    dim3 block(16, 16);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        outChannels * batchSize     /* z-dim encodes channel + batch */
    );

    const int warmupRuns    = 5;
    const int benchmarkRuns = 20;

    float alpha = 1.0f;    /* output = 1.0 * conv(input, kernel) ...  */
    float beta  = 0.0f;    /*        + 0.0 * output_old               */

    /* -------------------------------------------------------------------
     *  WARMUP
     * ------------------------------------------------------------------- */
    for (int i = 0; i < warmupRuns; i++) {
        /* ---------------------------------------------------------------
         *  cudnnConvolutionForward  (13 PARAMETERS)
         * ---------------------------------------------------------------
         *  cudnnConvolutionForward(
         *      handle,          // Library context
         *      &alpha,          // Scale for result: output = alpha * conv(...)
         *      inputDesc,       // Input tensor descriptor
         *      d_input,         // Input tensor data (device memory)
         *      filterDesc,      // Filter descriptor
         *      d_kernel,        // Filter weights (device memory)
         *      convDesc,        // Convolution parameters (pad, stride, dilation)
         *      algo,            // Which algorithm to use
         *      d_workspace,     // Temporary workspace (device memory, or NULL)
         *      workspaceSize,   // Size of workspace in bytes
         *      &beta,           // Scale for existing output: + beta * output_old
         *      outputDesc,      // Output tensor descriptor
         *      d_output_cudnn   // Output tensor data (device memory)
         *  );
         *
         *  Computes: output = alpha * conv(input, filter) + beta * output_old
         *  With alpha=1, beta=0: output = conv(input, filter)
         * --------------------------------------------------------------- */
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn, &alpha,
            inputDesc,  d_input,
            filterDesc, d_kernel,
            convDesc, algo, d_workspace, workspaceSize,
            &beta,
            outputDesc, d_output_cudnn
        ));

        naiveConv2d<<<grid, block>>>(d_input, d_kernel, d_output_naive,
                                     width, height, inChannels, outChannels,
                                     kernelSize, batchSize);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* -------------------------------------------------------------------
     *  TIMED BENCHMARK
     * ------------------------------------------------------------------- */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float totalTime_cudnn = 0.0f;
    float totalTime_naive = 0.0f;

    for (int i = 0; i < benchmarkRuns; i++) {
        float ms = 0.0f;

        /* Time cuDNN */
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn, &alpha,
            inputDesc,  d_input,
            filterDesc, d_kernel,
            convDesc, algo, d_workspace, workspaceSize,
            &beta,
            outputDesc, d_output_cudnn
        ));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        totalTime_cudnn += ms;

        /* Time naive kernel */
        CHECK_CUDA(cudaEventRecord(start));
        naiveConv2d<<<grid, block>>>(d_input, d_kernel, d_output_naive,
                                     width, height, inChannels, outChannels,
                                     kernelSize, batchSize);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        totalTime_naive += ms;
    }

    float avgTime_cudnn = totalTime_cudnn / benchmarkRuns;
    float avgTime_naive = totalTime_naive / benchmarkRuns;

    /* =================================================================
     *  COPY RESULTS AND VERIFY
     * ================================================================= */
    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output_cudnn,
                          outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output_naive,
                          outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    /* Compare cuDNN vs naive kernel */
    float maxDiff = 0.0f;
    for (int i = 0; i < outputSize; i++) {
        float diff = fabsf(h_output_cudnn[i] - h_output_naive[i]);
        if (diff > maxDiff) maxDiff = diff;
    }

    /* =================================================================
     *  PRINT RESULTS
     * ================================================================= */
    printf("--- Timing ---\n");
    printf("  cuDNN avg:   %.6f ms\n", avgTime_cudnn);
    printf("  Naive avg:   %.6f ms\n", avgTime_naive);
    printf("  Speedup:     %.2fx\n\n", avgTime_naive / avgTime_cudnn);

    printf("Max difference (cuDNN vs naive): %e\n\n", maxDiff);

    /* Print cuDNN output in 2D grid format */
    printf("cuDNN Output:\n");
    for (int b = 0; b < batchSize; b++) {
        for (int c = 0; c < outChannels; c++) {
            printf("  Batch %d, Channel %d:\n", b, c);
            for (int h = 0; h < height; h++) {
                printf("    ");
                for (int w = 0; w < width; w++) {
                    int idx = ((b * outChannels + c) * height + h) * width + w;
                    printf("%8.1f", h_output_cudnn[idx]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    /* Print naive kernel output */
    printf("Naive Kernel Output:\n");
    for (int b = 0; b < batchSize; b++) {
        for (int c = 0; c < outChannels; c++) {
            printf("  Batch %d, Channel %d:\n", b, c);
            for (int h = 0; h < height; h++) {
                printf("    ");
                for (int w = 0; w < width; w++) {
                    int idx = ((b * outChannels + c) * height + h) * width + w;
                    printf("%8.1f", h_output_naive[idx]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    /* Print flattened output for verification against PyTorch:
     *   import torch.nn.functional as F
     *   x = torch.tensor([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]],
     *                     dtype=torch.float32)
     *   k = torch.tensor([[[[1,2,3],[4,5,6],[7,8,9]]]], dtype=torch.float32)
     *   print(F.conv2d(x, k, padding=1))
     */
    printf("Flattened cuDNN Output (for PyTorch comparison):\n  [");
    for (int i = 0; i < outputSize; i++) {
        printf("%.1f", h_output_cudnn[i]);
        if (i < outputSize - 1) printf(", ");
    }
    printf("]\n\n");

    /* =================================================================
     *  CLEANUP
     * =================================================================
     *  Destroy in reverse order of creation.
     *  Descriptors first, then handle, then device memory.
     * ================================================================= */
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output_cudnn));
    CHECK_CUDA(cudaFree(d_output_naive));
    if (d_workspace) CHECK_CUDA(cudaFree(d_workspace));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_input);
    free(h_kernel);
    free(h_output_cudnn);
    free(h_output_naive);

    return 0;
}
