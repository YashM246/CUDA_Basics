/*
 * Matrix Multiplication with NVTX Profiling Annotations
 * ======================================================
 *
 * What this program demonstrates:
 *   How to use NVTX (NVIDIA Tools Extension) to annotate CUDA code
 *   for better visibility in profiling tools like Nsight Systems.
 *
 * ===========================================
 * What is NVTX?
 * ===========================================
 *
 *   NVTX is a lightweight library that lets you add named markers and
 *   ranges to your code. When you profile with Nsight Systems or Nsight
 *   Compute, these annotations appear on the timeline, making it easy
 *   to see which phase of your algorithm is executing.
 *
 *   Without NVTX:
 *     Timeline shows: [cudaMalloc][cudaMemcpy][kernel][cudaMemcpy][cudaFree]
 *     Hard to tell what each operation represents in your algorithm.
 *
 *   With NVTX:
 *     Timeline shows: [Memory Allocation][Copy H2D][Kernel Execution][Copy D2H]
 *     Clear, human-readable labels for each phase.
 *
 * ===========================================
 * Key NVTX Functions:
 * ===========================================
 *
 *   nvtxRangePushA("name") - Start a named range (ASCII string, can be nested)
 *   nvtxRangePop()         - End the most recent range
 *   nvtxMarkA("event")     - Mark a single point in time (ASCII string)
 *
 *   Note: On Windows, use the 'A' suffix (nvtxRangePushA) for ASCII strings.
 *   Without the suffix, Windows defaults to wide strings (wchar_t).
 *
 *   Ranges can be nested like parentheses:
 *     nvtxRangePushA("Outer");
 *       nvtxRangePushA("Inner");
 *       nvtxRangePop();  // Ends "Inner"
 *     nvtxRangePop();    // Ends "Outer"
 *
 * ===========================================
 * Performance Impact:
 * ===========================================
 *
 *   NVTX has ZERO overhead when profiling tools aren't attached.
 *   The calls become no-ops, so you can leave them in production code.
 *   When profiling IS active, overhead is minimal (~microseconds).
 *
 * ===========================================
 * How to Profile:
 * ===========================================
 *
 *   1. Compile (NVTX3 is header-only, no library needed):
 *      nvcc 01_nvtx_matmul.cu -o 01_nvtx_matmul.exe
 *
 *   2. Profile with Nsight Systems:
 *      nsys profile --stats=true ./01_nvtx_matmul.exe
 *
 *   3. Open the generated .nsys-rep file in Nsight Systems GUI
 *      to see the annotated timeline.
 *
 *   Alternative (Nsight Compute for kernel analysis):
 *      ncu --set full ./01_nvtx_matmul.exe
 *
 * Compile: nvcc 01_nvtx_matmul.cu -o 01_nvtx_matmul.exe
 * Run:     .\01_nvtx_matmul.exe
 * Profile: nsys profile ./01_nvtx_matmul.exe
 */

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>   // NVTX header for profiling annotations
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Block size: 16×16 = 256 threads per block
// Smaller than 32×32 to leave room for register usage in more complex kernels
#define BLOCK_SIZE 16

/*
 * Matrix Multiplication Kernel (Square Matrices)
 *
 * Each thread computes one element C[row][col] = dot(A[row][:], B[:][col])
 * This is the naive global memory approach (same as 03_matmul.cu).
 *
 * Parameters:
 *   A, B - Input matrices (N × N) in GPU global memory
 *   C    - Output matrix (N × N) in GPU global memory
 *   N    - Matrix dimension (square)
 */
__global__ void matrixMulKernel(float *A, float *B, float *C, int N){
    // Calculate which element this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check (grid may be larger than matrix)
    if(row < N && col < N){
        float sum = 0.0f;

        // Dot product of row from A and column from B
        for(int i = 0; i < N; i++){
            sum += A[row * N + i] * B[i * N + col];
        }

        C[row * N + col] = sum;
    }
}

/*
 * Matrix Multiplication Wrapper with NVTX Annotations
 *
 * This function wraps the entire matmul operation with NVTX ranges
 * so each phase is clearly visible in the profiler:
 *
 *   [Matrix Multiplication (outer range)]
 *     ├─ [Memory Allocation]
 *     ├─ [Memory Copy H2D]
 *     ├─ [Kernel Execution]
 *     ├─ [Memory Copy D2H]
 *     └─ [Memory Deallocation]
 *
 * Parameters:
 *   A, B - Input matrices on host (CPU memory)
 *   C    - Output matrix on host (CPU memory)
 *   N    - Matrix dimension
 */
void matrixMul(float *A, float *B, float *C, int N){
    // =========================================
    // Start the outer NVTX range
    // This encompasses the entire operation
    // =========================================
    nvtxRangePushA("Matrix Multiplication");

    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // =========================================
    // Phase 1: Allocate GPU memory
    // =========================================
    // nvtxRangePush starts a nested range inside "Matrix Multiplication"
    nvtxRangePushA("Memory Allocation");
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    nvtxRangePop();   // End "Memory Allocation"

    // =========================================
    // Phase 2: Copy input data to GPU
    // =========================================
    // H2D = Host to Device (CPU → GPU)
    nvtxRangePushA("Memory Copy H2D");
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    nvtxRangePop();   // End "Memory Copy H2D"

    // =========================================
    // Phase 3: Launch and execute kernel
    // =========================================
    // Configure grid dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);   // 16×16 = 256 threads

    // Ceiling division: ceil(N / BLOCK_SIZE) for both dimensions
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    dim3 numBlocks(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    nvtxRangePushA("Kernel Execution");
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();   // Wait for kernel to complete
    nvtxRangePop();   // End "Kernel Execution"

    // =========================================
    // Phase 4: Copy results back to CPU
    // =========================================
    // D2H = Device to Host (GPU → CPU)
    nvtxRangePushA("Memory Copy D2H");
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    nvtxRangePop();   // End "Memory Copy D2H"

    // =========================================
    // Phase 5: Free GPU memory
    // =========================================
    nvtxRangePushA("Memory Deallocation");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();   // End "Memory Deallocation"

    // =========================================
    // End the outer NVTX range
    // =========================================
    nvtxRangePop();   // End "Matrix Multiplication"
}

/*
 * Initialize matrix with random values in [0, 1]
 */
void initMatrix(float *mat, int N){
    for(int i = 0; i < N * N; i++){
        mat[i] = (float)rand() / RAND_MAX;
    }
}

/*
 * CPU matrix multiplication for verification
 */
void matrixMulCPU(float *A, float *B, float *C, int N){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            float sum = 0.0f;
            for(int k = 0; k < N; k++){
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(){
    // Matrix size: 1024 × 1024
    // Total elements: ~1 million
    // Memory per matrix: 4 MB
    const int N = 1024;
    size_t size = N * N * sizeof(float);

    printf("=== NVTX Profiling Demo: Matrix Multiplication ===\n\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Memory per matrix: %.2f MB\n", size / (1024.0 * 1024.0));
    printf("Total operations: %.2f GFLOPs\n\n", (2.0 * N * N * N) / 1e9);

    // =========================================
    // Allocate host memory
    // =========================================
    // Using nvtxRangePush/Pop in main() to annotate host operations too
    nvtxRangePushA("Host Memory Allocation");
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C_gpu = new float[N * N];
    float *C_cpu = new float[N * N];
    nvtxRangePop();

    // =========================================
    // Initialize matrices
    // =========================================
    nvtxRangePushA("Matrix Initialization");
    srand(42);
    initMatrix(A, N);
    initMatrix(B, N);
    nvtxRangePop();

    // =========================================
    // Run GPU matrix multiplication
    // =========================================
    // The matrixMul function has its own NVTX annotations inside
    printf("Running GPU matrix multiplication...\n");
    matrixMul(A, B, C_gpu, N);

    // =========================================
    // Verify with CPU (optional, slow)
    // =========================================
    printf("Running CPU verification...\n");
    nvtxRangePushA("CPU Verification");
    matrixMulCPU(A, B, C_cpu, N);
    nvtxRangePop();

    // Compare results
    nvtxRangePushA("Result Comparison");
    bool correct = true;
    float maxError = 0.0f;
    for(int i = 0; i < N * N; i++){
        float diff = fabs(C_cpu[i] - C_gpu[i]);
        if(diff > maxError) maxError = diff;
        if(diff / (fabs(C_cpu[i]) + 1e-6f) > 1e-4f){
            correct = false;
        }
    }
    nvtxRangePop();

    printf("\nResults:\n");
    printf("  Max error: %.6f\n", maxError);
    printf("  Verification: %s\n\n", correct ? "PASS" : "FAIL");

    // =========================================
    // Cleanup
    // =========================================
    nvtxRangePushA("Host Memory Deallocation");
    delete[] A;
    delete[] B;
    delete[] C_gpu;
    delete[] C_cpu;
    nvtxRangePop();

    printf("To see NVTX annotations, profile with:\n");
    printf("  nsys profile ./01_nvtx_matmul.exe\n");
    printf("Then open the .nsys-rep file in Nsight Systems GUI.\n");

    return 0;
}

/*
 * ===========================================
 * What You'll See in Nsight Systems:
 * ===========================================
 *
 * The timeline will show nested ranges like this:
 *
 * [Host Memory Allocation]
 * [Matrix Initialization]
 * [Matrix Multiplication─────────────────────────────────]
 *   [Memory Allocation]
 *   [Memory Copy H2D──────]
 *   [Kernel Execution─────]
 *       └─ matrixMulKernel (actual CUDA kernel)
 *   [Memory Copy D2H──────]
 *   [Memory Deallocation]
 * [CPU Verification───────────────────────────────────────]
 * [Result Comparison]
 * [Host Memory Deallocation]
 *
 * Benefits:
 *   - Instantly see where time is spent
 *   - Identify bottlenecks (often memory transfer!)
 *   - Compare kernel time vs. memory overhead
 *   - Track nested operations clearly
 *
 * ===========================================
 * Advanced NVTX Features (not shown here):
 * ===========================================
 *
 * 1. Colored ranges:
 *    nvtxEventAttributes_t attr = {0};
 *    attr.version = NVTX_VERSION;
 *    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
 *    attr.colorType = NVTX_COLOR_ARGB;
 *    attr.color = 0xFF00FF00;  // Green
 *    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
 *    attr.message.ascii = "My Range";
 *    nvtxRangePushEx(&attr);
 *
 * 2. Named threads:
 *    nvtxNameOsThread(GetCurrentThreadId(), "Main Thread");
 *
 * 3. Categories (domains):
 *    nvtxDomainHandle_t domain = nvtxDomainCreate("MyApp");
 *    nvtxDomainRangePush(domain, "MyRange");
 *
 * 4. Payload data:
 *    attr.payloadType = NVTX_PAYLOAD_TYPE_INT64;
 *    attr.payload.llValue = iterationCount;
 */
