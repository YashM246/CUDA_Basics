/*
 * CUDA Streams - Basics and Asynchronous Operations
 * ==================================================
 *
 * This program demonstrates CUDA streams, which enable concurrent execution
 * and overlap of data transfers with kernel computation.
 *
 * ===========================================
 * What is a CUDA Stream?
 * ===========================================
 *
 * A stream is a SEQUENCE of operations that execute IN ORDER on the GPU.
 * Operations in DIFFERENT streams can execute CONCURRENTLY.
 *
 * Think of streams like lanes on a highway:
 *   - Cars in the SAME lane must follow each other (in order)
 *   - Cars in DIFFERENT lanes can drive side by side (concurrent)
 *
 * Default behavior (no streams):
 *   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
 *   │ Copy A   │→ │ Copy B   │→ │ Kernel   │→ │ Copy C   │
 *   │ H→D      │  │ H→D      │  │          │  │ D→H      │
 *   └──────────┘  └──────────┘  └──────────┘  └──────────┘
 *   Total time = T_copyA + T_copyB + T_kernel + T_copyC
 *
 * With streams (overlapping operations):
 *   Stream 1: ┌──────────┐              ┌──────────┐  ┌──────────┐
 *             │ Copy A   │─────────────→│ Kernel   │→ │ Copy C   │
 *             │ H→D      │              │          │  │ D→H      │
 *             └──────────┘              └──────────┘  └──────────┘
 *   Stream 2:       ┌──────────┐
 *                   │ Copy B   │  (runs concurrently with Copy A!)
 *                   │ H→D      │
 *                   └──────────┘
 *   Total time ≈ max(T_copyA, T_copyB) + T_kernel + T_copyC (FASTER!)
 *
 * ===========================================
 * Key Stream Concepts
 * ===========================================
 *
 * 1. Default Stream (Stream 0)
 *    - Created automatically
 *    - All operations without explicit stream go here
 *    - BLOCKS other streams (synchronizing behavior)
 *
 * 2. Non-Default Streams
 *    - Created with cudaStreamCreate()
 *    - Run concurrently with other non-default streams
 *    - Must be explicitly synchronized when needed
 *
 * 3. Synchronization Options
 *    - cudaStreamSynchronize(stream): Wait for ONE stream to finish
 *    - cudaDeviceSynchronize(): Wait for ALL streams to finish
 *    - cudaStreamWaitEvent(): Make one stream wait for an event in another
 *
 * ===========================================
 * Requirements for True Asynchronous Execution
 * ===========================================
 *
 * IMPORTANT: cudaMemcpyAsync only runs asynchronously with PINNED memory!
 *
 * Regular malloc (pageable memory):
 *   - cudaMemcpyAsync BLOCKS until copy completes
 *   - No concurrency benefit
 *
 * cudaMallocHost (pinned memory):
 *   - Memory is "locked" in physical RAM (can't be swapped to disk)
 *   - DMA engine can transfer directly
 *   - TRUE asynchronous execution
 *
 * Compile: nvcc 01_stream_basics.cu -o 01_stream_basics.exe
 * Run:     .\01_stream_basics.exe
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * ===========================================
 * CUDA Error Checking Macro
 * ===========================================
 *
 * This macro checks the return value of CUDA API calls.
 * If an error occurred, it prints the error message and exits.
 *
 * Usage: CHECK_CUDA_ERROR(cudaMalloc(...));
 *
 * The macro captures:
 *   - val: The actual error code
 *   - #val: The function call as a string (for error message)
 *   - __FILE__, __LINE__: Where the error occurred
 */
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

/*
 * ===========================================
 * What is "template <typename T>"?
 * ===========================================
 *
 * Templates are a C++ feature that lets you write ONE function that works
 * with MULTIPLE types. The compiler generates the appropriate version
 * for each type you use.
 *
 * Without templates, you'd need to write multiple functions:
 *
 *   void checkInt(int err, ...) { ... }
 *   void checkFloat(float err, ...) { ... }
 *   void checkCudaError(cudaError_t err, ...) { ... }
 *
 * With templates, you write ONE function:
 *
 *   template <typename T>
 *   void check(T err, ...) { ... }
 *
 * How it works:
 *   - "template <typename T>" tells the compiler: "T is a placeholder for any type"
 *   - When you call check(cudaMalloc(...)), the compiler sees cudaMalloc returns cudaError_t
 *   - The compiler automatically generates: void check(cudaError_t err, ...)
 *
 * In this case:
 *   - cudaMalloc returns cudaError_t
 *   - cudaMemcpy returns cudaError_t
 *   - cudaStreamCreate returns cudaError_t
 *   - All CUDA API calls return cudaError_t, so T becomes cudaError_t
 *
 * Why use templates here?
 *   - Future-proofing: If CUDA ever changes return types, code still works
 *   - Flexibility: Same pattern works for other error types
 *   - It's a common C++ idiom for error checking utilities
 *
 * Simple analogy:
 *   Template is like a cookie cutter shape.
 *   T is like the dough color.
 *   You get the same shape cookie regardless of dough color.
 */
template <typename T>
void check(T err, const char *const func, const char *const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, static_cast<unsigned int>(err),
                cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

/*
 * ===========================================
 * Vector Addition Kernel
 * ===========================================
 *
 * Simple kernel: C[i] = A[i] + B[i]
 *
 * This kernel will be launched in a specific stream, allowing it
 * to run concurrently with memory transfers in other streams.
 */
__global__ void vectorAdd(float *A, float *B, float *C, int numElements) {
    // Calculate global thread index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Bounds check: only process valid elements
    if (idx < numElements) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    printf("CUDA Streams Demo - Asynchronous Operations\n");
    printf("============================================\n\n");

    /*
     * ===========================================
     * Configuration
     * ===========================================
     */
    int numElements = 1 << 20;  // 1M elements (2^20 = 1,048,576)
    size_t size = numElements * sizeof(float);

    printf("Vector size: %d elements (%.2f MB)\n", numElements, size / (1024.0 * 1024.0));

    /*
     * ===========================================
     * Host Memory Allocation - PINNED MEMORY
     * ===========================================
     *
     * cudaMallocHost allocates "pinned" (page-locked) memory.
     *
     * Why pinned memory?
     * ------------------
     * Regular memory (malloc):
     *   - Can be swapped to disk by the OS
     *   - GPU must copy to a staging buffer first
     *   - cudaMemcpyAsync actually blocks!
     *
     * Pinned memory (cudaMallocHost):
     *   - Locked in physical RAM (never swapped)
     *   - GPU's DMA engine can access directly
     *   - TRUE asynchronous transfers
     *   - Higher bandwidth (~2x faster transfers)
     *
     * Downside: Pinned memory is a limited resource.
     * Don't pin more memory than necessary.
     */
    float *h_A, *h_B, *h_C;

    printf("Allocating pinned host memory...\n");
    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_A, size));
    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_B, size));
    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_C, size));

    /*
     * ===========================================
     * Initialize Host Arrays
     * ===========================================
     */
    printf("Initializing host arrays...\n");
    for (int i = 0; i < numElements; i++) {
        h_A[i] = (float)(rand()) / RAND_MAX;
        h_B[i] = (float)(rand()) / RAND_MAX;
    }

    /*
     * ===========================================
     * Device Memory Allocation
     * ===========================================
     *
     * cudaMalloc signature:
     *   cudaError_t cudaMalloc(void **devPtr, size_t size);
     *
     * ===========================================
     * Why (void **) - The Double Pointer Explained
     * ===========================================
     *
     * This is confusing at first, but makes sense once you understand
     * how functions can "return" values through pointers.
     *
     * The Problem:
     * ------------
     * cudaMalloc needs to tell us WHERE it allocated memory (the address).
     * But its return value is already used for the error code (cudaError_t).
     * So how does it give us the allocated address?
     *
     * The Solution: Pass a pointer TO our pointer
     * --------------------------------------------
     *
     * Let's trace through what happens:
     *
     *   float *d_A;                    // d_A is a pointer (holds an address)
     *                                  // Currently d_A = garbage/undefined
     *
     *   &d_A                           // "Address of d_A" - where d_A itself lives in memory
     *                                  // This is a pointer TO a pointer: float**
     *
     *   (void **)&d_A                  // Cast to void** (generic double pointer)
     *                                  // void* means "pointer to anything"
     *
     *   cudaMalloc((void **)&d_A, size)
     *                                  // cudaMalloc receives the ADDRESS of d_A
     *                                  // It allocates GPU memory at address 0xABC123 (example)
     *                                  // It WRITES 0xABC123 into d_A
     *                                  // Now d_A = 0xABC123 (points to GPU memory!)
     *
     * Visual representation:
     *
     *   BEFORE cudaMalloc:
     *   ┌─────────────────┐
     *   │ d_A = ???       │  ← Our pointer variable (undefined)
     *   │ (at addr 0x100) │
     *   └─────────────────┘
     *
     *   We pass &d_A (0x100) to cudaMalloc
     *
     *   AFTER cudaMalloc:
     *   ┌─────────────────┐         ┌─────────────────────────┐
     *   │ d_A = 0xGPU456  │ ──────→ │ GPU Memory (size bytes) │
     *   │ (at addr 0x100) │         │ at address 0xGPU456     │
     *   └─────────────────┘         └─────────────────────────┘
     *
     * Why void**?
     * -----------
     * - void* is a "generic pointer" that can point to any type
     * - cudaMalloc is designed to work with ANY type (int*, float*, struct*, etc.)
     * - void** means "pointer to a generic pointer"
     * - We cast our float** to void** so cudaMalloc accepts it
     *
     * Compare with malloc (which RETURNS the address):
     * ------------------------------------------------
     *   float *h_A = (float *)malloc(size);   // malloc returns the address
     *   cudaMalloc((void **)&d_A, size);      // cudaMalloc writes to our pointer
     *
     * Why doesn't cudaMalloc just return the address like malloc?
     * Because CUDA functions return cudaError_t for error checking.
     */
    float *d_A, *d_B, *d_C;

    printf("Allocating device memory...\n");
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

    /*
     * ===========================================
     * Create CUDA Streams
     * ===========================================
     *
     * cudaStreamCreate() creates a new stream.
     *
     * We create two streams:
     *   - stream1: For copying A and running the kernel
     *   - stream2: For copying B (can overlap with stream1)
     *
     * The stream handle is just an identifier (like a file handle).
     */
    cudaStream_t stream1, stream2;

    printf("Creating CUDA streams...\n");
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    /*
     * ===========================================
     * Asynchronous Memory Transfers
     * ===========================================
     *
     * cudaMemcpyAsync vs cudaMemcpy:
     *
     *   cudaMemcpy (synchronous):
     *     - Blocks the CPU until transfer completes
     *     - CPU sits idle during transfer
     *
     *   cudaMemcpyAsync (asynchronous):
     *     - Returns immediately to the CPU
     *     - Transfer happens in the background
     *     - CPU can do other work (or launch more GPU operations)
     *
     * The last parameter specifies which stream to use.
     * Operations in the same stream execute in order.
     *
     * Timeline:
     *   stream1: [Copy A H→D]─────────────→[Kernel]→[Copy C D→H]
     *   stream2:      [Copy B H→D] (overlaps with Copy A!)
     */
    printf("Launching asynchronous memory copies...\n");

    // Copy A to device in stream1
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1));

    // Copy B to device in stream2 (can run concurrently with A's copy!)
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2));

    /*
     * ===========================================
     * Stream Synchronization Before Kernel
     * ===========================================
     *
     * The kernel needs BOTH d_A and d_B to be ready.
     *
     * d_A is in stream1 (same as kernel), so it's automatically ready.
     * d_B is in stream2, so we must wait for stream2 before launching
     * the kernel in stream1.
     *
     * cudaStreamSynchronize(stream2) blocks the CPU until all
     * operations in stream2 have completed.
     *
     * Alternative: Use cudaStreamWaitEvent() for GPU-side synchronization
     * (more advanced, avoids blocking the CPU).
     */
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    /*
     * ===========================================
     * Launch Kernel in Stream 1
     * ===========================================
     *
     * Kernel launch syntax with streams:
     *   kernel<<<gridDim, blockDim, sharedMem, stream>>>(...);
     *
     * The 4th parameter (stream) specifies which stream to use.
     * If omitted, the default stream (0) is used.
     *
     * Since the kernel is in stream1, it will wait for the
     * cudaMemcpyAsync(d_A) in stream1 to complete first.
     */
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching kernel in stream1...\n");
    printf("  Grid: %d blocks, Block: %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, numElements);

    /*
     * ===========================================
     * Copy Results Back Asynchronously
     * ===========================================
     *
     * NOTE: The destination (h_C) comes FIRST in cudaMemcpy!
     *   cudaMemcpy(destination, source, size, direction)
     *
     * This copy is in stream1, so it automatically waits for
     * the kernel (also in stream1) to complete.
     */
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1));

    /*
     * ===========================================
     * Final Synchronization
     * ===========================================
     *
     * Wait for all operations to complete before verifying results.
     *
     * cudaStreamSynchronize(stream1) waits for:
     *   - cudaMemcpyAsync(d_A) to complete
     *   - vectorAdd kernel to complete
     *   - cudaMemcpyAsync(h_C) to complete
     *
     * stream2 is already synchronized (we did it earlier).
     */
    printf("Synchronizing streams...\n");
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));

    /*
     * ===========================================
     * Verify Results
     * ===========================================
     *
     * Check that C[i] = A[i] + B[i] for all elements.
     * Use a small tolerance for floating-point comparison.
     */
    printf("Verifying results...\n");
    int errors = 0;
    for (int i = 0; i < numElements; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            if (errors < 10) {  // Only print first 10 errors
                fprintf(stderr, "Mismatch at %d: expected %.6f, got %.6f\n",
                        i, expected, h_C[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("\nTest PASSED! All %d elements correct.\n", numElements);
    } else {
        printf("\nTest FAILED! %d errors found.\n", errors);
    }

    /*
     * ===========================================
     * Cleanup
     * ===========================================
     *
     * Important: Destroy streams before freeing memory!
     *
     * cudaFreeHost: Frees pinned memory (NOT regular free())
     * cudaStreamDestroy: Releases stream resources
     */
    printf("\nCleaning up...\n");

    // Destroy streams first
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    // Free pinned host memory (use cudaFreeHost, NOT free!)
    CHECK_CUDA_ERROR(cudaFreeHost(h_A));
    CHECK_CUDA_ERROR(cudaFreeHost(h_B));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C));

    printf("Done!\n");

    return 0;
}

/*
 * ===========================================
 * Expected Output
 * ===========================================
 *
 * CUDA Streams Demo - Asynchronous Operations
 * ============================================
 *
 * Vector size: 1048576 elements (4.00 MB)
 * Allocating pinned host memory...
 * Initializing host arrays...
 * Allocating device memory...
 * Creating CUDA streams...
 * Launching asynchronous memory copies...
 * Launching kernel in stream1...
 *   Grid: 4096 blocks, Block: 256 threads
 * Synchronizing streams...
 * Verifying results...
 *
 * Test PASSED! All 1048576 elements correct.
 *
 * Cleaning up...
 * Done!
 *
 * ===========================================
 * Key Takeaways
 * ===========================================
 *
 * 1. Streams enable concurrent execution and overlap of operations
 *
 * 2. Operations in the SAME stream execute IN ORDER
 *
 * 3. Operations in DIFFERENT streams can execute CONCURRENTLY
 *
 * 4. cudaMemcpyAsync requires PINNED memory for true async behavior
 *
 * 5. Always synchronize before accessing results on the CPU
 *
 * 6. Use cudaFreeHost() for memory allocated with cudaMallocHost()
 */
