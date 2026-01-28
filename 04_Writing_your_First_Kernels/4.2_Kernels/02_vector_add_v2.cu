/*
 * Vector Addition: 1D vs 3D Grid Benchmark
 * =========================================
 *
 * What this program demonstrates:
 *   Compares three approaches to vector addition:
 *     1. CPU sequential (baseline)
 *     2. GPU with 1D grid/blocks (flat indexing)
 *     3. GPU with 3D grid/blocks (volumetric indexing)
 *
 * Why compare 1D vs 3D?
 *   CUDA supports up to 3D grids and blocks. For flat arrays,
 *   1D indexing is simpler. But for volumetric data (medical imaging,
 *   fluid simulation, 3D textures), 3D indexing maps naturally
 *   to the problem domain.
 *
 *   Performance-wise, 1D and 3D should be nearly identical because:
 *   - The GPU hardware doesn't care about dimensionality
 *   - Both map to the same flat memory underneath
 *   - The only difference is how we COMPUTE the index
 *   - 3D has slightly more index arithmetic (2 extra multiplies)
 *
 * ===========================================
 * 3D Indexing Concept:
 * ===========================================
 *
 *   A 3D volume of size (nx, ny, nz) is stored as a flat 1D array:
 *
 *     Physical 3D position: (i, j, k)
 *     Flat 1D index:        idx = i + j*nx + k*nx*ny
 *
 *   This is row-major order (x varies fastest, then y, then z).
 *
 *   Example: Volume 4×3×2 (nx=4, ny=3, nz=2)
 *
 *     k=0 (front slice):      k=1 (back slice):
 *     j\i  0  1  2  3        j\i  0  1  2  3
 *      0 [ 0  1  2  3]        0 [12 13 14 15]
 *      1 [ 4  5  6  7]        1 [16 17 18 19]
 *      2 [ 8  9 10 11]        2 [20 21 22 23]
 *
 *     Position (2, 1, 1): idx = 2 + 1*4 + 1*4*3 = 2 + 4 + 12 = 18
 *
 * ===========================================
 * Grid/Block Configuration:
 * ===========================================
 *
 *   1D approach:
 *     Block: 1024 threads (1D)
 *     Grid:  ceil(N / 1024) = 9,766 blocks (1D)
 *     Total: 9,766 × 1024 = 10,000,384 threads
 *
 *   3D approach:
 *     Volume dimensions: nx=200, ny=200, nz=250 (200×200×250 = 10,000,000)
 *     Block: 16 × 8 × 8 = 1024 threads (3D)
 *     Grid:  ceil(200/16) × ceil(200/8) × ceil(250/8)
 *          = 13 × 25 × 32 = 10,400 blocks (3D)
 *     Total: 10,400 × 1024 = 10,649,600 threads
 *     (649,600 extra threads handled by bounds check)
 *
 * Compile: nvcc 02_vector_add_v2.cu -o 02_vector_add_v2.exe
 * Run:     .\02_vector_add_v2.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>           // For fabs() - floating-point absolute value
#include <cuda_runtime.h>   // CUDA runtime API

// Windows-compatible high-resolution timing
// clock_gettime(CLOCK_MONOTONIC) is POSIX-only (Linux/Mac)
#ifdef _WIN32
#include <windows.h>
#endif

// ===========================================
// Constants
// ===========================================

// Total number of elements: 10 million
// Each float = 4 bytes, so 3 vectors = ~120 MB on GPU
#define N 10000000

// 1D block size: 1024 threads per block (maximum allowed)
// 1024 = 32 warps per block (each warp = 32 threads)
// Using max block size minimizes the number of blocks needed
#define BLOCK_SIZE_1D 1024

// 3D block dimensions: 16 × 8 × 8 = 1024 threads per block
// Same total thread count as 1D, just organized in 3 dimensions
//
// Why 16×8×8 instead of 10×10×10 (=1000)?
//   - Each dimension should be a multiple of the warp size (32) or
//     at least contribute to a total that's a multiple of 32
//   - 16×8×8 = 1024 = 32×32 (exactly 32 warps, max occupancy)
//   - 10×10×10 = 1000 (not a multiple of 32, wastes 24 threads in last warp)
//
// Why x=16 instead of x=8?
//   - The x-dimension maps to consecutive memory addresses
//   - Threads with consecutive threadIdx.x access consecutive memory (coalesced)
//   - Larger x-dimension = more coalesced memory accesses per warp
//   - 16 consecutive threads in x can service half a warp's memory request
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8
// 16 * 8 * 8 = 1024

// 3D volume dimensions
// Must satisfy: NX * NY * NZ = N (10,000,000)
// 200 × 200 × 250 = 10,000,000
#define NX 200
#define NY 200
#define NZ 250

/*
 * CPU Vector Addition (Sequential Baseline)
 *
 * Simple loop: processes all N elements one at a time.
 * Used as the performance baseline and for correctness verification.
 */
void vector_add_cpu(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
}

/*
 * GPU Kernel: 1D Vector Addition
 *
 * Each thread computes ONE element using flat 1D indexing.
 * This is the standard approach from v1.
 *
 * Index calculation:
 *   i = blockIdx.x * blockDim.x + threadIdx.x
 *
 * Grid layout: 1D grid of 1D blocks
 *   [Block 0][Block 1][Block 2]...[Block 9765]
 *   Each block has 1024 threads
 */
__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] + b[i];
    }
}

/*
 * GPU Kernel: 3D Vector Addition
 *
 * Each thread computes ONE element, but uses 3D indexing to find
 * which element it's responsible for.
 *
 * Parameters:
 *   a, b - Input vectors in GPU global memory (stored as flat 1D arrays)
 *   c    - Output vector in GPU global memory
 *   nx, ny, nz - Volume dimensions (nx × ny × nz = total elements)
 *
 * Index calculation (3 steps):
 *   1. Compute 3D position from thread/block indices:
 *      i = blockIdx.x * blockDim.x + threadIdx.x  (x position)
 *      j = blockIdx.y * blockDim.y + threadIdx.y  (y position)
 *      k = blockIdx.z * blockDim.z + threadIdx.z  (z position)
 *
 *   2. Bounds check: (i < nx && j < ny && k < nz)
 *      Each dimension may have extra threads beyond the volume boundary
 *
 *   3. Flatten 3D → 1D: idx = i + j*nx + k*nx*ny
 *      This maps the 3D position to a flat array index (row-major order)
 *
 * Example: Thread at (i=5, j=3, k=2) in a 200×200×250 volume:
 *   idx = 5 + 3*200 + 2*200*200 = 5 + 600 + 80,000 = 80,605
 *   This thread computes: c[80605] = a[80605] + b[80605]
 *
 * Cost vs 1D kernel:
 *   1D: 1 multiply + 1 add for index      (blockIdx.x * blockDim.x + threadIdx.x)
 *   3D: 3 multiplies + 3 adds for indices  (one per dimension)
 *       + 2 multiplies + 2 adds to flatten (j*nx + k*nx*ny)
 *   The extra arithmetic is negligible compared to memory access time
 */
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz){
    // Compute 3D thread position
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // x-coordinate
    int j = blockIdx.y * blockDim.y + threadIdx.y;   // y-coordinate
    int k = blockIdx.z * blockDim.z + threadIdx.z;   // z-coordinate

    // Bounds check: all three coordinates must be within the volume
    // Since grid dimensions are rounded up (ceiling division), some threads
    // will have coordinates outside the volume boundary
    if(i < nx && j < ny && k < nz){
        // Flatten 3D index to 1D (row-major order)
        // x varies fastest (stride 1), then y (stride nx), then z (stride nx*ny)
        int idx = i + (j * nx) + (k * nx * ny);
        c[idx] = a[idx] + b[idx];
    }
}

/*
 * Initialize vector with random float values in [0, 1]
 *
 * (float)rand() / RAND_MAX normalizes random ints to [0.0, 1.0]
 */
void init_vector(float *vec, int n){
    for(int i = 0; i < n; i++){
        vec[i] = (float)rand() / RAND_MAX;
    }
}

/*
 * High-resolution timer (platform-independent)
 * Returns current time in seconds as a double.
 *
 * Windows: QueryPerformanceCounter (sub-microsecond precision)
 * Linux/Mac: clock_gettime with CLOCK_MONOTONIC (nanosecond precision)
 */
double get_time(){
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

int main(){
    // ===========================================
    // Variable declarations
    // ===========================================

    // Host (CPU) pointers
    //   h_a, h_b:      Input vectors
    //   h_c_cpu:       CPU result
    //   h_c_gpu_1d:    GPU 1D result (copied back for verification)
    //   h_c_gpu_3d:    GPU 3D result (copied back for verification)
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;

    // Device (GPU) pointers - stored in VRAM, not accessible from CPU
    float *d_a, *d_b, *d_c;

    // Total bytes: 10,000,000 × 4 = ~38 MB per vector
    size_t size = N * sizeof(float);

    // Verify 3D dimensions match N
    // This compile-time-like check ensures our 3D decomposition is correct
    if(NX * NY * NZ != N){
        printf("Error: NX*NY*NZ (%d) != N (%d)\n", NX * NY * NZ, N);
        return 1;
    }

    // ===========================================
    // Step 1: Allocate host memory
    // ===========================================
    // 5 host arrays: 2 inputs + 3 outputs (CPU, GPU-1D, GPU-3D)
    h_a         = (float*)malloc(size);
    h_b         = (float*)malloc(size);
    h_c_cpu     = (float*)malloc(size);
    h_c_gpu_1d  = (float*)malloc(size);
    h_c_gpu_3d  = (float*)malloc(size);

    // ===========================================
    // Step 2: Initialize input vectors
    // ===========================================
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // ===========================================
    // Step 3: Allocate device memory
    // ===========================================
    // Only 3 device arrays needed - we reuse d_c for both kernels
    // (copy results back between runs)
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // ===========================================
    // Step 4: Copy input data from host → device
    // ===========================================
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // ===========================================
    // Step 5: Configure grid/block dimensions
    // ===========================================

    // --- 1D Configuration ---
    // Simple: one thread per element, blocks of 1024
    // num_blocks_1d = ceil(10,000,000 / 1024) = 9,766 blocks
    int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    // --- 3D Configuration ---
    // dim3 specifies 3D dimensions for grid and block
    // Block: 16 × 8 × 8 = 1024 threads
    dim3 block_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);

    // Grid: enough blocks in each dimension to cover the volume
    // Ceiling division per dimension:
    //   x: ceil(200/16) = 13 blocks
    //   y: ceil(200/8)  = 25 blocks
    //   z: ceil(250/8)  = 32 blocks
    // Total: 13 × 25 × 32 = 10,400 blocks
    dim3 grid_3d(
        (NX + BLOCK_SIZE_3D_X - 1) / BLOCK_SIZE_3D_X,
        (NY + BLOCK_SIZE_3D_Y - 1) / BLOCK_SIZE_3D_Y,
        (NZ + BLOCK_SIZE_3D_Z - 1) / BLOCK_SIZE_3D_Z
    );

    // Print configuration
    printf("=== Vector Addition: 1D vs 3D Benchmark ===\n\n");
    printf("Vector size:      %d elements (%.1f MB per vector)\n", N, size / (1024.0 * 1024.0));
    printf("Volume dims:      %d x %d x %d\n\n", NX, NY, NZ);
    printf("1D config:\n");
    printf("  Block:          %d threads\n", BLOCK_SIZE_1D);
    printf("  Grid:           %d blocks\n", num_blocks_1d);
    printf("  Total threads:  %d\n\n", num_blocks_1d * BLOCK_SIZE_1D);
    printf("3D config:\n");
    printf("  Block:          %d x %d x %d = %d threads\n",
           BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z,
           BLOCK_SIZE_3D_X * BLOCK_SIZE_3D_Y * BLOCK_SIZE_3D_Z);
    printf("  Grid:           %d x %d x %d = %d blocks\n",
           grid_3d.x, grid_3d.y, grid_3d.z,
           grid_3d.x * grid_3d.y * grid_3d.z);
    printf("  Total threads:  %d\n\n",
           grid_3d.x * grid_3d.y * grid_3d.z *
           BLOCK_SIZE_3D_X * BLOCK_SIZE_3D_Y * BLOCK_SIZE_3D_Z);

    // ===========================================
    // Step 6: Warmup runs
    // ===========================================
    // First kernel launch has JIT compilation and context setup overhead.
    // First CPU run warms the data into CPU caches.
    // Run all three approaches to ensure fair benchmarking.
    printf("Performing warmup runs...\n");
    for(int i = 0; i < 3; i++){
        vector_add_cpu(h_a, h_b, h_c_cpu, N);

        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();

        vector_add_gpu_3d<<<grid_3d, block_3d>>>(d_a, d_b, d_c, NX, NY, NZ);
        cudaDeviceSynchronize();
    }

    // ===========================================
    // Step 7: Benchmark CPU
    // ===========================================
    printf("Benchmarking CPU...\n");
    double cpu_total = 0.0;
    for(int i = 0; i < 20; i++){
        double start = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end = get_time();
        cpu_total += end - start;
    }
    double cpu_avg = cpu_total / 20.0;

    // ===========================================
    // Step 8: Benchmark GPU 1D kernel
    // ===========================================
    // cudaDeviceSynchronize() inside the timing loop ensures we measure
    // actual kernel execution time, not just launch overhead
    printf("Benchmarking GPU (1D)...\n");
    double gpu_1d_total = 0.0;
    for(int i = 0; i < 20; i++){
        double start = get_time();
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_1d_total += end - start;
    }
    double gpu_1d_avg = gpu_1d_total / 20.0;

    // ===========================================
    // Step 9: Benchmark GPU 3D kernel
    // ===========================================
    printf("Benchmarking GPU (3D)...\n");
    double gpu_3d_total = 0.0;
    for(int i = 0; i < 20; i++){
        double start = get_time();
        vector_add_gpu_3d<<<grid_3d, block_3d>>>(d_a, d_b, d_c, NX, NY, NZ);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_3d_total += end - start;
    }
    double gpu_3d_avg = gpu_3d_total / 20.0;

    // ===========================================
    // Step 10: Print benchmark results
    // ===========================================
    printf("\n=== Results ===\n");
    printf("  CPU Avg Time:      %.3f ms\n", cpu_avg * 1000.0);
    printf("  GPU 1D Avg Time:   %.3f ms\n", gpu_1d_avg * 1000.0);
    printf("  GPU 3D Avg Time:   %.3f ms\n", gpu_3d_avg * 1000.0);
    printf("\n");
    printf("  Speedup (CPU vs GPU 1D): %.2fx\n", cpu_avg / gpu_1d_avg);
    printf("  Speedup (CPU vs GPU 3D): %.2fx\n", cpu_avg / gpu_3d_avg);
    printf("  Ratio   (1D vs 3D):      %.4fx\n", gpu_1d_avg / gpu_3d_avg);

    // ===========================================
    // Step 11: Verify GPU results match CPU
    // ===========================================
    // Copy both GPU results back to host for comparison

    // --- Verify 1D kernel ---
    // Re-run 1D kernel once to get its result in d_c
    vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c_gpu_1d, d_c, size, cudaMemcpyDeviceToHost);

    // --- Verify 3D kernel ---
    // Re-run 3D kernel once to get its result in d_c
    vector_add_gpu_3d<<<grid_3d, block_3d>>>(d_a, d_b, d_c, NX, NY, NZ);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c_gpu_3d, d_c, size, cudaMemcpyDeviceToHost);

    // Compare each element with tolerance for floating-point differences
    // fabs(a - b) > 1e-5 allows for tiny rounding errors between CPU/GPU
    bool correct_1d = true;
    bool correct_3d = true;
    for(int i = 0; i < N; i++){
        if(fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-5){
            printf("  1D mismatch at %d: CPU=%.7f, GPU=%.7f\n",
                   i, h_c_cpu[i], h_c_gpu_1d[i]);
            correct_1d = false;
            break;
        }
        if(fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-5){
            printf("  3D mismatch at %d: CPU=%.7f, GPU=%.7f\n",
                   i, h_c_cpu[i], h_c_gpu_3d[i]);
            correct_3d = false;
            break;
        }
    }
    printf("\n=== Verification ===\n");
    printf("  GPU 1D: %s\n", correct_1d ? "PASS" : "FAIL");
    printf("  GPU 3D: %s\n", correct_3d ? "PASS" : "FAIL");

    // ===========================================
    // Step 12: Free all allocated memory
    // ===========================================
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);

    return 0;
}

/*
 * Expected results:
 *
 *   1D and 3D kernel times should be nearly identical.
 *   The small differences come from:
 *     - 3D has slightly more index arithmetic (2 extra multiplies + adds)
 *     - 3D launches more total threads (10,649,600 vs 10,000,384)
 *       due to rounding up in 3 dimensions instead of 1
 *     - 3D has a more complex bounds check (3 conditions vs 1)
 *
 *   These differences are negligible because vector addition is
 *   memory-bound: the bottleneck is reading/writing 120 MB of data,
 *   not the handful of extra integer operations.
 *
 * When to use 3D indexing:
 *
 *   - 3D simulations (fluid dynamics, weather, molecular dynamics)
 *   - Medical imaging (CT/MRI volumes are naturally 3D)
 *   - 3D convolutions (deep learning with volumetric data)
 *   - Voxel-based processing (games, 3D reconstruction)
 *
 *   Using 3D grids makes the code more readable when the problem
 *   is naturally 3D - thread (i,j,k) maps directly to voxel (i,j,k).
 *   For flat arrays, 1D is simpler and slightly more efficient.
 *
 * Memory layout note:
 *
 *   Even with 3D indexing, the data is still stored in a flat 1D array.
 *   The 3D → 1D mapping (idx = i + j*nx + k*nx*ny) is row-major order,
 *   identical to how C stores multi-dimensional arrays:
 *     float volume[NZ][NY][NX];   // volume[k][j][i] = flat[i + j*NX + k*NX*NY]
 *
 *   For coalesced memory access, consecutive threads in x (threadIdx.x)
 *   should access consecutive memory addresses. With our mapping,
 *   threads differing only in i (x-coordinate) access adjacent floats,
 *   so memory access is coalesced. This is why BLOCK_SIZE_3D_X = 16
 *   is the largest dimension.
 */
