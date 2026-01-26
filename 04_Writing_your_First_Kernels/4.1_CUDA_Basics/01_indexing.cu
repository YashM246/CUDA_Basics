/*
 * CUDA 3D Thread Indexing
 * =======================
 *
 * What this program demonstrates:
 *   How every thread in a CUDA kernel can compute its own
 *   unique global ID from 3D grid and block indices.
 *
 * Why is this important?
 *   In CUDA, each thread needs to know WHICH piece of data
 *   it should work on. Threads identify themselves using
 *   built-in variables (blockIdx, threadIdx, etc.).
 *   This program shows the math behind flattening 3D
 *   indices into a single unique ID.
 *
 * ===========================================
 * Built-in Variables Used:
 * ===========================================
 *
 *   blockIdx.x/y/z  - This block's position in the grid
 *   gridDim.x/y/z   - Total blocks in each dimension
 *   threadIdx.x/y/z - This thread's position in its block
 *   blockDim.x/y/z  - Total threads per block in each dimension
 *
 * ===========================================
 * The Flattening Formula (3D → 1D):
 * ===========================================
 *
 *   For a 3D index (x, y, z) in a space of size (X, Y, Z):
 *     flat_index = x + y * X + z * X * Y
 *
 *   This is row-major order, just like how C stores
 *   multi-dimensional arrays in memory.
 *
 *   Example: index (1, 2, 3) in a (4, 5, 6) space:
 *     flat = 1 + 2*4 + 3*4*5 = 1 + 8 + 60 = 69
 *
 * ===========================================
 * Grid Layout for this program:
 * ===========================================
 *
 *   Grid:  2 × 3 × 4 = 24 blocks
 *   Block: 4 × 4 × 4 = 64 threads per block
 *   Total: 24 × 64 = 1,536 threads
 *
 *   Each block has 64 threads = 2 warps (64 / 32)
 *
 * Compile: nvcc 01_indexing.cu -o 01_indexing.exe
 * Run:     .\01_indexing.exe
 */

#include <stdio.h>

/*
 * Kernel: whoami
 *
 * Each thread computes and prints its unique global ID.
 * No data is processed - this is purely for understanding indexing.
 *
 * The calculation has 3 steps:
 *   1. Flatten the 3D block index → block_id
 *   2. Compute block_offset (how many threads come before this block)
 *   3. Flatten the 3D thread index within the block → thread_offset
 *   4. Global ID = block_offset + thread_offset
 */
__global__ void whoami(void){
    // Step 1: Flatten block's 3D position into a single block ID
    // Formula: x + y * width + z * width * height (row-major)
    //
    // Example: Block(1, 2, 3) in a grid of (2, 3, 4):
    //   block_id = 1 + 2*2 + 3*2*3 = 1 + 4 + 18 = 23
    int block_id =
        blockIdx.x +
        blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;

    // Step 2: Calculate how many threads exist in all previous blocks
    // Each block contains (blockDim.x * blockDim.y * blockDim.z) threads
    //
    // Example: block_id=5, block size=64 → offset = 5 * 64 = 320
    //   Threads 0-319 belong to earlier blocks
    int block_offset =
        block_id *
        blockDim.x * blockDim.y * blockDim.z;

    // Step 3: Flatten thread's 3D position within its block
    // Same row-major formula, but using threadIdx and blockDim
    //
    // Example: Thread(3, 2, 1) in a block of (4, 4, 4):
    //   thread_offset = 3 + 2*4 + 1*4*4 = 3 + 8 + 16 = 27
    int thread_offset =
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    // Step 4: Combine to get unique global thread ID
    int id = block_offset + thread_offset;

    // Print: globalID | Block(x,y,z) = flatBlockID | Thread(x,y,z) = flatThreadOffset
    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
            id,
            blockIdx.x, blockIdx.y, blockIdx.z, block_id,
            threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main(int argc, char **argv){
    // Grid dimensions: 2 × 3 × 4 = 24 blocks
    const int b_x = 2, b_y = 3, b_z = 4;

    // Block dimensions: 4 × 4 × 4 = 64 threads per block
    // 64 threads = 2 warps (warp size is always 32)
    const int t_x = 4, t_y = 4, t_z = 4;

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    // dim3 is a CUDA type for specifying 3D dimensions
    // Unspecified dimensions default to 1
    // dim3(2, 3, 4) means x=2, y=3, z=4
    dim3 blocksPerGrid(b_x, b_y, b_z);
    dim3 threadsPerBlock(t_x, t_y, t_z);

    // Launch kernel: <<<grid dimensions, block dimensions>>>
    // This creates 24 blocks × 64 threads = 1,536 total threads
    // All threads execute whoami() simultaneously
    whoami<<<blocksPerGrid, threadsPerBlock>>>();

    // Kernel launches are asynchronous - CPU doesn't wait for GPU
    // cudaDeviceSynchronize() makes CPU wait until all GPU threads finish
    // Without this, the program would exit before printf output appears
    cudaDeviceSynchronize();
}

/*
 * Expected output (order may vary - blocks run in any order!):
 *
 *   24 blocks/grid
 *   64 threads/block
 *   1536 total threads
 *   0000 | Block(0 0 0) =   0 | Thread(0 0 0) =   0
 *   0001 | Block(0 0 0) =   0 | Thread(1 0 0) =   1
 *   ...
 *   1535 | Block(1 2 3) =  23 | Thread(3 3 3) =  63
 *
 * Note: Block execution order is NOT guaranteed.
 *   You might see Block(1 2 3) print before Block(0 0 0).
 *   This is a fundamental CUDA property - blocks are independent
 *   and can be scheduled in any order by the GPU.
 *
 * Indexing summary:
 *   block_id       = bx + by * gridDim.x + bz * gridDim.x * gridDim.y
 *   block_offset   = block_id * (blockDim.x * blockDim.y * blockDim.z)
 *   thread_offset  = tx + ty * blockDim.x + tz * blockDim.x * blockDim.y
 *   global_id      = block_offset + thread_offset
 */
