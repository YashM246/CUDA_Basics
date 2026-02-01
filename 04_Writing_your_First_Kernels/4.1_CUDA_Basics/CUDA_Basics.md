# CUDA Basics

This guide covers the fundamental concepts you'll use repeatedly in CUDA programming.

---

## Host vs Device

| Term | Hardware | Memory |
|------|----------|--------|
| **Host** | CPU | RAM (system memory) |
| **Device** | GPU | VRAM (video memory) |

### Naming Convention
```cpp
h_A  // Host variable (CPU)
d_A  // Device variable (GPU)
```

This convention makes it clear where your data lives.

---

## CUDA Program Flow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Allocate memory on GPU (cudaMalloc)                       │
│ 2. Copy input data from Host → Device (cudaMemcpy)           │
│ 3. Launch kernel (parallel execution on GPU)                 │
│ 4. Copy results from Device → Host (cudaMemcpy)              │
│ 5. Free GPU memory (cudaFree)                                │
└──────────────────────────────────────────────────────────────┘
```

---

## Function Qualifiers

CUDA extends C/C++ with special function qualifiers:

| Qualifier | Runs On | Called From | Returns |
|-----------|---------|-------------|---------|
| `__global__` | GPU | CPU (or GPU) | `void` only |
| `__device__` | GPU | GPU only | Any type |
| `__host__` | CPU | CPU only | Any type |

### `__global__` (Kernels)
The main entry point for GPU code. Called from CPU, executed by thousands of GPU threads.

```cpp
__global__ void addNumbers(int *a, int *b, int *result) {
    *result = *a + *b;
}
```

### `__device__`
Helper functions that run on GPU, callable only from other GPU code. Think of it as a library function for your kernels.

```cpp
__device__ float square(float x) {
    return x * x;
}

__global__ void compute(float *data) {
    int idx = threadIdx.x;
    data[idx] = square(data[idx]);  // Call device function
}
```

### `__host__`
Regular CPU function (same as omitting the qualifier). Can be combined with `__device__` to compile for both:

```cpp
__host__ __device__ float multiply(float a, float b) {
    return a * b;  // Works on both CPU and GPU
}
```

---

## CUDA Execution Hierarchy

```
Grid (entire kernel launch)
├── Block (0,0)
│   ├── Warp 0 (threads 0-31)
│   ├── Warp 1 (threads 32-63)
│   └── ... more warps
├── Block (1,0)
│   └── ... threads organized into warps
└── ... more blocks
```

### Thread
The smallest unit of execution. Each thread:
- Runs the kernel code independently
- Has a unique ID within its block
- Has private local memory (registers)

### Warp
- **32 threads** that execute in lockstep (SIMD)
- The actual unit of execution on hardware
- Instructions are issued to warps, not individual threads
- You can't avoid warps - they're fundamental to GPU architecture

![Warp Concept](./weft.png)

### Block
A group of threads that can cooperate:
- Share memory (shared memory)
- Synchronize with each other
- Can be 1D, 2D, or 3D
- Maximum 1024 threads per block

### Grid
The entire set of threads for one kernel launch:
- Collection of blocks
- Can be 1D, 2D, or 3D
- Blocks execute independently (no guaranteed order)

**Why this hierarchy?**
- Blocks execute independently → Scalability across different GPUs
- Threads in a block can cooperate → Efficient data sharing
- Warps execute in lockstep → Hardware efficiency

---

## Thread Indexing

Every thread knows its position through built-in variables:

| Variable | Type | Description |
|----------|------|-------------|
| `threadIdx` | `dim3` | Thread's position within its block (x, y, z) |
| `blockIdx` | `dim3` | Block's position within the grid (x, y, z) |
| `blockDim` | `dim3` | Dimensions of each block (threads per block) |
| `gridDim` | `dim3` | Dimensions of the grid (blocks per grid) |

### How Are These Values Assigned?

These variables are **set automatically by the GPU hardware** based on your kernel launch:

```cpp
dim3 threadsPerBlock(16, 16);  // You specify this
dim3 numBlocks(64, 64);        // You specify this
kernel<<<numBlocks, threadsPerBlock>>>(...);
```

| Variable | Set By | Value |
|----------|--------|-------|
| `blockDim` | Your `threadsPerBlock` argument | Same for all threads |
| `gridDim` | Your `numBlocks` argument | Same for all threads |
| `blockIdx` | Hardware assigns | 0 to gridDim-1 (unique per block) |
| `threadIdx` | Hardware assigns | 0 to blockDim-1 (unique within block) |

The GPU automatically assigns unique `blockIdx` and `threadIdx` values to each thread when the kernel launches.

### Calculating Global Thread ID

For 1D arrays, the most common formula:

```cpp
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
```

**Visual Example:**
```
Grid with 3 blocks, 4 threads each:

Block 0          Block 1          Block 2
[0][1][2][3]    [4][5][6][7]    [8][9][10][11]
 ↑               ↑
 threadIdx.x=0   threadIdx.x=0
 blockIdx.x=0    blockIdx.x=1
 global=0        global=4

For thread at blockIdx.x=1, threadIdx.x=2:
globalIdx = 1 * 4 + 2 = 6
```

### 2D Example

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int globalIdx = row * width + col;
```

---

## Launching Kernels

### The `<<<>>>` Syntax

```cpp
kernel<<<gridSize, blockSize>>>(arguments);
kernel<<<gridSize, blockSize, sharedMem, stream>>>(arguments);
```

### Using `dim3`

`dim3` specifies 3D dimensions with **default values of 1** for unspecified dimensions:

```cpp
dim3(x)        // Equivalent to dim3(x, 1, 1)
dim3(x, y)     // Equivalent to dim3(x, y, 1)
dim3(x, y, z)  // All three specified
```

Examples:
```cpp
dim3 blockSize(256);        // 1D: (256, 1, 1) = 256 threads
dim3 blockSize(16, 16);     // 2D: (16, 16, 1) = 256 threads
dim3 blockSize(8, 8, 4);    // 3D: (8, 8, 4)   = 256 threads

dim3 gridSize(8, 8);        // 8×8×1 = 64 blocks
myKernel<<<gridSize, blockSize>>>(data);
```

You can also use plain integers for 1D configurations:
```cpp
myKernel<<<numBlocks, 256>>>(data);  // 256 threads per block (1D)
```

### Calculating Grid Size

To cover an array of size N:

```cpp
int blockSize = 256;
int gridSize = (N + blockSize - 1) / blockSize;  // Ceiling division

myKernel<<<gridSize, blockSize>>>(data, N);
```

The `+ blockSize - 1` ensures we have enough threads even if N isn't divisible by blockSize.

### Choosing Block Size

How do you decide how many threads per block to use?

| Consideration | Guidance |
|---------------|----------|
| **Minimum** | At least 128-256 threads (for hiding memory latency) |
| **Maximum** | 1024 threads per block (hardware limit) |
| **Warp alignment** | Should be a multiple of 32 (warp size) |
| **Register pressure** | Complex kernels may need smaller blocks |
| **Shared memory** | Block size affects shared memory usage |

**Common choices:**
- 1D problems: 256 or 512 threads
- 2D problems: 16×16 (256) or 32×32 (1024)

**Rule of thumb:** Start with 256 threads, profile your kernel, then adjust if needed.

```cpp
// CUDA can help you find optimal block size:
int minGridSize, optimalBlockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, myKernel, 0, 0);
```

---

## Memory Management

### Calculating Memory Size

Memory functions need size in **bytes**, not element count. Use this formula:

```cpp
size_t size = numberOfElements * sizeof(dataType);
```

Examples:
```cpp
// 1D array: n elements
size_t size = n * sizeof(float);                  // n floats

// 2D matrix: rows × cols
size_t size = rows * cols * sizeof(float);        // rows×cols floats

// 3D volume: x × y × z
size_t size = x * y * z * sizeof(double);         // x×y×z doubles
```

**Why `size_t` instead of `int`?**

`size_t` is an unsigned integer type guaranteed to hold any valid memory size:

| Type | Typical size | Max value |
|------|--------------|-----------|
| `int` | 4 bytes | ~2 billion |
| `size_t` | 8 bytes (64-bit) | ~18 quintillion |

```cpp
// Problem: int overflow for large allocations!
int size = 50000 * 50000 * sizeof(float);    // Overflow! (10 billion > 2 billion)

// Solution: use size_t
size_t size = 50000ULL * 50000 * sizeof(float);  // Works correctly
```

Use `size_t` for any variable that stores memory sizes or array lengths.

### cudaMalloc
Allocates memory on the GPU:

```cpp
float *d_array;
cudaMalloc(&d_array, N * sizeof(float));
```

### cudaMemcpy
Copies data between host and device:

```cpp
// Host → Device
cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

// Device → Host
cudaMemcpy(h_array, d_array, N * sizeof(float), cudaMemcpyDeviceToHost);

// Device → Device
cudaMemcpy(d_dest, d_src, N * sizeof(float), cudaMemcpyDeviceToDevice);
```

### cudaFree
Releases GPU memory:

```cpp
cudaFree(d_array);
```

### cudaDeviceSynchronize
Forces CPU to wait until all GPU operations complete:

```cpp
myKernel<<<grid, block>>>(data);
cudaDeviceSynchronize();  // Wait for kernel to finish
// Now safe to use results
```

By default, kernel launches are asynchronous - CPU continues immediately. Use `cudaDeviceSynchronize()` when you need results before continuing.

### Memory Allocation Patterns

**Do you need to allocate host memory every time?** No, it depends on your use case:

**Pattern 1: One-time computation**
```cpp
float *h_data = (float*)malloc(size);
// ... use once ...
free(h_data);
```

**Pattern 2: Repeated computations (reuse memory)**
```cpp
// Allocate ONCE before the loop
float *h_input = (float*)malloc(size);
float *h_output = (float*)malloc(size);

for(int i = 0; i < 100; i++){
    // Just update data, don't reallocate
    fillWithNewData(h_input);
    processOnGPU(h_input, h_output);
}

// Free ONCE after all iterations
free(h_input);
free(h_output);
```

**Pattern 3: Data already exists**
```cpp
void processExistingData(float *existingArray, int n){
    // No allocation needed - use the pointer directly
    cudaMemcpy(d_data, existingArray, n * sizeof(float), cudaMemcpyHostToDevice);
}
```

The same patterns apply to device memory - allocate once and reuse when possible for better performance.

---

## GPU Memory Types

| Memory | Speed | Size | Scope | Use Case |
|--------|-------|------|-------|----------|
| **Registers** | Fastest | Very small | Per thread | Local variables, loop counters |
| **Shared** | Very fast | Small (~48KB) | Per block | Data shared within a block |
| **Global** | Slow | Large (VRAM) | All threads | Main data arrays |
| **Constant** | Fast (cached) | Small (64KB) | All threads (read-only) | Constants, config params |
| **Local** | Slow | Per thread | Per thread | Register spills (avoid!) |

### Memory Hierarchy Diagram
```
┌─────────────────────────────────────────┐
│              GLOBAL MEMORY              │  ← Largest, slowest
│         (Accessible by all threads)     │
└─────────────────────────────────────────┘
         ↑                    ↑
    ┌────┴────┐          ┌────┴────┐
    │  Block  │          │  Block  │
    │ SHARED  │          │ SHARED  │      ← Fast, per-block
    │ MEMORY  │          │ MEMORY  │
    └────┬────┘          └────┬────┘
    ┌────┴────┐          ┌────┴────┐
    │REGISTERS│          │REGISTERS│      ← Fastest, per-thread
    └─────────┘          └─────────┘
```

---

## NVCC Compiler

NVCC (NVIDIA CUDA Compiler) handles both host and device code:

```
┌────────────────┐
│   .cu file     │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│     NVCC       │
└───────┬────────┘
        │
   ┌────┴────┐
   ▼         ▼
┌──────┐  ┌──────┐
│ Host │  │Device│
│ Code │  │ Code │
└──┬───┘  └──┬───┘
   │         │
   ▼         ▼
┌──────┐  ┌──────┐
│ x86  │  │ PTX  │  ← Intermediate representation
│binary│  │      │
└──┬───┘  └──┬───┘
   │         │
   │         ▼
   │      ┌──────┐
   │      │ JIT  │  ← Just-In-Time compilation
   │      │      │    to native GPU instructions
   │      └──┬───┘
   │         │
   └────┬────┘
        ▼
┌────────────────┐
│  Executable    │
└────────────────┘
```

**PTX (Parallel Thread Execution):**
- Intermediate representation for device code
- Stable across GPU generations
- JIT-compiled to native instructions at runtime
- Enables forward compatibility

---

## Complete Example

Here's everything put together - adding two arrays:

```cpp
#include <stdio.h>

// Kernel definition
__global__ void addArrays(int *a, int *b, int *c, int size) {
    // Calculate unique global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check (important!)
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 1000;
    size_t bytes = N * sizeof(int);

    // Host arrays
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device arrays
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    addArrays<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Copy results back
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify
    printf("Results: %d + %d = %d\n", h_a[0], h_b[0], h_c[0]);
    printf("         %d + %d = %d\n", h_a[N-1], h_b[N-1], h_c[N-1]);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```

---

## Quick Reference

```cpp
// Function qualifiers
__global__ void kernel();    // GPU, called from CPU
__device__ void helper();    // GPU, called from GPU
__host__ void cpuFunc();     // CPU only

// Thread indexing
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// Memory management
cudaMalloc(&d_ptr, size);
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
cudaFree(d_ptr);
cudaDeviceSynchronize();

// Kernel launch
dim3 block(256);
dim3 grid((N + 255) / 256);
kernel<<<grid, block>>>(args);
```

---

## Key Takeaways

1. **Host = CPU, Device = GPU** - Data must be explicitly copied between them
2. **`__global__` kernels** are your entry point to GPU execution
3. **Threads → Warps → Blocks → Grid** - The execution hierarchy
4. **Always bounds check** - You often launch more threads than needed
5. **Memory matters** - Use shared memory for speed, global for capacity
6. **Blocks are independent** - They can execute in any order
7. **Use `size_t` for sizes** - Prevents overflow with large allocations
8. **Block size should be a multiple of 32** - Aligns with warp size for efficiency
9. **Reuse memory when possible** - Allocate once, use many times
