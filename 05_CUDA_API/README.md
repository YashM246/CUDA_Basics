# CUDA APIs - cuBLAS & cuDNN

> Pre-built, highly optimized libraries from NVIDIA that you call instead of writing kernels from scratch.

## What is a "CUDA API"?

The term "API" here means a **precompiled library** where you can't see the internals. NVIDIA provides documentation on the function calls, but the actual source code is hidden - it's a compiled binary (`.dll` on Windows, `.so` on Linux). The code inside is **extremely optimized** (hand-tuned assembly, architecture-specific tricks) but you can't read or modify it.

**Think of it like this:**
```
Writing your own kernel:   You write the recipe AND cook the meal
Using a CUDA API:          A master chef cooks for you, you just place the order
```

You tell cuBLAS "multiply these two matrices" and it picks the fastest algorithm for your specific GPU. You'd need months of optimization work to match its performance.

---

## Opaque Types

CUDA APIs use **opaque struct types** - you get a handle but can't see inside it.

```cpp
cublasHandle_t handle;      // You can't access handle's internal fields
cublasCreate(&handle);       // CUDA initializes it for you
cublasSgemm(handle, ...);   // You pass it to functions
cublasDestroy(handle);      // You ask CUDA to clean it up
```

**Why opaque?**
- NVIDIA can change internals between CUDA versions without breaking your code
- Internal state may include GPU-specific optimizations, memory pools, etc.
- You interact through documented functions only

Common opaque types you'll encounter:

| Type | Library | Purpose |
|------|---------|---------|
| `cublasHandle_t` | cuBLAS | Context for BLAS operations |
| `cublasLtHandle_t` | cuBLAS Lt | Context for lightweight BLAS operations |
| `cudnnHandle_t` | cuDNN | Context for DNN operations |
| `cudnnTensorDescriptor_t` | cuDNN | Describes tensor shape, layout, data type |
| `cudnnFilterDescriptor_t` | cuDNN | Describes convolution filter/kernel |
| `cudnnConvolutionDescriptor_t` | cuDNN | Describes convolution operation parameters |

---

## Error Checking (API-Specific)

Each CUDA library has its **own error type and checking pattern**. Same concept as `CHECK_CUDA_ERROR` from the runtime API, but different types.

### cuBLAS Error Checking
```cpp
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                    __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage:
CUBLAS_CHECK(cublasCreate(&handle));
CUBLAS_CHECK(cublasSgemm(handle, ...));
```

### cuDNN Error Checking
```cpp
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage:
CUDNN_CHECK(cudnnCreate(&handle));
CUDNN_CHECK(cudnnSetTensor4dDescriptor(...));
```

**Notice the difference:** cuDNN has `cudnnGetErrorString()` for human-readable messages, while cuBLAS only gives you a numeric status code.

**Why use error checking?**
Without it, failures are **silent**. Your code keeps running with garbage data or crashes with a cryptic segfault. With the macro, you get the exact file, line, and error description.

> For more details: [Proper CUDA Error Checking](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/)

---

## cuBLAS - CUDA Basic Linear Algebra Subroutines

### What is BLAS?

BLAS (Basic Linear Algebra Subroutines) is a **standard specification** (since the 1970s) for common linear algebra operations. cuBLAS is NVIDIA's GPU-accelerated implementation.

BLAS operations are divided into 3 levels:

| Level | Operations | Example | Complexity |
|-------|-----------|---------|------------|
| **Level 1** | Vector-Vector | dot product, vector add, scalar multiply | O(n) |
| **Level 2** | Matrix-Vector | matrix-vector multiply, triangular solve | O(n^2) |
| **Level 3** | Matrix-Matrix | matrix multiply (GEMM), triangular solve | O(n^3) |

> **Level 3 is the most important for deep learning** - matrix multiplication (GEMM) is the core operation in neural networks.

### cuBLAS Naming Convention

cuBLAS function names follow a pattern:
```
cublas [precision] [operation]
```

| Prefix | Precision | Size |
|--------|-----------|------|
| `S` | Single precision (float) | 32-bit |
| `D` | Double precision (double) | 64-bit |
| `H` | Half precision (FP16) | 16-bit |
| `C` | Single complex | 2x 32-bit |
| `Z` | Double complex | 2x 64-bit |

**Examples:**
- `cublasSgemm` - **S**ingle precision **G**eneral **M**atrix **M**ultiply
- `cublasDgemm` - **D**ouble precision **G**eneral **M**atrix **M**ultiply
- `cublasHgemm` - **H**alf precision **G**eneral **M**atrix **M**ultiply

### GEMM - The Most Important Operation

GEMM computes: **C = alpha * A * B + beta * C**

```cpp
cublasSgemm(handle,
    CUBLAS_OP_N,    // Don't transpose A
    CUBLAS_OP_N,    // Don't transpose B
    m, n, k,        // Dimensions: A is m×k, B is k×n, C is m×n
    &alpha,          // Scalar multiplier for A*B
    d_A, lda,       // Matrix A and its leading dimension
    d_B, ldb,       // Matrix B and its leading dimension
    &beta,           // Scalar multiplier for existing C
    d_C, ldc         // Matrix C and its leading dimension
);
```

### cuBLAS General Workflow
```
1. cublasCreate(&handle)          // Create context
2. Allocate device memory         // cudaMalloc
3. Copy data to device            // cudaMemcpy or cublasSetMatrix
4. Call cuBLAS function            // cublasSgemm, etc.
5. Copy results back              // cudaMemcpy or cublasGetMatrix
6. cublasDestroy(handle)          // Cleanup
```

### cuBLAS vs cuBLASLt

| Feature | cuBLAS | cuBLASLt (Lightweight) |
|---------|--------|----------------------|
| **API Style** | Simple, one-call functions | Flexible, descriptor-based |
| **Algorithm Selection** | Automatic | Manual (can tune for best performance) |
| **Mixed Precision** | Limited | Full support (FP16 input → FP32 output) |
| **Tensor Core Support** | Implicit | Explicit control |
| **Use Case** | Quick & easy GEMM | Maximum performance tuning |

> cuBLASLt is what frameworks like PyTorch use under the hood for matrix multiplication.

### IMPORTANT: Column-Major Order

cuBLAS uses **column-major** order (inherited from Fortran/BLAS standard), while C/C++ uses **row-major**:

```
Matrix:  | 1  2  3 |
         | 4  5  6 |

Row-major (C/C++):      [1, 2, 3, 4, 5, 6]     (row by row)
Column-major (cuBLAS):   [1, 4, 2, 5, 3, 6]     (column by column)
```

If you ignore this, your results will be **transposed** or **wrong**. Solutions:
1. Store matrices in column-major order from the start
2. Use transpose flags (`CUBLAS_OP_T`) to handle the conversion
3. Swap A and B: compute B*A instead of A*B (common trick)

### Compiling with cuBLAS
```bash
nvcc program.cu -o program.exe -lcublas
```

---

## cuDNN - CUDA Deep Neural Network Library

### What is cuDNN?

cuDNN is NVIDIA's GPU-accelerated library for **deep learning primitives**:
- Convolutions (forward + backward)
- Pooling
- Normalization (BatchNorm, LayerNorm)
- Activation functions (ReLU, Sigmoid, Tanh)
- Softmax
- RNNs (LSTM, GRU)
- Attention / Transformer operations

Every major deep learning framework (PyTorch, TensorFlow, JAX) uses cuDNN under the hood.

### cuDNN Workflow Pattern

cuDNN follows a **descriptor-based pattern** - you describe everything before executing:

```
1. cudnnCreate(&handle)                          // Create context
2. Create descriptors (tensor, filter, etc.)     // Describe shapes & types
3. Set descriptor values                          // Fill in the details
4. Find best algorithm                            // cudnnGetConvolutionForwardAlgorithm
5. Allocate workspace                             // Some algorithms need temp memory
6. Execute operation                              // cudnnConvolutionForward, etc.
7. Destroy descriptors                            // Cleanup
8. cudnnDestroy(handle)                           // Destroy context
```

**Why so many steps?** cuDNN needs all this info to pick the fastest algorithm. Different tensor sizes, data types, and GPU architectures benefit from different strategies.

### Tensor Descriptors

cuDNN needs to know the shape and layout of your data:

```cpp
cudnnTensorDescriptor_t desc;
cudnnCreateTensorDescriptor(&desc);
cudnnSetTensor4dDescriptor(desc,
    CUDNN_TENSOR_NCHW,    // Layout: Batch, Channels, Height, Width
    CUDNN_DATA_FLOAT,     // Data type
    batch_size,            // N
    channels,              // C
    height,                // H
    width                  // W
);
```

**Common data layouts:**

| Layout | Order | Used By |
|--------|-------|---------|
| `NCHW` | Batch, Channels, Height, Width | PyTorch default |
| `NHWC` | Batch, Height, Width, Channels | TensorFlow default, faster on Tensor Cores |

### Convolution Example (High-Level)

```cpp
// 1. Create descriptors
cudnnCreateTensorDescriptor(&inputDesc);
cudnnCreateFilterDescriptor(&filterDesc);
cudnnCreateConvolutionDescriptor(&convDesc);
cudnnCreateTensorDescriptor(&outputDesc);

// 2. Set descriptors (shapes, data types, padding, stride...)
cudnnSetTensor4dDescriptor(inputDesc, ...);
cudnnSetFilter4dDescriptor(filterDesc, ...);
cudnnSetConvolution2dDescriptor(convDesc, pad, pad, stride, stride, ...);

// 3. Find the fastest algorithm for this specific configuration
cudnnGetConvolutionForwardAlgorithm_v7(handle, inputDesc, filterDesc,
    convDesc, outputDesc, numRequestedAlgos, &numReturnedAlgos, perfResults);

// 4. Allocate workspace memory (some algorithms need scratch space)
cudnnGetConvolutionForwardWorkspaceSize(handle, ..., &workspaceSize);
cudaMalloc(&workspace, workspaceSize);

// 5. Execute the convolution
cudnnConvolutionForward(handle, &alpha, inputDesc, d_input,
    filterDesc, d_filter, convDesc, algo, workspace, workspaceSize,
    &beta, outputDesc, d_output);
```

### cuDNN Algorithm Selection

cuDNN offers multiple algorithms for the same operation (e.g., convolution):

| Algorithm | Best For | Memory |
|-----------|----------|--------|
| `IMPLICIT_GEMM` | Small inputs | Low |
| `IMPLICIT_PRECOMP_GEMM` | General purpose | Medium |
| `GEMM` | Large inputs | High (needs workspace) |
| `WINOGRAD` | 3x3 filters | Medium |
| `FFT` | Large filters | High |

You can let cuDNN benchmark and auto-select the fastest one. This is what `torch.backends.cudnn.benchmark = True` does in PyTorch.

### Compiling with cuDNN
```bash
nvcc program.cu -o program.exe -lcudnn
```

---

## Matrix Multiplication: cuBLAS vs cuDNN

| Feature | cuBLAS | cuDNN |
|---------|--------|-------|
| **Direct GEMM support** | Yes (primary purpose) | Implicit through operations |
| **Best for matmul** | Yes | No (use cuBLAS instead) |
| **Deep learning ops** | Only GEMM | Convolutions, pooling, normalization, etc. |
| **Ease of use for matmul** | Simple | Overkill |

> **Use cuBLAS for matrix multiplication, cuDNN for deep learning layers.**

---

## Navigating NVIDIA Documentation

NVIDIA's docs are extensive but can be hard to navigate. Tips:

1. **[NVIDIA cuBLAS Docs](https://docs.nvidia.com/cuda/cublas/)** - Official reference
2. **[NVIDIA cuDNN Docs](https://docs.nvidia.com/deeplearning/cudnn/)** - Official reference
3. **[CUDA Library Samples](https://github.com/NVIDIA/CUDALibrarySamples)** - Working code examples
4. **Search trick:** Google `site:docs.nvidia.com cublasSgemm` for specific functions

---

## Folder Contents

| Folder/File | Description |
|-------------|-------------|
| `README.md` | This guide |
| `cuBLAS/` | cuBLAS examples and comparison document |
| `cuBLAS/Comparison.md` | Detailed comparison of cuBLAS variants (cuBLAS, cuBLASLt, cuBLASXt, cuBLASDx, CUTLASS) |
| `cuBLAS/01_sgemm_hgemm_cublas.cu` | cuBLAS basics: SGEMM (FP32) vs HGEMM (FP16), row-major swap trick |
| `cuBLAS/02_matmul_cublaslt.cu` | cuBLASLt descriptor-based API: FP32 vs FP16 matmul |
| `cuBLAS/03_compare_cublas_cublaslt.cu` | Benchmark: cuBLAS vs cuBLASLt vs naive kernel (GFLOPS comparison) |
| `cuBLAS/04_matmul_cublasxt.cu` | cuBLASXt: host-pointer matmul with multi-GPU support |
| `cuBLAS/05_compare_cublas_cublasxt.cu` | Benchmark: cuBLAS vs cuBLASXt on large 16384x16384 matrices |
| `cuDNN/` | cuDNN overview and examples |
| `cuDNN/README.md` | cuDNN guide: descriptors, convolution workflow, Graph API, kernel fusion, engine types |
| `cuDNN/01_tanh.cu` | cuDNN activation (tanh) vs naive CUDA kernel benchmark |
| `cuDNN/02_compare_tanh.py` | PyTorch built-in tanh (cuDNN) vs custom tanh formula |
