# cuBLAS Library Variants Comparison

> Note: Before benchmarking, always do warmup runs first. Without warmup,
> cuBLAS has significant first-run overhead (~45ms) from library initialization,
> JIT compilation, and GPU context setup. Benchmark runs should be averaged
> over multiple iterations for accurate results.


## cuBLAS (Standard)

NVIDIA CUDA Basic Linear Algebra Subprograms - a GPU-accelerated library
for AI and HPC applications.

- Provides industry-standard BLAS APIs (Level 1, 2, 3)
- GEMM (General Matrix Multiply) is the most-used operation
- Runs operations on a SINGLE GPU
- Host-side API: you call from CPU, it executes on GPU
- Automatic algorithm selection (picks the fastest kernel for your problem)
- Column-major storage (inherited from Fortran BLAS)
  - Pay attention to your shaping when using row-major C/C++ data!
  - Reference: https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication

Best for: Standard BLAS operations on a single GPU with minimal setup.


## cuBLAS-Lt (Lightweight)

cuBLASLt is an extension that provides a more flexible, descriptor-based API.
Almost all its datatypes and API calls revolve around matrix multiplication.

Key differences from standard cuBLAS:
- Flexible API: Uses descriptors to configure operations (like cuDNN)
- Algorithm selection: YOU choose the algorithm (can benchmark to find fastest)
- Mixed precision: Full support for FP16, BF16, FP8, INT8 inputs
- Tensor Core control: Explicit control over Tensor Core usage
- Epilogue fusions: Can fuse bias add, ReLU, GELU after GEMM in one kernel
- Problem decomposition: If a problem can't run in a single kernel,
  cuBLASLt decomposes it into sub-problems automatically

When to use cuBLASLt over cuBLAS:
- You need mixed precision (FP16/FP8/INT8 compute)
- You want to tune algorithm selection for maximum performance
- You need epilogue fusions (GEMM + activation in one kernel)
- You're building a deep learning framework

This is what PyTorch and TensorFlow use under the hood for matmul.


## cuBLAS-Xt (Multi-GPU)

cuBLASXt enables BLAS operations across MULTIPLE GPUs and host CPU.

Key features:
- Multi-GPU: Distributes BLAS operations across multiple GPUs
- Host + GPU: Can use both CPU and GPU memory for computation
- Thread-safe: Concurrent execution on different GPUs
- Automatic work partitioning: Splits matrices across devices

The catch - it's SLOWER for single-GPU workloads because:
- Data must transfer between motherboard DRAM and GPU VRAM
- Memory bandwidth bottleneck between CPU and GPU (PCIe)
- Synchronization overhead between devices

Benchmark example:
  (M, N) @ (N, K) where M = N = K = 16384
  cuBLAS (single GPU) is significantly faster than cuBLAS-Xt on one GPU
  cuBLAS-Xt only wins when the problem is too large for one GPU's memory

When to use cuBLAS-Xt:
- Problem exceeds single GPU memory
- You have multiple GPUs and a large enough problem to justify the overhead
- Scaling across GPUs is more important than single-GPU latency


## cuBLASDx (Device Extensions)

** NOT used in this course **

cuBLASDx is a DEVICE-SIDE API - you call BLAS operations from INSIDE
CUDA kernels (not from the host).

- Enables fusing BLAS with other operations in a single kernel
- Reduces latency by avoiding kernel launch overhead
- Preview/experimental library (not part of CUDA Toolkit)
- Must be downloaded separately: https://developer.nvidia.com/cublasdx-downloads
- Documentation: https://docs.nvidia.com/cuda/cublasdx

When you'd use it:
- You're writing custom CUDA kernels and need GEMM as a building block
- You want to fuse GEMM with other operations at the thread-block level
- You need maximum performance and are comfortable with device-side programming


## CUTLASS (CUDA Templates for Linear Algebra)

CUTLASS is NVIDIA's open-source C++ template library for matrix multiplication.
GitHub: https://github.com/NVIDIA/cutlass

Why CUTLASS exists:
- cuBLAS and variants are HOST-side, opaque libraries (can't see the code)
- cuBLASDx is device-side but not well-documented or mature
- Matrix multiplication is THE most important operation in deep learning
- cuBLAS doesn't let you easily FUSE operations together

CUTLASS provides:
- Open-source CUDA C++ templates you CAN read and modify
- Fine-grained control over tiling, memory access, Tensor Core usage
- Easy operation fusion (GEMM + custom epilogues)
- Composable building blocks for custom kernels

Note: Flash Attention does NOT use CUTLASS - it uses hand-written optimized
CUDA kernels. (Source: https://arxiv.org/pdf/2205.14135)


## Quick Comparison Table

  Library      | Runs On         | API Side | Open Source | Precision Support      | Best For
  -------------|-----------------|----------|-------------|------------------------|---------
  cuBLAS       | Single GPU      | Host     | No          | FP32, FP64, FP16      | Simple GEMM, standard BLAS
  cuBLAS-Lt    | Single GPU      | Host     | No          | FP32, FP16, BF16, INT8 | DL frameworks, tuned GEMM
  cuBLAS-Xt    | Multi-GPU + CPU | Host     | No          | FP32, FP64             | Large problems, multi-GPU
  cuBLASDx     | Single GPU      | Device   | No          | FP32, FP16             | In-kernel BLAS (experimental)
  CUTLASS      | Single GPU      | Device   | Yes         | All                    | Custom kernels, fusion, research


## Decision Guide

  Need simple matrix multiply?
    --> cuBLAS

  Need mixed precision or algorithm tuning?
    --> cuBLAS-Lt

  Problem too large for one GPU?
    --> cuBLAS-Xt

  Need GEMM inside your own kernel?
    --> cuBLASDx or CUTLASS

  Need to fuse operations or customize the GEMM?
    --> CUTLASS

  Building a deep learning framework?
    --> cuBLAS-Lt (what PyTorch/TensorFlow use)
