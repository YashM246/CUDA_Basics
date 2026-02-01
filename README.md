# CUDA Programming

A learning repository for CUDA programming, starting from C/C++ fundamentals and progressing to GPU programming with NVIDIA CUDA.

## Prerequisites

- NVIDIA GPU (tested with RTX 4070)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (tested with CUDA 12.6)
- Visual Studio Build Tools 2022 with "Desktop development with C++" workload
- VS Code (optional, but recommended)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) (for profiling, optional)

## Getting Started

1. Clone this repository
2. Open in VS Code
3. Open terminal (Ctrl + `) - it will automatically use Developer PowerShell
4. Verify CUDA installation:
   ```powershell
   nvcc --version
   nvidia-smi
   ```
5. Compile and run the test file:
   ```powershell
   nvcc 01_test_cuda_installation.cu -o out.exe
   .\out.exe
   ```

## Repository Structure

```
CUDA_Programming/
├── 01_test_cuda_installation.cu        # Verify CUDA setup
├── 02_c_and_c++_review/                # C/C++ fundamentals review
│   ├── 2.1_Pointers/                   # Pointer concepts (6 examples)
│   ├── 2.2_Custom_Types/               # Structs, unions, enums
│   ├── 2.3_Type_Casting/               # Type conversion
│   ├── 2.4_Macros_and_Global_Variables/
│   ├── 2.5_Compilers/                  # Compiler concepts (Compilers.md)
│   ├── 2.6_Makefiles/                  # Build automation (Makefiles.md)
│   └── 2.7_Debuggers/                  # Debugging tools (Debuggers.md)
├── 03_CPUs_and_GPUs.md                 # CPU vs GPU architecture comparison
├── 04_Writing_your_First_Kernels/      # CUDA programming
│   ├── 4.1_CUDA_Basics/
│   │   ├── CUDA_Basics.md              # Thread indexing, memory, block sizes
│   │   └── 01_indexing.cu              # Thread/block index demonstration
│   ├── 4.2_Kernels/
│   │   ├── Kernel_Concepts.md          # Launch config, sync, SIMT model
│   │   ├── 01_vector_add_v1.cu         # CPU vs GPU vector addition benchmark
│   │   ├── 02_vector_add_v2.cu         # 1D vs 3D grid comparison
│   │   └── 03_matmul.cu                # Naive matrix multiplication
│   └── 4.3_Profiling/
│       ├── 01_nvtx_matmul.cu           # NVTX profiling annotations demo
│       └── 02_tiled_matmul.cu          # Optimized tiled matrix multiplication
└── README.md
```

## Contents

### Part 1: Test CUDA Installation
- `01_test_cuda_installation.cu` - Verify your CUDA setup is working correctly

### Part 2: C/C++ Review

A quick review of C and C++ concepts essential for CUDA programming:

| Section | Topics |
|---------|--------|
| 2.1 Pointers | Address-of (&), dereference (*), multi-level pointers, void pointers, arrays |
| 2.2 Custom Types | Structs, unions, enums, typedef |
| 2.3 Type Casting | Implicit/explicit conversion, pointer casting |
| 2.4 Macros | #define, global variables, preprocessor |
| 2.5 Compilers | GCC, Clang, MSVC, compilation process |
| 2.6 Makefiles | Build automation, targets, dependencies |
| 2.7 Debuggers | GDB, LLDB, debugging techniques |

### Part 3: CPUs and GPUs

`03_CPUs_and_GPUs.md` - Understanding the architectural differences:
- CPU: Few powerful cores optimized for sequential tasks
- GPU: Thousands of simple cores optimized for parallel tasks
- When to use each, and why GPUs excel at data-parallel workloads

### Part 4: Writing Your First Kernels

#### 4.1 CUDA Basics
- **CUDA_Basics.md** - Comprehensive guide covering:
  - Execution model (Grid → Blocks → Warps → Threads)
  - Thread indexing (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`)
  - Memory hierarchy (global, shared, registers)
  - Choosing block sizes (warp alignment, occupancy)
  - Memory allocation patterns
- **01_indexing.cu** - See thread/block indices in action

#### 4.2 Kernels
- **Kernel_Concepts.md** - Launch configuration, synchronization, SIMT model
- **01_vector_add_v1.cu** - CPU vs GPU vector addition with benchmarking
- **02_vector_add_v2.cu** - Compare 1D vs 3D grid configurations
- **03_matmul.cu** - Naive matrix multiplication (understanding the baseline)

#### 4.3 Profiling
- **01_nvtx_matmul.cu** - Using NVTX annotations for profiling visibility
- **02_tiled_matmul.cu** - Optimized matrix multiplication with shared memory tiling
  - Demonstrates 5-10x speedup over naive implementation
  - Extensive comments explaining the tiling algorithm

## Compiling CUDA Programs

```powershell
# Basic compilation
nvcc program.cu -o program.exe

# Run the program
.\program.exe

# Profile with Nsight Systems (if installed)
nsys profile .\program.exe
```

## Key Concepts Covered

| Concept | Description | Example File |
|---------|-------------|--------------|
| Thread Indexing | Calculate global thread ID | `01_indexing.cu` |
| Memory Transfer | Host ↔ Device data movement | `01_vector_add_v1.cu` |
| 1D vs 3D Grids | Different grid configurations | `02_vector_add_v2.cu` |
| Naive MatMul | Understanding memory-bound kernels | `03_matmul.cu` |
| NVTX Profiling | Annotate code for profilers | `01_nvtx_matmul.cu` |
| Shared Memory | Tiled algorithms for optimization | `02_tiled_matmul.cu` |

## Performance Results (RTX 4070)

Example results from `02_tiled_matmul.cu` (1024x1024 matrices):

| Implementation | Time | GFLOPS | Speedup |
|----------------|------|--------|---------|
| CPU | ~2500 ms | ~0.9 | 1x |
| GPU (Naive) | ~15 ms | ~140 | ~170x vs CPU |
| GPU (Tiled) | ~3 ms | ~700 | ~5x vs Naive |

*Results may vary based on hardware and system load*

## Troubleshooting

See the comments in `01_test_cuda_installation.cu` for common setup issues and fixes.

Common issues:
- **nvcc not found**: Ensure CUDA bin directory is in PATH
- **No CUDA-capable device**: Check `nvidia-smi` output
- **Linker errors**: Ensure Visual Studio Build Tools are installed
