# CUDA Programming

A learning repository for CUDA programming, starting from C/C++ fundamentals and progressing to GPU programming with NVIDIA CUDA.

## Prerequisites

- NVIDIA GPU (tested with RTX 4070)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (tested with CUDA 12.6)
- Visual Studio Build Tools 2022 with "Desktop development with C++" workload
- VS Code (optional, but recommended)

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
├── 01_test_cuda_installation.cu    # Verify CUDA setup
├── 02_c_and_c++_review/            # C/C++ fundamentals review
│   └── ...
├── .vscode/                        # VS Code configuration
│   └── settings.json
└── README.md
```

## Contents

### Part 1: C/C++ Review

A quick review of C and C++ concepts essential for CUDA programming:
- Pointers and memory management
- Arrays and dynamic allocation
- Structs and classes
- Function pointers

### Part 2: CUDA Programming

Core CUDA concepts and parallel programming:
- CUDA kernel basics
- Thread hierarchy (grids, blocks, threads)
- Memory management (host vs device)
- Memory types (global, shared, constant)
- Synchronization
- Performance optimization

## Troubleshooting

See the comments in `01_test_cuda_installation.cu` for common setup issues and fixes.
