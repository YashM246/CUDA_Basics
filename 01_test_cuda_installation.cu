/*
 * CUDA Installation Verification
 * ==============================
 *
 * Prerequisites:
 *   1. NVIDIA GPU (this repo tested with RTX 4070)
 *   2. CUDA Toolkit installed (https://developer.nvidia.com/cuda-downloads)
 *   3. Visual Studio Build Tools 2022 with "Desktop development with C++" workload
 *
 * Step 1: Verify CUDA installation
 *   Run these commands in terminal:
 *     nvcc --version    (should show CUDA version)
 *     nvidia-smi        (should show GPU info)
 *
 * Step 2: Compile and run
 *   Open terminal in VS Code (uses Developer PowerShell via .vscode/settings.json)
 *     nvcc 01_test_cuda_installation.cu -o out.exe
 *     .\out.exe
 *
 *   Expected output:
 *     Hello from GPU!
 *     Hello from CPU
 *
 * Troubleshooting:
 * ----------------
 * Error: "Cannot find compiler 'cl.exe' in PATH"
 *   -> Use Developer PowerShell for VS 2022, not regular PowerShell
 *   -> The .vscode/settings.json in this repo configures this automatically
 *
 * Error: "'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)"
 *   -> This happens when nvcc uses the x86 compiler instead of x64
 *   -> Fix: Prepend the x64 compiler path before compiling:
 *      $env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<VERSION>\bin\Hostx64\x64;" + $env:PATH
 *   -> Replace <VERSION> with your MSVC version (check in that folder)
 *   -> The .vscode/settings.json in this repo includes this fix
 */

#include <iostream>

// Simple CUDA Kernel
__global__ void helloFromGPU(){
    printf("Hello from GPU! \n");
}

int main(){
    helloFromGPU<<<1, 1>>>();

    cudaDeviceSynchronize();

    std::cout << "Hello from CPU" << std::endl;
    return 0;
}