# Compilers: From Source Code to Executable

## What is a Compiler?

A **compiler** translates human-readable source code into machine code (binary) that your CPU/GPU can execute.

```
Source Code (.c, .cpp, .cu)  →  Compiler  →  Executable (.exe, .out)
```

---

## The Compilation Process

Compilation happens in **4 stages**:

### 1. Preprocessing
- Handles all `#` directives (`#include`, `#define`, `#ifdef`)
- Expands macros and includes header files
- Removes comments
- Output: Expanded source code

```bash
# View preprocessor output
cl /P myfile.c        # Creates myfile.i (Windows)
gcc -E myfile.c -o myfile.i   # Linux/Mac
```

### 2. Compilation
- Converts preprocessed code to **assembly language**
- Performs syntax checking and optimizations
- Output: Assembly code (.s or .asm)

```bash
# View assembly output
cl /FA myfile.c       # Creates myfile.asm (Windows)
gcc -S myfile.c       # Creates myfile.s (Linux/Mac)
```

### 3. Assembly
- Converts assembly code to **object code** (machine code)
- Output: Object file (.obj on Windows, .o on Linux)

```bash
# Create object file only
cl /c myfile.c        # Creates myfile.obj (Windows)
gcc -c myfile.c       # Creates myfile.o (Linux/Mac)
```

### 4. Linking
- Combines object files with libraries
- Resolves external references (function calls between files)
- Output: Executable (.exe on Windows, no extension on Linux)

```bash
# Full compilation (all stages)
cl myfile.c /Fe:output.exe    # Windows
gcc myfile.c -o output        # Linux/Mac
```

---

## Visual Diagram

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│ Source Code │ --> │ Preprocessor │ --> │   Compiler   │ --> │  Assembler  │ --> │   Linker   │
│   (.c/.cu)  │     │  (expand #)  │     │ (to assembly)│     │ (to object) │     │ (to .exe)  │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘     └────────────┘
                           │                    │                    │                   │
                           ▼                    ▼                    ▼                   ▼
                        (.i file)           (.s/.asm)            (.obj/.o)           (.exe)
```

---

## Common Compilers

| Compiler | Platform | Languages | Command |
|----------|----------|-----------|---------|
| **MSVC (cl.exe)** | Windows | C, C++ | `cl file.c` |
| **GCC** | Linux, Mac, Windows | C, C++ | `gcc file.c` |
| **Clang** | Cross-platform | C, C++ | `clang file.c` |
| **NVCC** | Cross-platform | CUDA | `nvcc file.cu` |

---

## MSVC (Microsoft Visual C++) - `cl.exe`

The compiler we use on Windows for this project.

### Common Flags

| Flag | Description | Example |
|------|-------------|---------|
| `/Fe:<name>` | Set output executable name | `cl file.c /Fe:out.exe` |
| `/Fo:<name>` | Set output object file name | `cl /c file.c /Fo:out.obj` |
| `/c` | Compile only (no linking) | `cl /c file.c` |
| `/O2` | Optimize for speed | `cl /O2 file.c` |
| `/Od` | Disable optimization (debug) | `cl /Od file.c` |
| `/Zi` | Generate debug info | `cl /Zi file.c` |
| `/W4` | Warning level 4 (recommended) | `cl /W4 file.c` |
| `/Wall` | Enable all warnings | `cl /Wall file.c` |
| `/P` | Preprocess only (output .i) | `cl /P file.c` |
| `/FA` | Generate assembly listing | `cl /FA file.c` |

### Examples

```powershell
# Basic compilation
cl myfile.c

# Named output
cl myfile.c /Fe:program.exe

# With optimizations and warnings
cl /O2 /W4 myfile.c /Fe:program.exe

# Debug build
cl /Od /Zi myfile.c /Fe:debug.exe

# Compile multiple files
cl file1.c file2.c /Fe:program.exe
```

---

## NVCC (NVIDIA CUDA Compiler)

NVCC compiles CUDA code (.cu files) and works alongside a host compiler (like MSVC).

### How NVCC Works

```
┌──────────────┐
│  .cu file    │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌─────────────────┐
│    NVCC      │ --> │ Device Code     │ --> PTX/cubin (GPU)
│  (splitter)  │     │ (__global__, etc)│
└──────┬───────┘     └─────────────────┘
       │
       ▼
┌─────────────────┐
│   Host Code     │ --> cl.exe/gcc (CPU)
│ (regular C/C++) │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│   Linker        │ --> Final Executable
└─────────────────┘
```

**NVCC splits your code:**
- **Device code** (`__global__`, `__device__`) → Compiled to GPU binary (PTX)
- **Host code** (regular C/C++) → Sent to host compiler (cl.exe on Windows)

### Common NVCC Flags

| Flag | Description | Example |
|------|-------------|---------|
| `-o <name>` | Output file name | `nvcc file.cu -o out.exe` |
| `-c` | Compile only (no linking) | `nvcc -c file.cu` |
| `-arch=sm_XX` | Target GPU architecture | `nvcc -arch=sm_89 file.cu` |
| `-G` | Debug device code | `nvcc -G file.cu` |
| `-g` | Debug host code | `nvcc -g file.cu` |
| `-O3` | Optimization level 3 | `nvcc -O3 file.cu` |
| `--ptxas-options=-v` | Show register/memory usage | `nvcc --ptxas-options=-v file.cu` |
| `-Xcompiler <flag>` | Pass flag to host compiler | `nvcc -Xcompiler /W4 file.cu` |

### GPU Architecture Codes (Compute Capability)

| GPU Series | Architecture | SM Code |
|------------|--------------|---------|
| RTX 4000 (Ada) | Ada Lovelace | `sm_89` |
| RTX 3000 (Ampere) | Ampere | `sm_86` |
| RTX 2000 (Turing) | Turing | `sm_75` |
| GTX 1000 (Pascal) | Pascal | `sm_61` |

**For RTX 4070:** Use `-arch=sm_89`

### Examples

```powershell
# Basic CUDA compilation
nvcc hello.cu -o hello.exe

# Target specific GPU architecture (RTX 4070)
nvcc -arch=sm_89 kernel.cu -o kernel.exe

# With optimizations
nvcc -O3 -arch=sm_89 kernel.cu -o kernel.exe

# Debug build (both host and device)
nvcc -g -G kernel.cu -o debug.exe

# Pass flags to MSVC
nvcc -Xcompiler "/W4 /O2" kernel.cu -o kernel.exe

# See GPU resource usage
nvcc --ptxas-options=-v kernel.cu -o kernel.exe
```

---

## Compilation Errors vs Warnings

### Errors (Compilation stops)
```c
int main() {
    int x = "hello";  // Error: can't assign string to int
    printf("%d", y);  // Error: 'y' undeclared
}
```

### Warnings (Compilation continues)
```c
int main() {
    int x;
    printf("%d", x);  // Warning: 'x' used uninitialized
    return;           // Warning: no return value
}
```

**Best Practice:** Treat warnings as errors with `/WX` (MSVC) or `-Werror` (GCC)

---

## Common Errors and Solutions

### 1. "Cannot find compiler 'cl.exe' in PATH"
**Cause:** NVCC can't find MSVC
**Solution:** Use Developer PowerShell or set up PATH correctly

### 2. "unresolved external symbol"
**Cause:** Linker can't find function definition
**Solution:**
- Make sure all .c/.cu files are included in compilation
- Check for missing libraries

### 3. "cudafe++ died with status 0xC0000005"
**Cause:** x86/x64 architecture mismatch
**Solution:** Use `-Arch amd64` when launching Developer PowerShell

### 4. "identifier not found"
**Cause:** Variable/function used before declaration
**Solution:** Declare or include the proper header

---

## Debug vs Release Builds

| Aspect | Debug | Release |
|--------|-------|---------|
| **Optimization** | Disabled (`/Od`, `-O0`) | Enabled (`/O2`, `-O3`) |
| **Debug Symbols** | Included (`/Zi`, `-g`) | Excluded |
| **Speed** | Slow | Fast |
| **Size** | Larger | Smaller |
| **Debuggable** | Yes | Limited |

```powershell
# Debug build
cl /Od /Zi myfile.c /Fe:debug.exe
nvcc -g -G kernel.cu -o debug.exe

# Release build
cl /O2 myfile.c /Fe:release.exe
nvcc -O3 kernel.cu -o release.exe
```

---

## Multi-File Projects

### Separate Compilation
```powershell
# Compile each file to object
cl /c main.c /Fo:main.obj
cl /c utils.c /Fo:utils.obj

# Link together
cl main.obj utils.obj /Fe:program.exe
```

### CUDA with Multiple Files
```powershell
# Compile CUDA and C files together
nvcc main.cu utils.c -o program.exe

# Or separate compilation
nvcc -c kernel.cu -o kernel.obj
cl /c main.c /Fo:main.obj
nvcc kernel.obj main.obj -o program.exe
```

---

## Quick Reference

### MSVC (cl.exe)
```powershell
cl file.c                    # Basic compile
cl file.c /Fe:out.exe        # Named output
cl /O2 /W4 file.c            # Optimized with warnings
cl /c file.c                 # Compile only
```

### NVCC
```powershell
nvcc file.cu -o out.exe      # Basic compile
nvcc -arch=sm_89 file.cu     # Target GPU
nvcc -O3 file.cu             # Optimized
nvcc -g -G file.cu           # Debug
```

### Check Versions
```powershell
cl                           # Shows MSVC version
nvcc --version               # Shows CUDA version
nvidia-smi                   # Shows GPU and driver info
```

---

## Summary

1. **Compilation is a 4-stage process:** Preprocess → Compile → Assemble → Link
2. **MSVC (cl.exe)** compiles C/C++ on Windows
3. **NVCC** compiles CUDA code, using MSVC for host code
4. **Always specify GPU architecture** with `-arch=sm_XX` for CUDA
5. **Use Developer PowerShell** to ensure cl.exe is available for NVCC
6. **Debug builds** for development, **Release builds** for performance
