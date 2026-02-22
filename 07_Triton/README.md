# Triton

Triton is an open-source programming language and compiler for writing highly efficient GPU kernels using Python. It abstracts away low-level CUDA complexities while achieving performance comparable to hand-written CUDA or cuBLAS/cuDNN.

**Key Innovation:** Triton shifts from scalar programs to **block programs**, letting the compiler handle thread-level details.

---

## The Paradigm Shift: Blocked Program + Scalar Threads

Understanding this distinction is critical to using Triton effectively.

### CUDA: Scalar Program + Blocked Threads

In CUDA, you write a **scalar program** (operates on single elements) and launch it with **blocked threads**:

```cuda
// CUDA: Each thread processes ONE element
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];  // Scalar operation
    }
}

// Launch with 256 threads per block
add_kernel<<<(n + 255) / 256, 256>>>(a, b, c, n);
```

You must manually:
- Calculate thread indices (`blockIdx.x * blockDim.x + threadIdx.x`)
- Handle boundary conditions (`if (tid < n)`)
- Manage shared memory, synchronization, coalesced access
- Reason about warp divergence, occupancy, register pressure

### Triton: Blocked Program + Scalar Threads

In Triton, you write a **block program** (operates on tiles) and the compiler generates **scalar threads**:

```python
# Triton: Each program instance processes a TILE of elements
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    # This program instance handles elements [pid * BLOCK_SIZE : (pid+1) * BLOCK_SIZE]
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load a TILE of data (compiler generates threads for you)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    c = a + b  # Operate on the entire tile

    tl.store(c_ptr + offsets, c, mask=mask)

# Launch one program instance per tile
grid = (triton.cdiv(n, BLOCK_SIZE),)
add_kernel[grid](a, b, c, n, BLOCK_SIZE=1024)
```

The compiler automatically:
- Generates the scalar threads within each block
- Optimizes memory coalescing
- Manages shared memory for tiling
- Vectorizes loads/stores where possible

![](../assets/triton1.png)
![](../assets/triton2.png)

**Bottom line:**
- **CUDA:** You write at the thread level (scalar), worry about blocks yourself
- **Triton:** You write at the block level (tiles), compiler worries about threads

---

## What This Means in Practice

### 1. Higher-Level Abstraction for Deep Learning

Triton is designed for operations common in neural networks:
- Matrix multiplication (GEMM)
- Activation functions (ReLU, GELU, Softmax)
- Convolutions
- Layer normalization, batch normalization

You write these at a **block level**, similar to how you think about the algorithm mathematically, not how threads execute it.

### 2. Compiler Handles Boilerplate

The Triton compiler automatically optimizes:
- **Memory tiling:** Breaks operations into tiles that fit in shared memory
- **Coalesced access:** Ensures adjacent threads access adjacent memory
- **Vectorization:** Uses wide loads (128-bit) where possible
- **Instruction scheduling:** Hides memory latency with computation

### 3. Performance Comparable to Hand-Tuned CUDA

Because the compiler applies established optimization patterns, Triton kernels often match or exceed:
- cuBLAS (NVIDIA's matrix multiplication library)
- cuDNN (NVIDIA's deep learning primitives)
- Hand-written CUDA by expert GPU programmers

**Example:** PyTorch's `torch.compile` uses Triton to generate fused kernels for model layers, achieving speedups without manual kernel writing.

---

## Key Triton Concepts

### 1. Program ID (`tl.program_id`)

Identifies which **tile** this program instance is processing (analogous to `blockIdx` in CUDA).

```python
pid = tl.program_id(0)  # 0 = x-axis, 1 = y-axis, 2 = z-axis
```

### 2. Tile Pointers and Offsets

You work with **ranges of indices** (tiles) instead of single elements:

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# Example: pid=2, BLOCK_SIZE=256 → offsets = [512, 513, ..., 767]
```

### 3. Masked Loads/Stores

Handle boundary conditions with masks instead of `if` statements:

```python
mask = offsets < n
a = tl.load(a_ptr + offsets, mask=mask)  # Out-of-bounds elements → 0
tl.store(c_ptr + offsets, c, mask=mask)  # Out-of-bounds writes ignored
```

This avoids warp divergence (no `if` branches within a warp).

### 4. JIT Compilation

Triton kernels are compiled **just-in-time** when first called:

```python
@triton.jit
def my_kernel(...):
    pass

# First call: compiles CUDA kernel from Triton IR
my_kernel[grid](..., BLOCK_SIZE=256)

# Subsequent calls: uses cached binary
my_kernel[grid](..., BLOCK_SIZE=256)
```

Compilation happens **per unique set of template parameters** (`BLOCK_SIZE`, `num_stages`, etc.).

### 5. `tl.constexpr` for Template Parameters

Mark compile-time constants to enable aggressive optimizations:

```python
def kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compiler unrolls loops with constexpr bounds
    for i in tl.static_range(BLOCK_SIZE):
        ...
```

---

## When to Use Triton vs CUDA

| Use **Triton** when: | Use **CUDA** when: |
|----------------------|---------------------|
| Writing deep learning kernels (matmul, activations, attention) | You need low-level control (warp-level primitives, inline PTX) |
| You want quick iteration (Python, not C++) | Interfacing with existing CUDA libraries |
| Performance needs to match cuBLAS/cuDNN | Triton doesn't support the operation (e.g., graphics, ray tracing) |
| You're a Python developer without deep GPU expertise | You're optimizing for a specific GPU architecture (Tensor Cores) |
| Fusing multiple operations (e.g., matmul + activation + bias) | Debugging requires assembly inspection (PTX/SASS) |

**Rule of thumb:** Start with Triton for productivity. Drop to CUDA only if Triton can't express your operation or you need the last 5% performance.

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA Toolkit (11.0+)
- PyTorch (recommended for data handling)

### Install Triton

```bash
pip install triton
```

Verify installation:

```python
import triton
print(triton.__version__)
```

---

## Understanding Triton Through CUDA

Why learn CUDA if Triton is easier?

1. **Triton is built on CUDA**
   Triton compiles to PTX (CUDA assembly). Understanding CUDA helps you reason about what the compiler generates.

2. **Debugging and profiling**
   When Triton kernels are slow, you need to understand:
   - Occupancy (warps per SM)
   - Shared memory bank conflicts
   - Register spilling
   - Memory coalescing

   These are CUDA concepts.

3. **Custom optimizations**
   Some operations benefit from CUDA-specific features:
   - Warp shuffle (`__shfl_down_sync`)
   - Cooperative groups
   - Tensor Cores (WMMA / MMA instructions)

4. **Reading Triton-generated code**
   To verify the compiler did what you expected, inspect the PTX:
   ```python
   kernel.src  # View generated Triton IR
   kernel.asm  # View generated PTX assembly (requires compilation)
   ```

**Conclusion:** CUDA knowledge makes you a **better Triton programmer**, not an outdated one.

---

## Triton vs Other Frameworks

| Framework | Abstraction Level | Performance | Use Case |
|-----------|-------------------|-------------|----------|
| **CUDA** | Low (threads, warps) | Highest (manual control) | Custom kernels, library development |
| **Triton** | Medium (blocks, tiles) | High (compiler-optimized) | Deep learning ops, rapid prototyping |
| **PyTorch** | High (tensors) | Medium (unless using torch.compile) | Model training, inference |
| **CuPy** | High (NumPy-like) | Medium | Array operations, SciPy replacement |
| **Numba** | Medium (NumPy + JIT) | Medium | General Python GPU acceleration |

Triton occupies the sweet spot: **high-level enough for productivity, low-level enough for performance**.

---

## Example: Vector Addition Comparison

### CUDA (54 lines with error checking)
```cuda
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] + b[tid];
}

// Host code: allocate, copy, launch, copy back, free...
```

### Triton (12 lines total)
```python
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)

grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
add_kernel[grid](a, b, c, n, BLOCK=1024)
```

**Both achieve the same performance.** Triton is 4x less code.

---

## Resources

### Official Documentation
- [Triton Language Reference](https://triton-lang.org/main/index.html) — Complete API docs
- [GitHub Repository](https://github.com/triton-lang/triton) — Source code, examples, issues
- [OpenAI Blog Post](https://openai.com/index/triton/) — Original announcement and motivation

### Academic Paper
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) (MAPL 2019)

### Tutorials
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) — Vector add, matrix multiplication, fused softmax
- [PyTorch Blog: Triton](https://pytorch.org/blog/introducing-triton/) — Integration with `torch.compile`

### Community
- [Triton Discussions](https://github.com/triton-lang/triton/discussions) — Q&A and announcements

---

## Folder Contents

| File | Description |
|------|-------------|
| `README.md` | This guide |

_Example kernels coming soon._

---

## Summary

**Triton is a Python-based GPU programming language that lets you write high-performance kernels without low-level CUDA complexity.**

Key takeaways:
- Write **block programs** (operate on tiles), not scalar programs
- Compiler handles threads, memory optimization, vectorization
- Achieves cuBLAS/cuDNN-level performance with less code
- Built on CUDA — understanding CUDA makes you a better Triton programmer
- Ideal for deep learning ops, kernel fusion, rapid prototyping

**Next steps:** Explore the [official tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) to write your first Triton kernel.
