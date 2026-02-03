/*
 * Atomic Operations in CUDA - atomicAdd Demo
 * ============================================
 *
 * This program demonstrates WHY atomic operations are essential in CUDA
 * by showing the difference between atomic and non-atomic memory access.
 *
 * ===========================================
 * The Problem: Race Conditions
 * ===========================================
 *
 * When multiple threads try to update the SAME memory location at the
 * same time, a race condition occurs. The result becomes unpredictable.
 *
 * Consider 3 threads all trying to increment a counter from 0 to 3:
 *
 *   Expected (sequential):
 *     Thread 0: read 0, write 1
 *     Thread 1: read 1, write 2
 *     Thread 2: read 2, write 3   --> Final: 3 (correct)
 *
 *   Actual (parallel, non-atomic):
 *     Thread 0: read 0  ─┐
 *     Thread 1: read 0  ─┤  All read the SAME value!
 *     Thread 2: read 0  ─┘
 *     Thread 0: write 1
 *     Thread 1: write 1  ← Overwrites Thread 0's work
 *     Thread 2: write 1  ← Overwrites Thread 1's work
 *     Final: 1 (WRONG! Should be 3)
 *
 * ===========================================
 * The Solution: Atomic Operations
 * ===========================================
 *
 * Atomic operations guarantee that a read-modify-write sequence
 * completes as a SINGLE, INDIVISIBLE operation. No other thread
 * can interfere between the read and the write.
 *
 *   With atomicAdd (correct):
 *     Thread 0: atomicAdd → read 0, write 1 (locked, no interruption)
 *     Thread 1: atomicAdd → read 1, write 2 (waits for Thread 0)
 *     Thread 2: atomicAdd → read 2, write 3 (waits for Thread 1)
 *     Final: 3 (correct!)
 *
 * Note: The actual order of threads is non-deterministic, but the
 * final result is always correct because each operation is atomic.
 *
 * ===========================================
 * Available Atomic Functions in CUDA
 * ===========================================
 *
 *   atomicAdd(addr, val)     - *addr += val (returns old value)
 *   atomicSub(addr, val)     - *addr -= val
 *   atomicExch(addr, val)    - *addr = val (swap)
 *   atomicMin(addr, val)     - *addr = min(*addr, val)
 *   atomicMax(addr, val)     - *addr = max(*addr, val)
 *   atomicAnd(addr, val)     - *addr &= val
 *   atomicOr(addr, val)      - *addr |= val
 *   atomicXor(addr, val)     - *addr ^= val
 *   atomicCAS(addr, cmp, val)- Compare-and-swap (if *addr == cmp, set to val)
 *
 * ===========================================
 * Performance Consideration
 * ===========================================
 *
 * Atomics are slower than regular memory operations because they
 * serialize access (threads must wait their turn). Use them only
 * when necessary:
 *
 *   - Counters (counting events across threads)
 *   - Histograms (binning data)
 *   - Reductions (finding sum/min/max)
 *   - Lock-free data structures
 *
 * When many threads target the SAME address, contention is high
 * and performance drops. Strategies to reduce contention:
 *   - Use shared memory atomics first, then one global atomic
 *   - Privatization: each block has its own copy, merge at the end
 *   - Warp-level reductions before atomic operations
 *
 * Compile: nvcc 01_atomic_add.cu -o 01_atomic_add.exe
 * Run:     .\01_atomic_add.exe
 */

#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Total threads = NUM_BLOCKS * NUM_THREADS = 1,000 * 1,000 = 1,000,000
 * If every thread increments once, the expected result is 1,000,000
 */
#define NUM_THREADS 1000
#define NUM_BLOCKS 1000
#define EXPECTED_COUNT (NUM_BLOCKS * NUM_THREADS)

/*
 * ===========================================
 * Non-Atomic Increment Kernel (INCORRECT)
 * ===========================================
 *
 * This kernel demonstrates the WRONG way to update shared memory.
 *
 * The operation "counter = counter + 1" is actually 3 steps:
 *   1. READ:  Load current value from global memory into register
 *   2. MODIFY: Add 1 to the register value
 *   3. WRITE: Store the new value back to global memory
 *
 * Between steps 1 and 3, thousands of other threads may also read
 * the same old value, so most increments get "lost".
 *
 * With 1,000,000 threads, you'll typically get a result far less
 * than 1,000,000 (often just a few thousand or even less).
 */
__global__ void incrementCounterNonAtomic(int *counter) {
    /*
     * Step 1: READ - Load current value from global memory
     *
     * Multiple threads execute this at the same time.
     * They all see the SAME old value (e.g., 42).
     */
    int old = *counter;

    /*
     * Step 2: MODIFY - Compute new value in a register
     *
     * All threads that read 42 now compute 42 + 1 = 43.
     */
    int new_val = old + 1;

    /*
     * Step 3: WRITE - Store back to global memory
     *
     * All those threads write 43, not 43, 44, 45, ...
     * We lost all but one of their increments!
     */
    *counter = new_val;
}

/*
 * ===========================================
 * Atomic Increment Kernel (CORRECT)
 * ===========================================
 *
 * atomicAdd(address, value):
 *   - Reads *address, adds value, writes result back
 *   - All three steps happen as ONE indivisible operation
 *   - Returns the OLD value (before the add)
 *
 * The hardware ensures no other thread can access the same
 * memory location between the read and write. This is done
 * using special hardware instructions on the GPU's memory
 * controllers.
 */
__global__ void incrementCounterAtomic(int *counter) {
    /*
     * atomicAdd returns the old value before the addition.
     * This is useful for things like:
     *   - Getting a unique index: int myIndex = atomicAdd(counter, 1);
     *   - Building dynamic lists: int pos = atomicAdd(size, 1); list[pos] = data;
     *
     * Even though we don't use the return value here, the atomic
     * guarantee ensures every increment is counted.
     */
    atomicAdd(counter, 1);
}

int main() {
    printf("Atomic vs Non-Atomic Counter Increment\n");
    printf("Total threads: %d blocks x %d threads = %d\n",
           NUM_BLOCKS, NUM_THREADS, EXPECTED_COUNT);
    printf("Expected result: %d\n\n", EXPECTED_COUNT);

    /*
     * ===========================================
     * Host Variables
     * ===========================================
     *
     * Both counters start at 0. After 1,000,000 increments:
     *   - Non-atomic: will be far less than 1,000,000 (race condition)
     *   - Atomic: will be exactly 1,000,000
     */
    int h_counterNonAtomic = 0;
    int h_counterAtomic = 0;
    int *d_counterNonAtomic, *d_counterAtomic;

    // Allocate device memory for both counters
    cudaMalloc((void **)&d_counterNonAtomic, sizeof(int));
    cudaMalloc((void **)&d_counterAtomic, sizeof(int));

    // Copy initial values (0) to device
    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);

    /*
     * ===========================================
     * Launch Both Kernels
     * ===========================================
     *
     * Both kernels launch 1,000 blocks x 1,000 threads = 1,000,000 threads.
     * Each thread increments the counter exactly once.
     */
    incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);

    // Wait for both kernels to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);

    /*
     * ===========================================
     * Print and Analyze Results
     * ===========================================
     *
     * The non-atomic result will vary between runs due to the
     * unpredictable nature of race conditions.
     *
     * The atomic result will ALWAYS be exactly 1,000,000.
     */
    printf("========== Results ==========\n");
    printf("Non-atomic counter: %d", h_counterNonAtomic);
    if (h_counterNonAtomic != EXPECTED_COUNT) {
        printf("  (WRONG! Lost %d increments due to race conditions)",
               EXPECTED_COUNT - h_counterNonAtomic);
    }
    printf("\n");

    printf("Atomic counter:     %d", h_counterAtomic);
    if (h_counterAtomic == EXPECTED_COUNT) {
        printf("  (CORRECT!)");
    } else {
        printf("  (UNEXPECTED ERROR)");
    }
    printf("\n");

    printf("\nAccuracy - Non-atomic: %.2f%%, Atomic: %.2f%%\n",
           (h_counterNonAtomic / (double)EXPECTED_COUNT) * 100.0,
           (h_counterAtomic / (double)EXPECTED_COUNT) * 100.0);

    // Cleanup
    cudaFree(d_counterNonAtomic);
    cudaFree(d_counterAtomic);

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    return 0;
}

/*
 * ===========================================
 * Expected Output (example)
 * ===========================================
 *
 * Atomic vs Non-Atomic Counter Increment
 * Total threads: 1000 blocks x 1000 threads = 1000000
 * Expected result: 1000000
 *
 * ========== Results ==========
 * Non-atomic counter: 4721  (WRONG! Lost 995279 increments due to race conditions)
 * Atomic counter:     1000000  (CORRECT!)
 *
 * Accuracy - Non-atomic: 0.47%, Atomic: 100.00%
 *
 * ===========================================
 * Key Takeaways
 * ===========================================
 *
 * 1. Never use regular read-modify-write on shared global memory
 *    when multiple threads access the same address.
 *
 * 2. atomicAdd (and other atomic functions) guarantee correctness
 *    at the cost of some performance (serialization).
 *
 * 3. The non-atomic result varies every run because race conditions
 *    are non-deterministic - the exact thread scheduling differs.
 *
 * 4. atomicAdd returns the OLD value, which is useful for:
 *    - Assigning unique indices to threads
 *    - Building dynamic arrays/queues on the GPU
 *    - Implementing locks (atomicCAS)
 */
