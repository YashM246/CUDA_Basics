/*
 * Advanced CUDA Streams - Priorities, Events, and Callbacks
 * ==========================================================
 *
 * This program demonstrates advanced stream features:
 *   1. Stream Priorities - Give important work higher priority
 *   2. Events - Synchronize between different streams
 *   3. Callbacks - Execute host functions when stream operations complete
 *
 * ===========================================
 * Recap: What are Streams?
 * ===========================================
 *
 * Streams are sequences of operations that execute IN ORDER on the GPU.
 * Operations in DIFFERENT streams can overlap (run concurrently).
 *
 *   Default stream (stream 0):
 *     - Implicitly synchronizes with all other streams
 *     - Safe but limits concurrency
 *
 *   User-created streams:
 *     - Can run concurrently with each other
 *     - Need explicit synchronization when dependencies exist
 *
 * ===========================================
 * Stream Priorities
 * ===========================================
 *
 * CUDA allows assigning priority levels to streams. When the GPU has
 * limited resources and must choose which work to schedule, higher
 * priority streams get preference.
 *
 * Priority values:
 *   - LOWER numbers = HIGHER priority (counterintuitive!)
 *   - Range varies by GPU (typically -1 to 0, or -2 to 0)
 *   - Use cudaDeviceGetStreamPriorityRange() to query your GPU
 *
 * Example priority range on most GPUs:
 *   leastPriority = 0      (lowest priority, default)
 *   greatestPriority = -1  (highest priority)
 *
 * When to use priorities:
 *   - Real-time processing: Give display/audio streams high priority
 *   - Latency-sensitive work: Prioritize user-facing operations
 *   - Background work: Give batch processing low priority
 *
 * IMPORTANT: Priorities only matter when there's resource contention.
 * If the GPU has enough resources for all work, priorities have no effect.
 *
 * ===========================================
 * Events - Cross-Stream Synchronization
 * ===========================================
 *
 * Events are markers that can be inserted into streams. They enable:
 *   1. Timing GPU operations
 *   2. Synchronizing between streams
 *   3. Checking if operations have completed
 *
 * Key functions:
 *   cudaEventCreate(&event)       - Create an event object
 *   cudaEventRecord(event, stream) - Insert event marker into stream
 *   cudaStreamWaitEvent(stream, event, 0) - Make stream wait for event
 *   cudaEventSynchronize(event)   - Block CPU until event completes
 *   cudaEventElapsedTime(&ms, start, end) - Measure time between events
 *
 * Event flow in this program:
 *
 *   Stream1: [MemcpyH2D] → [Kernel1] → [EVENT recorded here]
 *                                            ↓
 *   Stream2: ─────────────────────[WAIT]→ [Kernel2] → [Callback] → [MemcpyD2H]
 *
 * Stream2 waits at the WAIT point until Stream1 records the EVENT.
 * This ensures Kernel1 completes before Kernel2 starts (they share data).
 *
 * ===========================================
 * Stream Callbacks
 * ===========================================
 *
 * Callbacks let you execute a HOST function when a stream reaches
 * a certain point. The callback runs on a CPU thread managed by CUDA.
 *
 * Use cases:
 *   - Logging/debugging: Print when operations complete
 *   - Resource cleanup: Free memory after kernel finishes
 *   - Triggering next steps: Start new work based on results
 *   - Progress reporting: Update UI with completion status
 *
 * IMPORTANT callback rules:
 *   1. Callbacks run on a separate CPU thread (not the main thread!)
 *   2. Callbacks should be SHORT - don't do heavy computation
 *   3. DON'T call CUDA APIs from callbacks (can cause deadlocks)
 *   4. The stream is blocked until the callback returns
 *
 * Callback signature:
 *   void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void *userData)
 *
 *   - stream: The stream that triggered the callback
 *   - status: cudaSuccess if all prior operations succeeded
 *   - userData: Custom data you passed to cudaStreamAddCallback
 *
 * Note: cudaStreamAddCallback is deprecated in CUDA 10+.
 * The modern alternative is cudaLaunchHostFunc(), but the old API
 * still works and is clearer for learning purposes.
 *
 * ===========================================
 * Program Flow
 * ===========================================
 *
 *   1. Create two streams with different priorities
 *   2. Stream1 (low priority): Copy data to GPU, run Kernel1 (multiply by 2)
 *   3. Record event after Kernel1
 *   4. Stream2 (high priority): Wait for event, then run Kernel2 (add 1)
 *   5. Callback fires when Kernel2 completes
 *   6. Copy results back and verify
 *
 * Expected result: Each element i becomes (i * 2) + 1
 *
 * Compile: nvcc 02_stream_advanced.cu -o 02_stream_advanced.exe
 * Run:     .\02_stream_advanced.exe
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * ===========================================
 * CUDA Error Checking Macro
 * ===========================================
 *
 * Wraps every CUDA call to check for errors immediately.
 * See 01_stream_basics.cu for detailed explanation of how this works.
 */
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char *const func, const char *const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, static_cast<unsigned int>(err),
                cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

/*
 * ===========================================
 * Kernel 1: Multiply by 2
 * ===========================================
 *
 * Simple kernel that doubles each element.
 * This will run in stream1 (low priority).
 */
__global__ void kernel1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

/*
 * ===========================================
 * Kernel 2: Add 1
 * ===========================================
 *
 * Simple kernel that adds 1 to each element.
 * This will run in stream2 (high priority).
 *
 * IMPORTANT: This kernel DEPENDS on Kernel1's output!
 * We use an event to ensure Kernel1 completes first.
 */
__global__ void kernel2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

/*
 * ===========================================
 * Stream Callback Function
 * ===========================================
 *
 * This function is called by CUDA when the stream reaches this point.
 *
 * ===========================================
 * What is CUDART_CB?
 * ===========================================
 *
 * CUDART_CB is a MACRO that specifies the "calling convention" for the callback.
 *
 * What is a calling convention?
 * -----------------------------
 * When one function calls another, they need to agree on:
 *   1. How to pass arguments (registers? stack? what order?)
 *   2. Who cleans up the stack after the call (caller or callee?)
 *   3. Which registers must be preserved
 *
 * Different calling conventions exist:
 *   - __cdecl     : C default. Caller cleans stack. Args right-to-left.
 *   - __stdcall   : Windows API standard. Callee cleans stack.
 *   - __fastcall  : First args in registers (faster).
 *
 * Why does CUDA need a specific calling convention?
 * -------------------------------------------------
 * CUDA's internal thread (that calls your callback) expects a specific convention.
 * If your function uses a DIFFERENT convention, bad things happen:
 *   - Stack corruption
 *   - Wrong argument values
 *   - Crashes
 *
 * What CUDART_CB expands to:
 * --------------------------
 *   On Windows:  #define CUDART_CB __stdcall
 *   On Linux:    #define CUDART_CB  (empty - Linux uses __cdecl by default)
 *
 * So on Windows, your callback becomes:
 *   void __stdcall myStreamCallback(...)
 *
 * And on Linux, it's just:
 *   void myStreamCallback(...)
 *
 * Why the difference?
 * -------------------
 * Windows historically uses __stdcall for callbacks (like in Win32 API).
 * Linux/Unix systems standardized on __cdecl, so no special annotation needed.
 *
 * What happens WITHOUT CUDART_CB on Windows?
 * ------------------------------------------
 * Without it, your function defaults to __cdecl, but CUDA expects __stdcall.
 *
 *   Your function (__cdecl):     Expects CALLER to clean stack
 *   CUDA's caller (__stdcall):   Expects CALLEE to clean stack
 *
 *   Result: Stack gets cleaned TWICE or NOT AT ALL → crash/corruption!
 *
 * Simple analogy:
 * ---------------
 * Imagine a relay race where runners pass batons:
 *   - Calling convention = the rules for how to pass the baton
 *   - If one runner uses different rules, they drop the baton (crash!)
 *   - CUDART_CB ensures everyone uses the same passing rules
 *
 * TL;DR:
 * ------
 * CUDART_CB = "Use the calling convention CUDA expects for callbacks"
 * Always use it for CUDA callbacks. It's required on Windows, harmless on Linux.
 *
 * ===========================================
 * Callback Signature Breakdown
 * ===========================================
 *
 * void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void *userData)
 *       ↑              ↑           ↑                   ↑                    ↑
 *       │              │           │                   │                    │
 *       │              │           │                   │                    └─ Your custom data
 *       │              │           │                   │                       (passed to cudaStreamAddCallback)
 *       │              │           │                   │
 *       │              │           │                   └─ Error status of prior operations
 *       │              │           │                      cudaSuccess = all good
 *       │              │           │
 *       │              │           └─ Which stream triggered this callback
 *       │              │
 *       │              └─ Your function name (can be anything)
 *       │
 *       └─ Calling convention macro (REQUIRED!)
 *
 * ===========================================
 * Practical Usage Examples
 * ===========================================
 *
 * In this example, we just print a message. In real applications:
 *   - Signal completion to main thread (condition variable, atomic flag)
 *   - Log timing information
 *   - Trigger next stage of processing
 *
 * WARNINGS:
 *   1. This runs on a DIFFERENT thread than main()!
 *   2. Keep callbacks SHORT - stream is blocked while callback runs
 *   3. DON'T call CUDA APIs from here (deadlock risk)
 *   4. Use thread-safe methods if modifying shared data
 */
void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    /*
     * Check if previous operations succeeded.
     * If any CUDA operation before this callback failed,
     * status will contain the error code.
     */
    if (status != cudaSuccess) {
        printf("Callback: ERROR - Previous operation failed: %s\n",
               cudaGetErrorString(status));
        return;
    }

    /*
     * userData can be used to pass custom data to the callback.
     * Example: pass a struct with timing info, file handles, etc.
     *
     * In this example, we pass NULL, but you could do:
     *   int *counter = (int *)userData;
     *   (*counter)++;
     */
    printf("Stream callback: Kernel2 completed successfully!\n");
}

int main(void) {
    printf("Advanced CUDA Streams Demo\n");
    printf("==========================\n\n");

    /*
     * ===========================================
     * Configuration
     * ===========================================
     */
    const int N = 1000000;          // 1 million elements
    size_t size = N * sizeof(float);
    float *h_data, *d_data;
    cudaStream_t stream1, stream2;
    cudaEvent_t event;

    /*
     * ===========================================
     * Query Stream Priority Range
     * ===========================================
     *
     * cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority)
     *
     * This function queries what priority values YOUR GPU supports.
     * Different GPUs may have different ranges!
     *
     * Parameters:
     *   leastPriority    - Output: Lowest priority value (numerically highest, e.g., 0)
     *   greatestPriority - Output: Highest priority value (numerically lowest, e.g., -1)
     *
     * IMPORTANT: Lower number = Higher priority!
     *   Priority -2 is HIGHER priority than Priority 0
     *
     * Why this design?
     *   - Matches Unix nice values (lower = higher priority)
     *   - 0 is the default, negative numbers are "better than default"
     *
     * Typical ranges:
     *   - Most GPUs: 0 (lowest) to -1 (highest) → 2 priority levels
     *   - Some GPUs: 0 to -2 → 3 priority levels
     *   - Some GPUs: 0 to 0 → No priority support!
     *
     * If leastPriority == greatestPriority, the GPU doesn't support priorities.
     */
    int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

    printf("Stream Priority Range:\n");
    printf("  Least priority (lowest):    %d\n", leastPriority);
    printf("  Greatest priority (highest): %d\n", greatestPriority);

    if (leastPriority == greatestPriority) {
        printf("  WARNING: This GPU does not support stream priorities!\n");
    } else {
        printf("  Available levels: %d\n", leastPriority - greatestPriority + 1);
    }
    printf("\n");

    /*
     * ===========================================
     * Create Streams with Priorities
     * ===========================================
     *
     * cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority)
     *
     * Parameters:
     *   pStream  - Output: The created stream handle
     *   flags    - Stream behavior flags (see below)
     *   priority - Priority level (from cudaDeviceGetStreamPriorityRange)
     *
     * Flags options:
     *   cudaStreamDefault     - Default stream behavior
     *                           Implicitly synchronizes with stream 0 (legacy behavior)
     *
     *   cudaStreamNonBlocking - Does NOT synchronize with stream 0
     *                           Allows maximum concurrency
     *                           RECOMMENDED for most use cases
     *
     * The difference matters when mixing with default stream:
     *
     *   With cudaStreamDefault:
     *     Stream 0:  [Kernel A]──────────────[Kernel C]
     *     Stream 1:  ──────────[Kernel B]────────────
     *                          ↑ Waits    ↑ Stream 0 waits
     *
     *   With cudaStreamNonBlocking:
     *     Stream 0:  [Kernel A]────[Kernel C]
     *     Stream 1:  [Kernel B]──────────────
     *                ↑ Can run in parallel!
     *
     * In this example:
     *   - stream1: Low priority (background work)
     *   - stream2: High priority (gets resources first when contention occurs)
     */
    printf("Creating streams with priorities...\n");

    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(
        &stream1,
        cudaStreamNonBlocking,  // Don't synchronize with default stream
        leastPriority           // Low priority (numerically higher value)
    ));
    printf("  Stream1 created with priority %d (low)\n", leastPriority);

    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(
        &stream2,
        cudaStreamNonBlocking,  // Don't synchronize with default stream
        greatestPriority        // High priority (numerically lower value)
    ));
    printf("  Stream2 created with priority %d (high)\n", greatestPriority);
    printf("\n");

    /*
     * ===========================================
     * Create Event for Cross-Stream Synchronization
     * ===========================================
     *
     * cudaEventCreate(cudaEvent_t *event)
     *
     * Creates an event that can be:
     *   - Recorded in a stream (marks a point in time)
     *   - Waited on by another stream
     *   - Used for timing measurements
     *
     * Event flags (cudaEventCreateWithFlags):
     *   cudaEventDefault        - Default behavior
     *   cudaEventBlockingSync   - CPU thread blocks efficiently (less CPU usage)
     *   cudaEventDisableTiming  - Faster, but can't use for timing
     *
     * For this demo, we use default flags.
     */
    CHECK_CUDA_ERROR(cudaEventCreate(&event));
    printf("Event created for cross-stream synchronization\n\n");

    /*
     * ===========================================
     * Allocate Memory
     * ===========================================
     *
     * Using pinned (page-locked) memory for async transfers.
     * See 01_stream_basics.cu for detailed explanation.
     */
    printf("Allocating memory...\n");
    CHECK_CUDA_ERROR(cudaMallocHost(&h_data, size));  // Pinned host memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));      // Device memory
    printf("  Host (pinned): %zu bytes\n", size);
    printf("  Device: %zu bytes\n\n", size);

    /*
     * ===========================================
     * Initialize Data
     * ===========================================
     *
     * Fill array with values 0, 1, 2, ..., N-1
     * After processing: each element i becomes (i * 2) + 1
     */
    printf("Initializing data (h_data[i] = i)...\n");
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }
    printf("\n");

    /*
     * ===========================================
     * Stream1: Copy to GPU and Run Kernel1
     * ===========================================
     *
     * Operations in stream1:
     *   1. Async copy: Host → Device
     *   2. Kernel1: Multiply each element by 2
     *
     * These execute IN ORDER within stream1.
     */
    printf("Stream1: Copying data to GPU and running Kernel1...\n");

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, h_data, size,
                                      cudaMemcpyHostToDevice, stream1));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, N);
    CHECK_CUDA_ERROR(cudaGetLastError());  // Check kernel launch errors

    /*
     * ===========================================
     * Record Event in Stream1
     * ===========================================
     *
     * cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
     *
     * "Records" the event in the stream. This means:
     *   - The event is placed AFTER all previous operations in the stream
     *   - When those operations complete, the event is "triggered"
     *   - Other streams waiting on this event can then proceed
     *
     * Timeline:
     *   Stream1: [MemcpyH2D] → [Kernel1] → [EVENT]
     *                                         ↓
     *                              Event is "recorded" here
     *                              (triggered when Kernel1 finishes)
     *
     * Note: cudaEventRecord is asynchronous - it returns immediately.
     * The event is actually recorded when the GPU reaches that point.
     */
    CHECK_CUDA_ERROR(cudaEventRecord(event, stream1));
    printf("  Event recorded after Kernel1\n");

    /*
     * ===========================================
     * Stream2: Wait for Event, Then Run Kernel2
     * ===========================================
     *
     * cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
     *
     * Makes `stream` wait until `event` is triggered (recorded).
     *
     * Parameters:
     *   stream - The stream that should wait
     *   event  - The event to wait for
     *   flags  - Must be 0 (reserved for future use)
     *
     * This is ESSENTIAL when streams have dependencies:
     *   - Kernel2 reads data modified by Kernel1
     *   - Without synchronization: race condition (undefined behavior)
     *   - With event: Kernel2 guaranteed to see Kernel1's results
     *
     * Timeline:
     *   Stream1: [MemcpyH2D] → [Kernel1] → [EVENT recorded]
     *                                            ↓
     *   Stream2: ─────────────────────[WAIT]→ [Kernel2] → ...
     *                                   ↑
     *                          Stream2 is blocked here
     *                          until event is triggered
     *
     * Note: This is GPU-side waiting, NOT CPU blocking!
     * The CPU continues executing after cudaStreamWaitEvent returns.
     */
    printf("Stream2: Waiting for event, then running Kernel2...\n");

    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream2, event, 0));

    kernel2<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data, N);
    CHECK_CUDA_ERROR(cudaGetLastError());  // Check kernel launch errors

    /*
     * ===========================================
     * Add Callback to Stream2
     * ===========================================
     *
     * cudaStreamAddCallback(cudaStream_t stream,
     *                       cudaStreamCallback_t callback,
     *                       void *userData,
     *                       unsigned int flags)
     *
     * Parameters:
     *   stream   - Stream to add callback to
     *   callback - Function pointer to the callback
     *   userData - Custom data to pass to callback (can be NULL)
     *   flags    - Must be 0 (reserved)
     *
     * When the GPU reaches this point in the stream, it:
     *   1. Pauses the stream
     *   2. Calls your callback function on a CPU thread
     *   3. Resumes the stream after callback returns
     *
     * Timeline:
     *   Stream2: [WAIT] → [Kernel2] → [CALLBACK] → [MemcpyD2H]
     *                                     ↓
     *                           CPU: myStreamCallback() runs
     *                                     ↓
     *                           Stream resumes after return
     *
     * DEPRECATION NOTE:
     *   cudaStreamAddCallback is deprecated in CUDA 10+.
     *   The modern alternative is cudaLaunchHostFunc():
     *
     *     void myFunc(void *userData) { ... }
     *     cudaLaunchHostFunc(stream, myFunc, userData);
     *
     *   cudaLaunchHostFunc has a simpler signature (no stream/status params)
     *   and integrates better with CUDA graphs. However, the old API
     *   still works and is kept here for clarity.
     */
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, myStreamCallback, NULL, 0));
    printf("  Callback registered\n");

    /*
     * ===========================================
     * Copy Results Back (in Stream2)
     * ===========================================
     *
     * This happens AFTER the callback (stream operations are in order).
     */
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_data, size,
                                      cudaMemcpyDeviceToHost, stream2));
    printf("  Async copy back to host queued\n\n");

    /*
     * ===========================================
     * Synchronize Streams
     * ===========================================
     *
     * cudaStreamSynchronize(cudaStream_t stream)
     *
     * Blocks the CPU until ALL operations in `stream` complete.
     *
     * Alternatives:
     *   cudaDeviceSynchronize()  - Wait for ALL streams
     *   cudaEventSynchronize(e)  - Wait for specific event
     *   cudaStreamQuery(stream)  - Non-blocking check (returns immediately)
     *
     * We synchronize both streams to ensure all work is done.
     */
    printf("Synchronizing streams...\n");
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    printf("  Stream1 synchronized\n");
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));
    printf("  Stream2 synchronized\n\n");

    /*
     * ===========================================
     * Verify Results
     * ===========================================
     *
     * Each element started as i.
     * Kernel1: i * 2
     * Kernel2: (i * 2) + 1
     *
     * Expected final value: 2*i + 1
     */
    printf("Verifying results...\n");
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = (static_cast<float>(i) * 2.0f) + 1.0f;
        if (fabs(h_data[i] - expected) > 1e-5f) {
            if (errors < 5) {  // Only print first 5 errors
                fprintf(stderr, "  MISMATCH at element %d: expected %.2f, got %.2f\n",
                        i, expected, h_data[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("  All %d elements correct!\n", N);
        printf("\nTest PASSED\n");
    } else {
        printf("  %d errors found!\n", errors);
        printf("\nTest FAILED\n");
    }

    /*
     * ===========================================
     * Cleanup
     * ===========================================
     *
     * Always destroy resources in reverse order of creation.
     * Events should be destroyed before streams that use them.
     */
    printf("\nCleaning up...\n");
    CHECK_CUDA_ERROR(cudaEventDestroy(event));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFreeHost(h_data));
    printf("Done!\n");

    return 0;
}

/*
 * ===========================================
 * Expected Output
 * ===========================================
 *
 * Advanced CUDA Streams Demo
 * ==========================
 *
 * Stream Priority Range:
 *   Least priority (lowest):    0
 *   Greatest priority (highest): -1
 *   Available levels: 2
 *
 * Creating streams with priorities...
 *   Stream1 created with priority 0 (low)
 *   Stream2 created with priority -1 (high)
 *
 * Event created for cross-stream synchronization
 *
 * Allocating memory...
 *   Host (pinned): 4000000 bytes
 *   Device: 4000000 bytes
 *
 * Initializing data (h_data[i] = i)...
 *
 * Stream1: Copying data to GPU and running Kernel1...
 *   Event recorded after Kernel1
 * Stream2: Waiting for event, then running Kernel2...
 *   Callback registered
 *   Async copy back to host queued
 *
 * Synchronizing streams...
 *   Stream1 synchronized
 * Stream callback: Kernel2 completed successfully!
 *   Stream2 synchronized
 *
 * Verifying results...
 *   All 1000000 elements correct!
 *
 * Test PASSED
 *
 * Cleaning up...
 * Done!
 *
 * ===========================================
 * Key Takeaways
 * ===========================================
 *
 * 1. STREAM PRIORITIES:
 *    - Use cudaDeviceGetStreamPriorityRange() to query supported range
 *    - Lower number = Higher priority (counterintuitive!)
 *    - Only matters when GPU resources are contended
 *
 * 2. EVENTS FOR SYNCHRONIZATION:
 *    - cudaEventRecord(event, stream) - Mark a point in a stream
 *    - cudaStreamWaitEvent(stream, event, 0) - Make stream wait
 *    - Essential for dependencies between streams
 *
 * 3. STREAM CALLBACKS:
 *    - Run host code when GPU reaches a point in stream
 *    - Keep callbacks SHORT (stream is blocked)
 *    - DON'T call CUDA APIs from callbacks
 *    - Consider cudaLaunchHostFunc() for new code
 *
 * 4. NON-BLOCKING STREAMS:
 *    - cudaStreamNonBlocking allows maximum concurrency
 *    - Streams don't implicitly sync with default stream
 *
 * ===========================================
 * Common Patterns Using These Features
 * ===========================================
 *
 * Pattern 1: Pipeline with dependencies
 *   Stream1: [Stage1] → [Event]
 *   Stream2:     [Wait] → [Stage2] → [Event]
 *   Stream3:                  [Wait] → [Stage3]
 *
 * Pattern 2: Priority-based scheduling
 *   High priority stream: User-facing real-time work
 *   Low priority stream: Background batch processing
 *
 * Pattern 3: Completion notification
 *   Use callbacks to signal main thread when GPU work is done
 *   Main thread can then process results without polling
 */
