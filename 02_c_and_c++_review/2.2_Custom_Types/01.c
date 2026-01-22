/*
 * size_t - The Size Type
 * =======================
 *
 * What is size_t?
 *   A typedef for an unsigned integer type used to represent sizes.
 *   It's guaranteed to be big enough to hold the size of the largest
 *   possible object on the system.
 *
 * Definition (simplified):
 *   typedef unsigned long long size_t;  // On 64-bit systems (8 bytes)
 *   typedef unsigned int size_t;        // On 32-bit systems (4 bytes)
 *
 * Where is size_t used?
 *   - sizeof() operator returns size_t
 *   - Array indexing and loop counters for arrays
 *   - malloc(), calloc() take size_t as parameter
 *   - strlen() returns size_t
 *
 * Why use size_t instead of int?
 *   - Can't be negative (sizes are never negative)
 *   - Large enough for any object size
 *   - Portable across 32-bit and 64-bit systems
 *   - Makes intent clear: "this is a size"
 *
 * Printf format specifier:
 *   %zu  ->  z = size_t modifier, u = unsigned
 *
 * CUDA Note:
 *   CUDA uses size_t for memory allocation sizes:
 *   cudaMalloc(&ptr, size_t numBytes);
 *
 * VS Code Tip:
 *   Ctrl + Click or F12 on any type/function to see its definition.
 *
 * Compile: cl 01.c
 * Run:     .\01.exe
 */

#include <stdio.h>
#include <stdlib.h>  // size_t is defined here (also in stddef.h)

int main(){
    int arr[] = {12, 24, 36, 48, 60};

    // sizeof returns size_t, so we store it in size_t
    size_t numElements = sizeof(arr) / sizeof(arr[0]);
    size_t totalBytes = sizeof(arr);

    printf("Array info:\n");
    printf("  Number of elements: %zu\n", numElements);      // 5
    printf("  Total size (bytes): %zu\n", totalBytes);       // 20 (5 * 4)

    printf("\nType sizes on this system:\n");
    printf("  sizeof(int):    %zu bytes\n", sizeof(int));    // 4
    printf("  sizeof(size_t): %zu bytes\n", sizeof(size_t)); // 8 (on 64-bit)
    printf("  sizeof(void*):  %zu bytes\n", sizeof(void*));  // 8 (on 64-bit)

    // Using size_t for loop counter (best practice for array iteration)
    printf("\nArray elements: ");
    for(size_t i = 0; i < numElements; i++){
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}

/*
 * Common Mistake:
 *
 *   int size = sizeof(arr);  // Works, but not ideal
 *
 *   On 64-bit systems, sizeof can return values larger than
 *   INT_MAX for very large arrays. Using int could overflow.
 *   Always use size_t for sizes.
 */