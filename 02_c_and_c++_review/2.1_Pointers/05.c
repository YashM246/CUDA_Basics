/*
 * Pointer Arithmetic and Arrays
 * ==============================
 *
 * Arrays and pointers are closely related in C:
 *   - Array name (arr) is a pointer to the first element
 *   - arr[i] is equivalent to *(arr + i)
 *
 * Pointer Arithmetic:
 *   ptr + 1  doesn't add 1 byte, it adds sizeof(*ptr) bytes
 *   For int* (4 bytes): ptr + 1 moves 4 bytes forward
 *   For char* (1 byte): ptr + 1 moves 1 byte forward
 *
 * Operations:
 *   ptr++      Move to next element
 *   ptr--      Move to previous element
 *   ptr + n    Move n elements forward
 *   ptr - n    Move n elements backward
 *   ptr2 - ptr1  Number of elements between two pointers
 *
 * Memory Layout:
 *   arr:  [12] [24] [36] [48] [60]
 *   addr:  +0   +4   +8  +12  +16  (bytes from start)
 *
 * CUDA Note:
 *   GPU memory is also contiguous. Understanding pointer arithmetic
 *   is essential for navigating arrays in CUDA kernels.
 *
 * Compile: cl 05.c
 * Run:     .\05.exe
 */

#include <stdio.h>

int main(){
    int arr[] = {12, 24, 36, 48, 60};

    // Array name decays to pointer to first element
    // These two are equivalent:
    int* ptr = arr;       // arr is &arr[0]
    // int* ptr = &arr[0];

    printf("Array base address: %p\n", arr);
    printf("First element: %d\n\n", *ptr);

    // Traverse array using pointer arithmetic
    printf("Traversing array with pointer arithmetic:\n");
    printf("Value\tAddress\t\t\tOffset\n");
    printf("-----\t-------\t\t\t------\n");

    for(int i = 0; i < 5; i++){
        printf("%d\t%p\t+%d bytes\n", *ptr, (void*)ptr, (int)((char*)ptr - (char*)arr));
        ptr++;  // Move to next element (adds sizeof(int) = 4 bytes)
    }

    // Alternative: array indexing vs pointer arithmetic
    printf("\nEquivalent ways to access arr[2]:\n");
    ptr = arr;  // Reset pointer to start
    printf("arr[2]      = %d\n", arr[2]);
    printf("*(arr + 2)  = %d\n", *(arr + 2));
    printf("*(ptr + 2)  = %d\n", *(ptr + 2));
    printf("ptr[2]      = %d\n", ptr[2]);  // Yes, this works too!

    return 0;
}

/*
 * Why pointer arithmetic matters:
 *
 * 1. Efficient array traversal (no index multiplication)
 * 2. Working with memory buffers
 * 3. Implementing data structures
 * 4. CUDA: calculating thread-specific memory offsets
 *
 * Example in CUDA:
 *   int idx = threadIdx.x + blockIdx.x * blockDim.x;
 *   data[idx] = ...;  // Each thread accesses different memory
 */