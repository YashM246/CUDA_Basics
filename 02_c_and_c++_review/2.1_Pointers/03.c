/*
 * Void Pointers (Generic Pointers)
 * =================================
 *
 * A void pointer (void*) is a special pointer that can hold 
 * the address of any data type. It's called a "generic" pointer.
 *
 * Key Properties:
 *   - Can point to any data type (int, float, char, struct, etc.)
 *   - Cannot be dereferenced directly (compiler doesn't know the 
 *     size)
 *   - Must be cast to a specific type before dereferencing
 *
 * Syntax:
 *   void* vptr;           // Declare a void pointer
 *   vptr = &num;          // Assign address of any type
 *   *(int*)vptr           // Cast to int*, then dereference
 *
 * Why use void pointers?
 *   - Generic functions that work with any data type
 *   - Memory allocation (malloc returns void*)
 *   - Callback functions and function pointers
 *   - CUDA: cudaMalloc uses void** for generic memory allocation
 *
 * Compile: cl 03.c
 * Run:     .\03.exe
 */

#include <stdio.h>

int main(){
    int num = 10;
    float fnum = 3.14f;

    // Declare a void pointer - can hold address of any type
    void* vptr;

    // Point to an integer
    vptr = &num;

    // To get the value, we must:
    //   1. Cast void* to int*:  (int*)vptr
    //   2. Dereference it:      *(int*)vptr
    printf("Integer: %d\n", *(int*)vptr);

    // Same pointer can now point to a float
    vptr = &fnum;

    // Cast to float* before dereferencing
    printf("Float: %.2f\n", *(float*)vptr);

    // WARNING: Wrong cast gives garbage/undefined behavior
    // printf("Wrong: %d\n", *(int*)vptr);  // Don't do this!

    return 0;
}

/*
 * Fun fact: malloc() returns a void pointer
 *
 *   void* malloc(size_t size);
 *
 *   int* arr = (int*)malloc(10 * sizeof(int));
 *
 * This allows malloc to allocate memory for any data type.
 * Similarly, CUDA's cudaMalloc uses void** to allocate GPU memory.
 */