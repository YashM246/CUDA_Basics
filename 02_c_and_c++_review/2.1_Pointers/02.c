/*
 * Pointer to Pointer (Multi-level Pointers)
 * ==========================================
 *
 * A pointer can point to another pointer, creating multiple levels of indirection.
 *
 * Levels:
 *   int*    ptr1  - Pointer to int (1 level)
 *   int**   ptr2  - Pointer to pointer to int (2 levels)
 *   int***  ptr3  - Pointer to pointer to pointer to int (3 levels)
 *
 * Dereferencing:
 *   *ptr1   - Gets the int value
 *   *ptr2   - Gets ptr1 (an int*)
 *   **ptr2  - Gets the int value
 *   ***ptr3 - Gets the int value (3 dereferences)
 *
 * Memory Layout:
 *   ptr3 ---> ptr2 ---> ptr1 ---> value (42)
 *
 * Why use multi-level pointers?
 *   - Modify a pointer inside a function
 *   - 2D dynamic arrays (int** for matrix)
 *   - Linked data structures (e.g., linked lists)
 *
 * Compile: cl 02.c
 * Run:     .\02.exe
 */

#include <stdio.h>

int main(){
    int value = 42;

    // ptr1 stores address of value
    int* ptr1 = &value;

    // ptr2 stores address of ptr1
    int** ptr2 = &ptr1;

    // ptr3 stores address of ptr2
    int*** ptr3 = &ptr2;

    // Visualize the chain
    printf("value   = %d\n", value);
    printf("*ptr1   = %d\n", *ptr1);       // 1 dereference
    printf("**ptr2  = %d\n", **ptr2);      // 2 dereferences
    printf("***ptr3 = %d\n", ***ptr3);     // 3 dereferences

    // All print 42 because they all eventually point to 'value'

    return 0;
}