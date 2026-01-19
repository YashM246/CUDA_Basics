/*
 * Introduction to Pointers
 * ========================
 *
 * What is a pointer?
 *   A pointer is a variable that stores the memory address of 
 *   another variable.
 *   Instead of holding a value like 10 or 'A', it holds an 
 *   address like 0x7fff5fbff8ac.
 *
 * Why use pointers?
 *   - Pass large data to functions without copying
 *   - Modify variables inside functions
 *   - Dynamic memory allocation
 *   - Essential for CUDA (managing host/device memory)
 *
 * Key Operators:
 *   &  (Address-of operator)  - Gets the memory address of a variable
 *   *  (Dereference operator) - Gets the value at an address
 *
 * Pointer Declaration:
 *   int* ptr;    // ptr is a pointer to an integer
 *   char* cptr;  // cptr is a pointer to a char
 *
 * Compile: cl 01.c /Fe:01.exe
 * Run:     .\01.exe
 */

#include <stdio.h>

int main(){
    int x = 10;

    // Declare a pointer and store the address of x
    // int* means "pointer to int"
    // &x means "address of x"
    int* ptr = &x;

    // Printing the pointer (the address it holds)
    printf("Address of x: %p\n", ptr);

    // Dereferencing: use * to get the value at that address
    printf("Value of x:   %d\n", *ptr);

    // Demonstration: modifying x through the pointer
    *ptr = 20;  // This changes x to 20
    printf("\nAfter *ptr = 20:\n");
    printf("Value of x:   %d\n", x);

    return 0;
}