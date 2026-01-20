/*
 * NULL Pointers
 * =============
 *
 * NULL is a special value indicating a pointer points to "nothing".
 * It's defined as ((void*)0) - the address zero.
 *
 * Why use NULL?
 *   - Initialize pointers that don't yet have a valid address
 *   - Indicate failure (e.g., malloc returns NULL on failure)
 *   - Mark the end of data structures (e.g., linked list terminator)
 *   - Safe default value for pointers
 *
 * Key Rules:
 *   1. Always initialize pointers to NULL if not assigning immediately
 *   2. Always check for NULL before dereferencing
 *   3. Dereferencing NULL causes a crash (segmentation fault)
 *
 * CUDA Note:
 *   cudaMalloc also returns an error code, and you should check
 *   if memory allocation succeeded before using device pointers.
 *
 * Compile: cl 04.c
 * Run:     .\04.exe
 */

#include <stdio.h>
#include <stdlib.h>  // For malloc, free, NULL

int main(){
    // Initialize pointer to NULL (good practice)
    int* ptr = NULL;
    printf("1. Initial ptr value: %p\n", (void*)ptr);

    // ALWAYS check for NULL before dereferencing
    if (ptr == NULL){
        printf("2. ptr is NULL, cannot dereference safely.\n");
    }

    // Allocate memory - malloc returns NULL if allocation fails
    ptr = (int*)malloc(sizeof(int));

    if (ptr == NULL){
        printf("3. Memory allocation failed!\n");
        return 1;  // Exit with error code
    }

    // Now ptr is valid, we can use it
    *ptr = 100;
    printf("4. Allocated memory, value: %d\n", *ptr);

    // Free memory when done
    free(ptr);

    // After free, set to NULL to avoid "dangling pointer"
    ptr = NULL;
    printf("5. After free, ptr reset to: %p\n", (void*)ptr);

    return 0;
}

/*
 * Common Mistakes:
 *
 *   int* ptr;           // Uninitialized - contains garbage address!
 *   *ptr = 10;          // CRASH - dereferencing garbage
 *
 *   int* ptr = NULL;
 *   *ptr = 10;          // CRASH - dereferencing NULL
 *
 *   free(ptr);
 *   *ptr = 10;          // CRASH - dangling pointer (use after free)
 */