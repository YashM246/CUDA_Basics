/*
 * Arrays of Pointers (Simulating 2D Arrays)
 * ==========================================
 *
 * An array of pointers can represent a 2D structure:
 *   int* matrix[2]  ->  Array containing 2 int pointers
 *                       Each pointer points to a row of integers
 *
 * Memory Layout:
 *   matrix[0] ---> [1][2][3][4]  (arr)
 *   matrix[1] ---> [5][6][7][8]  (arr2)
 *
 * This is different from a true 2D array:
 *   int matrix[2][4]  ->  All 8 integers stored contiguously
 *   int* matrix[2]    ->  Pointers stored together, data can be anywhere
 *
 * Accessing elements:
 *   matrix[i][j]    ->  Element at row i, column j
 *   *(matrix[i]+j)  ->  Same thing using pointer arithmetic
 *   *matrix[i]++    ->  Get current element, then move pointer forward
 *
 * Why use arrays of pointers?
 *   - Rows can have different lengths (jagged arrays)
 *   - Rows can be allocated/freed independently
 *   - Easy to swap rows (just swap pointers)
 *
 * CUDA Note:
 *   This pattern is common when managing multiple data buffers.
 *   However, for GPU matrices, contiguous 2D arrays are often preferred
 *   for better memory coalescing.
 *
 * Compile: cl 06.c
 * Run:     .\06.exe
 */

#include <stdio.h>

int main(){
    // Two separate arrays (rows)
    int arr[] = {1, 2, 3, 4};
    int arr2[] = {5, 6, 7, 8};

    // Pointers to each row
    int* ptr = arr;
    int* ptr2 = arr2;

    // Array of pointers - each element points to a row
    int* matrix[] = {ptr, ptr2};

    printf("Matrix using array of pointers:\n");

    // Traverse using pointer increment
    // WARNING: This modifies the pointers in matrix[]
    for(int i = 0; i < 2; i++){
        printf("Row %d: ", i);
        for(int j = 0; j < 4; j++){
            // *matrix[i]   = value at current position
            // matrix[i]++  = move pointer to next element
            printf("%d ", *matrix[i]++);
        }
        printf("\n");
    }

    // Alternative: Access without modifying pointers
    printf("\nAccessing with matrix[i][j] syntax:\n");
    // Reset pointers (they were modified above)
    matrix[0] = arr;
    matrix[1] = arr2;

    for(int i = 0; i < 2; i++){
        printf("Row %d: ", i);
        for(int j = 0; j < 4; j++){
            printf("%d ", matrix[i][j]);  // Same as *(matrix[i] + j)
        }
        printf("\n");
    }

    return 0;
}

/*
 * Pointer Type Breakdown:
 *
 *   matrix      ->  int**  (pointer to pointer to int)
 *   matrix[i]   ->  int*   (pointer to int, i.e., a row)
 *   matrix[i][j]->  int    (the actual value)
 *
 * This is why int** is used for dynamic 2D arrays!
 */