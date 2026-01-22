/*
 * Structs and typedef - Custom Data Types
 * ========================================
 *
 * What is a struct?
 *   A struct groups related variables together into a single unit.
 *   Each variable inside is called a "member" or "field".
 *
 * Basic syntax:
 *   struct Point {
 *       float x;
 *       float y;
 *   };
 *   struct Point p;  // Must use "struct" keyword
 *
 * With typedef:
 *   typedef struct {
 *       float x;
 *       float y;
 *   } Point;
 *   Point p;  // No "struct" keyword needed - cleaner!
 *
 * Memory Layout:
 *   Point:  [  x (4 bytes)  ][  y (4 bytes)  ]  = 8 bytes total
 *
 * Accessing members:
 *   p.x       Direct access with dot operator
 *   ptr->x    Through pointer (same as (*ptr).x)
 *
 * CUDA Note:
 *   Structs are commonly used for:
 *   - Passing multiple parameters to kernels
 *   - Representing vectors (float3, float4 are built-in)
 *   - Organizing data for GPU processing
 *
 * Compile: cl 02.c
 * Run:     .\02.exe
 */

#include <stdio.h>

// Define a custom type using typedef + struct
typedef struct {
    float x;
    float y;
} Point;

int main(){
    // Method 1: Initialize with values in order
    Point p1 = {1.1f, 2.5f};

    // Method 2: Designated initializers (C99+)
    Point p2 = {.x = 3.0f, .y = 4.0f};

    // Method 3: Assign members individually
    Point p3;
    p3.x = 5.0f;
    p3.y = 6.0f;

    printf("Point sizes and values:\n");
    printf("  sizeof(Point): %zu bytes\n", sizeof(Point));
    printf("  sizeof(float): %zu bytes\n", sizeof(float));
    printf("\n");

    printf("  p1 = (%.1f, %.1f)\n", p1.x, p1.y);
    printf("  p2 = (%.1f, %.1f)\n", p2.x, p2.y);
    printf("  p3 = (%.1f, %.1f)\n", p3.x, p3.y);

    // Accessing through pointer
    Point* ptr = &p1;
    printf("\nAccessing p1 through pointer:\n");
    printf("  ptr->x = %.1f\n", ptr->x);   // Arrow operator
    printf("  (*ptr).y = %.1f\n", (*ptr).y);  // Dereference + dot

    return 0;
}

/*
 * Why use structs?
 *
 *   1. Organization - Group related data together
 *   2. Clarity - Code becomes self-documenting
 *   3. Pass by value - Copy entire struct to functions
 *   4. Pass by pointer - Efficient for large structs
 *
 * CUDA example:
 *   typedef struct {
 *       float* data;
 *       int width;
 *       int height;
 *   } Matrix;
 *
 *   __global__ void processMatrix(Matrix m) { ... }
 */