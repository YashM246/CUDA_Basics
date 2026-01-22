/*
 * Preprocessor Macros and Conditional Compilation
 * ================================================
 *
 * What is the preprocessor?
 *   A text-processing step that runs BEFORE compilation.
 *   It handles all directives starting with #
 *
 * ===========================================
 * Macro Types:
 * ===========================================
 *
 * 1. Object-like macros (constants):
 *    #define PI 3.14159
 *    Usage: float area = PI * r * r;
 *
 * 2. Function-like macros:
 *    #define SQUARE(x) ((x) * (x))
 *    Usage: int result = SQUARE(5);  // expands to ((5) * (5))
 *
 *    WARNING: Always wrap parameters in parentheses!
 *    #define BAD(x) x * x
 *    BAD(1+2) expands to 1+2 * 1+2 = 1+2+2 = 5  (not 9!)
 *
 * ===========================================
 * Conditional Compilation Directives:
 * ===========================================
 *
 *   #if        - If expression is non-zero (integers only)
 *   #ifdef     - If macro is defined
 *   #ifndef    - If macro is NOT defined
 *   #elif      - Else if
 *   #else      - Else
 *   #endif     - End of conditional block
 *
 *   #define    - Define a macro
 *   #undef     - Undefine a macro
 *
 * ===========================================
 * Common Use Cases:
 * ===========================================
 *
 *   1. Header guards (prevent double inclusion):
 *      #ifndef MY_HEADER_H
 *      #define MY_HEADER_H
 *      // header content
 *      #endif
 *
 *   2. Debug code:
 *      #ifdef DEBUG
 *      printf("Debug: x = %d\n", x);
 *      #endif
 *
 *   3. Platform-specific code:
 *      #ifdef _WIN32
 *      // Windows code
 *      #else
 *      // Linux/Mac code
 *      #endif
 *
 * CUDA Note:
 *   CUDA uses macros extensively:
 *   - __CUDA_ARCH__ : defined when compiling device code
 *   - __global__, __device__, __host__ are macro-like keywords
 *   - Often used for block/grid size configuration:
 *     #define BLOCK_SIZE 256
 *
 * Compile: cl 01.c
 * Run:     .\01.exe
 */

#include <stdio.h>

// ===========================================
// Object-like macros (constants)
// ===========================================
#define PI 3.14159

// ===========================================
// Function-like macro
// ===========================================
// Note: Extra parentheses prevent operator precedence bugs
#define AREA(r) (PI * (r) * (r))

// ===========================================
// Conditional macro: define only if not already defined
// ===========================================
#ifndef RADIUS
#define RADIUS 7
#endif

// ===========================================
// Conditional compilation with #if / #elif / #else
// Note: Can only use integer constant expressions
// ===========================================
#define SIZE_MODE 2  // 1=small, 2=medium, 3=large

#if SIZE_MODE == 1
    #define ARRAY_SIZE 10
#elif SIZE_MODE == 2
    #define ARRAY_SIZE 100
#else
    #define ARRAY_SIZE 1000
#endif

int main(){
    // Using object-like macro
    printf("PI = %.5f\n\n", PI);

    // Using function-like macro
    printf("Circle calculations:\n");
    printf("  Radius: %d\n", RADIUS);
    printf("  Area:   %.2f\n\n", AREA(RADIUS));

    // Demonstrating AREA macro with expression
    // This is why we need ((r) * (r)) not (r * r)
    printf("  AREA(2+1) = %.2f  (should be ~28.27)\n\n", AREA(2+1));

    // Using conditionally defined macro
    printf("Conditional compilation:\n");
    printf("  SIZE_MODE:  %d\n", SIZE_MODE);
    printf("  ARRAY_SIZE: %d\n", ARRAY_SIZE);

    return 0;
}

/*
 * Macros vs const vs inline:
 *
 *   #define PI 3.14159      // Macro: text replacement, no type
 *   const float PI = 3.14;  // Const: typed, has memory address
 *   inline float square(x)  // Inline: typed, debuggable
 *
 * Macros:
 *   + No function call overhead
 *   + Works with any type
 *   - No type checking
 *   - Can cause unexpected behavior
 *   - Hard to debug (no symbol in debugger)
 *
 * Best Practice:
 *   - Use const for typed constants in C++
 *   - Use inline functions instead of function macros in C++
 *   - Use macros for conditional compilation
 *   - Use macros for constants in C (traditional style)
 */