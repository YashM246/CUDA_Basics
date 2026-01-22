/*
 * Type Casting in C/C++
 * =====================
 *
 * What is type casting?
 *   Converting a value from one data type to another.
 *   Also called "type conversion" or "coercion".
 *
 * ===========================================
 * C-Style Cast (used in this file):
 * ===========================================
 *   Syntax: (target_type)expression
 *   Example: int x = (int)3.14;
 *
 *   This is the traditional C way. It works but provides
 *   no safety checks. C++ has better alternatives.
 *
 * ===========================================
 * C++ Cast Operators (safer alternatives):
 * ===========================================
 *
 * 1. static_cast<T>(expr)
 *    - Most common, compile-time checked
 *    - For related types (int<->float, base<->derived*)
 *    - Example: int x = static_cast<int>(3.14);
 *
 * 2. dynamic_cast<T>(expr)
 *    - Runtime-checked for polymorphic types
 *    - For safe downcasting in class hierarchies
 *    - Returns nullptr if cast fails (for pointers)
 *    - Example: Derived* d = dynamic_cast<Derived*>(basePtr);
 *
 * 3. const_cast<T>(expr)
 *    - Add or remove const/volatile qualifiers
 *    - Only use when you know the data isn't truly const
 *    - Example: int* p = const_cast<int*>(constPtr);
 *
 * 4. reinterpret_cast<T>(expr)
 *    - Bit-level reinterpretation (dangerous!)
 *    - For unrelated types, pointer<->int conversions
 *    - Example: int* p = reinterpret_cast<int*>(0x1234);
 *
 * When to use which:
 *   static_cast     -> 90% of cases (numeric, related types)
 *   dynamic_cast    -> Polymorphism, runtime safety needed
 *   const_cast      -> Interfacing with legacy const-incorrect code
 *   reinterpret_cast-> Low-level, hardware, memory manipulation
 *   C-style cast    -> Avoid in C++ (use above instead)
 *
 * CUDA Note:
 *   - cudaMalloc uses void**, often needs static_cast
 *   - Kernel arithmetic often mixes float/int (static_cast)
 *   - Memory addresses may need reinterpret_cast
 *
 * Compile: cl 01.c
 * Run:     .\01.exe
 */

#include <stdio.h>

int main(){
    // ==========================================
    // 1. Float to Int (truncation)
    // ==========================================
    // C-style: (int)f
    // C++ equivalent: static_cast<int>(f)
    float f = 69.69f;
    int i = (int)f;  // Explicit cast - truncates decimal

    printf("Float to Int:\n");
    printf("  float f = %.2f\n", f);
    printf("  (int)f  = %d  (decimal truncated, not rounded)\n\n", i);

    // ==========================================
    // 2. Int to Char (ASCII conversion)
    // ==========================================
    // C-style: (char)i
    // C++ equivalent: static_cast<char>(i)
    char c = (char)i;  // 69 -> 'E' in ASCII

    printf("Int to Char:\n");
    printf("  int i   = %d\n", i);
    printf("  (char)i = '%c'  (ASCII character for 69)\n\n", c);

    // ==========================================
    // 3. Char to Int (get ASCII value)
    // ==========================================
    char letter = 'A';
    int ascii = (int)letter;

    printf("Char to Int:\n");
    printf("  char letter = '%c'\n", letter);
    printf("  (int)letter = %d  (ASCII code)\n\n", ascii);

    // ==========================================
    // 4. Integer division vs floating-point
    // ==========================================
    int a = 5, b = 2;

    printf("Division casting:\n");
    printf("  5 / 2         = %d  (integer division)\n", a / b);
    printf("  (float)5 / 2  = %.1f  (cast before division)\n", (float)a / b);

    return 0;
}

/*
 * Warning - Data Loss:
 *
 *   Large int to small type:
 *     int big = 300;
 *     char c = (char)big;  // Overflow! char max is 127 or 255
 *
 *   Float to int with large values:
 *     float f = 1e20;
 *     int i = (int)f;  // Undefined behavior!
 *
 * Best Practice:
 *   - Use C++ casts in C++ code for clarity and safety
 *   - static_cast for most conversions
 *   - Be aware of value ranges when casting
 *   - In CUDA kernels, float<->int conversions can
 *     affect performance (use __float2int_rn() for rounding)
 */