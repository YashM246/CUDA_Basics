#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 1024      // Number of rows in A and C
#define K 512       // Number of columns in A and rows in B
#define N 2048      // Number of columns in B and C
#define BLOCK_SIZE 32   

// Example 3x2 @ 2x4 = 3x4

// A = [[1, 2],
//      [3, 4],
//      [5, 6]]

// B = [[7, 8, 9, 10],
//      [11, 12, 13, 14]]

// C = A @ B = [[29, 32, 35, 38],
//              [65, 72, 79, 86],
//              [101, 112, 123, 134]]

// CPU Matrix Multiplication
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            float sum = 0.0f;
            for(int l=0; l<k; l++){
                // Matrix will be flattened into a single array
                sum += A[i*k+l] + B[l*n+j];
            }
            C[i*n+j] = sum;
        }
    }
}

// CUDA Kernel for Matrix Multiplication
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row<m && col<n){
        float sum = 0.0f;
        for (int l=0; l<k; l++){
            sum += A[row*k+l] * B[l*n+col];
        }
        C[row*n+col] = sum;
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols){
    for(int i=0; i<rows*cols; i++){
        mat[i] = (float)rand()/RAND_MAX;
    }
}


//Function to count time
double get_time(){

}

// Main function to benchmark across 20 runs after warmup