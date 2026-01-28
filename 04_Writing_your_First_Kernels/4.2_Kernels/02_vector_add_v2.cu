#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 10000000
#define BLOCK_SIZE_ID 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8
// 16 * 8 * 8 = 1024

// CPU Vector Addition
void vector_add_cpu(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
}

// CUDA Kernel for 1D Vector Addition
__global__ void vector_add_gpu(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] + b[i];
    }
}

// CUDA Kernel for 3D Vector Addition
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    // 3 add, 3 multiples, 3 stores

    if (i<nx && j<ny && k<nz){
        int idx = i + (j*nx) + (k*nx*ny);
        if(idx < (nx*ny*nz)){
            c[idx] = a[idx] + b[idx];
        }
    }
    // you get the point
}