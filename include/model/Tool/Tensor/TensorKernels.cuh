#pragma once

#include "model/CudaConfig.cuh"


// les ops
__global__ void add_kernel(float* a, const float* b, size_t n);
__global__ void sub_kernel(float* a, const float* b, size_t n);
__global__ void mul_kernel(float* a, const float* b, size_t n);
__global__ void div_kernel(float* a, const float* b, size_t n);

__global__ void add_kernel_scalar(float* a, const float scalar, size_t n);
__global__ void sub_kernel_scalar(float* a, const float scalar, size_t n);
__global__ void mul_kernel_scalar(float* a, const float scalar, size_t n);
__global__ void div_kernel_scalar(float* a, const float scalar, size_t n);


__global__ void exp_kernel(float* a,size_t n);
__global__ void pow_kernel(float* a, float val, size_t n);
__global__ void max_kernel(float* a, float val, size_t n);
__global__ void round_kernel(float* a, float factor, size_t n);

__global__ void clip_kernel(float* a, float b_min, float b_max, size_t n);
__global__ void log_kernel(float* a, size_t n);
__global__ void sup_kernel(float* a, float scalar, size_t n);
__global__ void transpose_kernel(float* dest, const float* source, int rows, int cols, int batch = 1);

__global__ void matmul_kernel(float* dest,const float* source_a,const float* source_b, int rows, int trans, int cols);
__global__ void matmul_kernel_shared(float* dest, const float* source_a, const float* source_b, int rows, int trans, int cols);
__global__ void shuffle_axis0_kernel(float* dest, const float* src, const int* indices, int stride, int total);
__global__ void extraction_section_axe_0_kernel(float* dest,const float* src,int debut,int stride,int total);

//methode specifique
__global__ void sum_per_row_kernel(float* dest, const float* src, int rows, int cols);
__global__ void max_per_row_kernel(float* dest, const float* src, int rows, int cols);
__global__ void sum_axis0_kernel(float* dest, const float* src, int rows, int cols);




//methode boradcast dim 0
__global__ void add_broadcast_axis0_kernel(float* dest, const float* src, int total, int stride);
__global__ void sub_broadcast_axis0_kernel(float* dest, const float* src, int total, int stride);
__global__ void mul_broadcast_axis0_kernel(float* dest, const float* src, int total, int stride);
__global__ void div_broadcast_axis0_kernel(float* dest, const float* src, int total, int stride);



// matmul version broadcast
__global__ void broadcast_matmul_kernel_shared(float* dest,const float* source_a,const float* source_b,int batch,int rows,int trans,int cols,bool batch_on_a);
__global__ void broadcast_all_matmul_kernel_shared(float* dest,const float* source_a,const float* source_b,int batch,int rows,int trans,int cols);

