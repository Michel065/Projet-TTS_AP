#pragma once
#include "model/Tool/Tensor/Tensor.h"
#include "model/CudaConfig.cuh"


__global__ void im2col_kernel(float* Dest, const float* Source, size_t Batch, size_t Height, size_t Width, size_t Channel, size_t Kernel, size_t pad, size_t rows, size_t cols);
__global__ void add_bias_conv_kernel(float* output, const float* bias, size_t batch, size_t nb_filters, size_t height, size_t width);

__global__ void col2im_kernel(float* Dest, const float* Source, size_t Batch, size_t Height, size_t Width,  size_t Channel, size_t Kernel, size_t pad, size_t rows, size_t cols);
__global__ void sum_bias_conv_kernel(float* grad_b, const float* grad_input, size_t batch, size_t nb_filters, size_t height, size_t width);
__global__ void sum_batch_kernel(float* dest, const float* source, size_t batch, size_t rows, size_t cols);
