#pragma once
#include "model/Tool/Tensor/Tensor.h"
#include "model/CudaConfig.cuh"

__global__ void MaxPool2D_div2_kernel(float* output,float* mask,const float* input,size_t batch,size_t channels,size_t out_h,size_t out_w,size_t in_h,size_t in_w);

__global__ void MaxPool2D_mul2_kernel(float* grad_out,float* mask,const float* grad_input,size_t batch,size_t channels,size_t out_h,size_t out_w,size_t in_h,size_t in_w);