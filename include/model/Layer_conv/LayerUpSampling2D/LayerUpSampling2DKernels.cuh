#pragma once
#include "model/Tool/Tensor/Tensor.h"
#include "model/CudaConfig.cuh"


__global__ void UpSampling2D_mul2_kernel(float* output,const float* input,size_t batch,size_t channels,size_t out_h,size_t out_w,size_t in_h,size_t in_w);

__global__ void UpSampling2D_div2_kernel(float* grad_prev,const float* grad_next,size_t batch,size_t channels,size_t out_h,size_t out_w,size_t in_h,size_t in_w);
