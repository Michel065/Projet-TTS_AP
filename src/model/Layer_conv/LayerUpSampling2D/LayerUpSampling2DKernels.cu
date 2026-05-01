#include "model/Layer_conv/LayerUpSampling2D/LayerUpSampling2DKernels.cuh"

__global__ void UpSampling2D_mul2_kernel(float* output, const float* input,size_t batch, size_t channels,size_t out_h, size_t out_w,size_t in_h, size_t in_w){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t yc = blockIdx.y * blockDim.y + threadIdx.y;
    size_t b = blockIdx.z;

    if(x >= out_w || yc >= channels * out_h || b >= batch) return;

    size_t c = yc / out_h;
    size_t y = yc % out_h;

    size_t in_y = y / 2;
    size_t in_x = x / 2;

    size_t base_in  = b * (channels * in_h * in_w) + c * (in_h * in_w);
    size_t base_out = b * (channels * out_h * out_w) + c * (out_h * out_w);

    size_t out_idx = base_out + y * out_w + x;
    size_t in_idx  = base_in + in_y * in_w + in_x;

    output[out_idx] = input[in_idx];
}


__global__ void UpSampling2D_div2_kernel(float* grad_prev,const float* grad_next,size_t batch,size_t channels,size_t out_h,size_t out_w,size_t in_h,size_t in_w){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t yc = blockIdx.y * blockDim.y + threadIdx.y;
    size_t b = blockIdx.z;

    if(x >= in_w || yc >= channels * in_h || b >= batch) return;

    size_t c = yc / in_h;
    size_t y = yc % in_h;

    size_t out_y = y * 2;
    size_t out_x = x * 2;

    size_t base_in  = b * (channels * in_h * in_w) + c * (in_h * in_w);
    size_t base_out = b * (channels * out_h * out_w) + c * (out_h * out_w);

    float val =
        grad_next[base_out + out_y * out_w + out_x] + // c'est pour pas avoir de for je sais pas si c'est pas plus simple pour cuda.
        grad_next[base_out + (out_y + 1) * out_w + out_x] +
        grad_next[base_out + out_y * out_w + (out_x + 1)] +
        grad_next[base_out + (out_y + 1) * out_w + (out_x + 1)];

    grad_prev[base_in + y * in_w + x] = val;
}