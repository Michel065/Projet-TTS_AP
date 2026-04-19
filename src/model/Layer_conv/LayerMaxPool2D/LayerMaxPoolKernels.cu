#include "model/Layer_conv/LayerMaxPool2D/LayerMaxPoolKernels.cuh"

__global__ void MaxPool2D_div2_kernel(float* output,float* mask,const float* input,size_t batch,size_t channels,size_t out_h,size_t out_w,size_t in_h,size_t in_w){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z;

    if(x >= out_w || y >= out_h || z >= batch * channels) return;

    size_t b = z / channels;
    size_t c = z % channels;

    size_t in_y = y * 2;
    size_t in_x = x * 2;

    size_t depart_id = b * (channels * in_h * in_w) + c * (in_h * in_w);

    float max_val = input[depart_id + in_y * in_w + in_x];
    size_t m_y = in_y;
    size_t m_x = in_x;

    for(size_t dy = 0; dy < 2; dy++){
        for(size_t dx = 0; dx < 2; dx++){
            size_t yy = in_y + dy;
            size_t xx = in_x + dx;
            float val = input[depart_id + yy * in_w + xx];
            if(val > max_val){
                max_val = val;
                m_y = yy;
                m_x = xx;
            }
        }
    }

    size_t out_idx = b * (channels * out_h * out_w) + c * (out_h * out_w) + y * out_w + x;
    output[out_idx] = max_val;

    size_t mask_idx = depart_id + m_y * in_w + m_x;
    mask[mask_idx] = 1.0f;
}


__global__ void MaxPool2D_mul2_kernel(float* grad_out,float* mask,const float* grad_input,size_t batch,size_t channels,size_t out_h,size_t out_w,size_t in_h,size_t in_w){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z;

    if(x >= in_w || y >= in_h || z >= batch * channels) return;

    size_t b = z / channels;
    size_t c = z % channels;

    size_t out_y = y / 2;
    size_t out_x = x / 2;

    size_t base_in  = b*(channels*in_h*in_w)+c*(in_h*in_w);
    size_t base_out = b*(channels*out_h*out_w)+c*(out_h*out_w);

    size_t mask_idx = base_in + y * in_w + x;

    if(mask[mask_idx] == 1.0f){
        size_t out_idx = base_in + y * in_w + x;
        size_t in_idx  = base_out + out_y * out_w + out_x;
        grad_out[out_idx] = grad_input[in_idx];
    }
}