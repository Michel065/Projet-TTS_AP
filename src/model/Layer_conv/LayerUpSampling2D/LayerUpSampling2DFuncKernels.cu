#include "model/Layer_conv/LayerUpSampling2D/LayerUpSampling2DFuncKernels.cuh"

void gpu_UpSampling2D_mul2(Tensor& output,Tensor& input, size_t _taille_batch, Shape shape_input, Shape shape_output){
    float* d_output = check_and_get_if_is_gpu(output);
    float* d_input = check_and_get_if_is_gpu(input);

    size_t channels = shape_output[0];
    size_t out_h = shape_output[1];
    size_t out_w = shape_output[2];
    size_t in_h = shape_input[1];
    size_t in_w = shape_input[2];

    dim3 threads = CudaConfig::threads_per_block_2D();
    dim3 blocks = CudaConfig::calculs_blocks_2D(out_w,channels * out_h, _taille_batch);    
    UpSampling2D_mul2_kernel<<<blocks, threads>>>(d_output,d_input,_taille_batch,channels,out_h,out_w,in_h,in_w);
    cuda_check_all("maxpool2d_forward_kernel");
}

void gpu_UpSampling2D_div2(Tensor& grad_out, Tensor& grad_in, size_t _taille_batch, Shape shape_input, Shape shape_output){
    float* d_grad_out = check_and_get_if_is_gpu(grad_out);
    float* d_grad_in = check_and_get_if_is_gpu(grad_in);

    size_t channels = shape_output[0];
    size_t out_h = shape_output[1];
    size_t out_w = shape_output[2];
    size_t in_h = shape_input[1];
    size_t in_w = shape_input[2];
    
    dim3 threads = CudaConfig::threads_per_block_2D();
    dim3 blocks = CudaConfig::calculs_blocks_2D(in_w,channels * in_h, _taille_batch);  
    UpSampling2D_div2_kernel<<<blocks, threads>>>(d_grad_out,d_grad_in,_taille_batch,channels,out_h,out_w,in_h,in_w);
    cuda_check_all("maxpool2d_backward_kernel");
}