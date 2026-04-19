#include "model/Layer_conv/LayerMaxPool2D/LayerMaxPoolFuncKernels.cuh"

void gpu_MaxPool2D_div2(Tensor& output, Tensor& _mask, Tensor& input, size_t _taille_batch, Shape shape_input, Shape shape_output){
    check_is_gpu(output);
    check_is_gpu(_mask);
    check_is_gpu(input);

    TensorDataGPU* data_output = dynamic_cast<TensorDataGPU*>(output.get_data());
    float* d_output = data_output->get_data_gpu().data();

    TensorDataGPU* data_mask = dynamic_cast<TensorDataGPU*>(_mask.get_data());
    float* d_mask = data_mask->get_data_gpu().data();
    
    TensorDataGPU* data_input = dynamic_cast<TensorDataGPU*>(input.get_data());
    float* d_input = data_input->get_data_gpu().data();

    size_t channels = shape_output[0];//j'ai mis les shapes a la place de recup dans in et out sans raison, c'est plus jolie.
    size_t out_h = shape_output[1];
    size_t out_w = shape_output[2];
    size_t in_h = shape_input[1];
    size_t in_w = shape_input[2];

    dim3 threads = CudaConfig::threads_per_block_2D();
    dim3 blocks = CudaConfig::calculs_blocks_2D(out_w,channels * out_h, _taille_batch);    
    MaxPool2D_div2_kernel<<<blocks, threads>>>(d_output,d_mask,d_input,_taille_batch,channels,out_h,out_w,in_h,in_w);
    cuda_check_all("maxpool2d_forward_kernel");
}

void gpu_MaxPool2D_mul2(Tensor& grad_out, Tensor& _mask, Tensor& grad_in, size_t _taille_batch, Shape shape_input, Shape shape_output){
    check_is_gpu(grad_out);
    check_is_gpu(_mask);
    check_is_gpu(grad_in);

    TensorDataGPU* data_grad_out = dynamic_cast<TensorDataGPU*>(grad_out.get_data());
    float* d_grad_out = data_grad_out->get_data_gpu().data();

    TensorDataGPU* data_mask = dynamic_cast<TensorDataGPU*>(_mask.get_data());
    float* d_mask = data_mask->get_data_gpu().data();
    
    TensorDataGPU* data_grad_in = dynamic_cast<TensorDataGPU*>(grad_in.get_data());
    float* d_grad_in = data_grad_in->get_data_gpu().data();

    size_t channels = shape_output[0];//plus clair
    size_t out_h = shape_output[1];
    size_t out_w = shape_output[2];
    size_t in_h = shape_input[1];
    size_t in_w = shape_input[2];
    
    dim3 threads = CudaConfig::threads_per_block_2D();
    dim3 blocks = CudaConfig::calculs_blocks_2D(out_w,channels * out_h, _taille_batch);  
    MaxPool2D_mul2_kernel<<<blocks, threads>>>(d_grad_out,d_mask,d_grad_in,_taille_batch,channels,out_h,out_w,in_h,in_w);
    cuda_check_all("maxpool2d_forward_kernel");
}