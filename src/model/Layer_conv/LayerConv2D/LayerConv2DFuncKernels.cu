#include "model/Layer_conv/LayerConv2D/LayerConv2DFuncKernels.cuh"

void check_is_gpu(const Tensor& tens){
    if(!tens.is_gpu()){
        Throw_Error("Tensor non GPU, utilisation de Cuda impossible");
    }
}

void gpu_im2col(Tensor& input_col, Tensor& input, size_t Kernel, size_t pad){
    check_is_gpu(input_col);
    check_is_gpu(input);

    Shape shape = input.get_shape();
    if(shape.len() != 4)
        Throw_Error("gpu_im2col, Source shape != [Batch, Channel, Height, Width]" ,shape.print());

    size_t Batch   = shape[0];
    size_t Channel = shape[1];
    size_t Height  = shape[2];
    size_t Width   = shape[3];

    dim3 threads = CudaConfig::threads_per_block_2D();
    dim3 blocks  = CudaConfig::calculs_blocks_2D(Batch * Height * Width, Channel * Kernel * Kernel);

    TensorDataGPU* datagpu_input_col = dynamic_cast<TensorDataGPU*>(input_col.get_data());
    float* Dest = datagpu_input_col->get_data_gpu().data();

    TensorDataGPU* datagpu_input = dynamic_cast<TensorDataGPU*>(input.get_data());
    float* Source = datagpu_input->get_data_gpu().data();

    im2col_kernel<<<blocks, threads>>>(Dest,Source,Batch,Height,Width,Channel,Kernel,pad,Channel * Kernel * Kernel,Height * Width);
    cuda_check_all("gpu_im2col");
}


void gpu_add_bias_conv(Tensor& output, const Tensor& bias, size_t batch, size_t nb_filters, size_t height, size_t width){
    check_is_gpu(output);
    check_is_gpu(bias);
    
    size_t total = batch * nb_filters * height * width;

    TensorDataGPU* datagpu_output = dynamic_cast<TensorDataGPU*>(output.get_data());
    float* Dest = datagpu_output->get_data_gpu().data();

    TensorDataGPU* datagpu_bias = dynamic_cast<TensorDataGPU*>(bias.get_data());
    float* Source = datagpu_bias->get_data_gpu().data();

    int blocks = CudaConfig::calculs_blocks_1D(total);
    add_bias_conv_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>( Dest, Source, batch, nb_filters, height, width);
    cuda_check_all("add_bias_conv_kernel");
}   














void gpu_col2im(Tensor& Dest,const Tensor& Source, Shape shape, size_t Kernel, size_t pad){
    check_is_gpu(Source);
    check_is_gpu(Dest);

    TensorDataGPU* datagpu_Dest = dynamic_cast<TensorDataGPU*>(Dest.get_data());
    float* dest = datagpu_Dest->get_data_gpu().data();

    const TensorDataGPU* datagpu_Source = dynamic_cast<const TensorDataGPU*>(Source.get_data());
    const float* source = datagpu_Source->get_data_gpu().data();

    if(shape.len() != 4)
        Throw_Error("gpu_col2im, Dest shape != [Batch, Channel, Height, Width]", shape.print());

    size_t Batch   = shape[0];
    size_t Channel = shape[1];
    size_t Height  = shape[2];
    size_t Width   = shape[3];

    size_t rows = Channel * Kernel * Kernel;
    size_t cols = Height * Width;

    dim3 threads = CudaConfig::threads_per_block_2D();
    dim3 blocks  = CudaConfig::calculs_blocks_2D(Batch * cols, rows);

    col2im_kernel<<<blocks, threads>>>(dest, source, Batch, Height, Width, Channel, Kernel, pad, rows, cols);
    cuda_check_all("gpu_col2im");
}

void gpu_sum_bias_conv(Tensor& grad_b, const Tensor& grad_input, size_t batch, size_t nb_filters, size_t height, size_t width){
    check_is_gpu(grad_b);
    check_is_gpu(grad_input);

    TensorDataGPU* datagpu_grad_b = dynamic_cast<TensorDataGPU*>(grad_b.get_data());
    float* Dest = datagpu_grad_b->get_data_gpu().data();

    TensorDataGPU* datagpu_grad_input = dynamic_cast<TensorDataGPU*>(grad_input.get_data());
    float* Source = datagpu_grad_input->get_data_gpu().data();


    int blocks = CudaConfig::calculs_blocks_1D(nb_filters);
    sum_bias_conv_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(Dest, Source, batch, nb_filters, height, width);
    cuda_check_all("sum_bias_conv_kernel");
}

void gpu_sum_batch(Tensor& Dest, Tensor& Source, size_t batch, size_t rows, size_t cols){
    check_is_gpu(Source);
    check_is_gpu(Dest);

    TensorDataGPU* datagpu_Dest = dynamic_cast<TensorDataGPU*>(Dest.get_data());
    float* dest = datagpu_Dest->get_data_gpu().data();

    TensorDataGPU* datagpu_Source = dynamic_cast<TensorDataGPU*>(Source.get_data());
    float* source = datagpu_Source->get_data_gpu().data();

    dim3 threads = CudaConfig::threads_per_block_2D();
    dim3 blocks = CudaConfig::calculs_blocks_2D(cols, rows);

    sum_batch_kernel<<<blocks, threads>>>(dest, source, batch, rows, cols);
    cuda_check_all("sum_batch_kernel");
}