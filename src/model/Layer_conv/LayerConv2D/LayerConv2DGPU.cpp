#include "model/Layer_conv/LayerConv2D/LayerConv2DGPU.h"

void LayerConv2DGPU::forward(Tensor& output, Tensor& input,
    const Tensor& _W, const Tensor& _b,
    size_t nb_filters, size_t kernel, size_t pad,
    Shape shape_input){

    size_t batch   = input.get_shape()[0];
    size_t channel = shape_input[0];
    size_t height  = shape_input[1];
    size_t width   = shape_input[2];

    Shape output_ori = output.get_shape();

    Tensor input_col(DeviceType::GPU, Shape({batch, channel * kernel * kernel, height * width}), false);

    gpu_im2col(input_col, input, kernel, pad);
    output = std::move(_W.prod_mat(input_col));

    gpu_add_bias_conv(output, _b, batch, nb_filters, height, width);

    //on rebascule sur le format d'origine {batch, nb_filters, height, width}
    output.reshape(output_ori);
}







void LayerConv2DGPU::backward(Tensor& grad_W, Tensor& grad_b, Tensor& grad_output,Tensor& grad_input, Tensor& last_input, Tensor& _W,size_t nb_filters, size_t kernel, size_t pad, Shape shape_input){  
    size_t batch   = last_input.get_shape()[0];
    size_t channel = shape_input[0];
    size_t height  = shape_input[1];
    size_t width   = shape_input[2];

    size_t rows = channel * kernel * kernel;
    size_t cols = height * width;

    grad_input.reshape(Shape({batch, nb_filters, cols}));

    Tensor last_input_col(DeviceType::GPU, Shape({batch, rows, cols}), false);
    gpu_im2col(last_input_col, last_input, kernel, pad);

    Tensor last_input_col_T = std::move(last_input_col);
    last_input_col_T.transpose(true);

    gpu_sum_bias_conv(grad_b, grad_input, batch, nb_filters, height, width);// grad b

    Tensor grad_W_batch = grad_input.prod_mat(last_input_col_T);
    gpu_sum_batch(grad_W, grad_W_batch, batch, nb_filters, rows);
    grad_W /= (float)batch;// grad W 

    Tensor W_T = _W;
    W_T.transpose(false);

    Tensor grad_col = W_T.prod_mat(grad_input);
    gpu_col2im(grad_output, grad_col, last_input.get_shape(), kernel, pad); // grad suivant
}