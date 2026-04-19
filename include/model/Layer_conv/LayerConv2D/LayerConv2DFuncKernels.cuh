#pragma once
#include "model/Layer_conv/LayerConv2D/LayerConv2DKernels.cuh"


void gpu_im2col(Tensor& input_col, Tensor& input, size_t Kernel, size_t pad);
void gpu_add_bias_conv(Tensor& output, const Tensor& bias, size_t batch, size_t nb_filters, size_t height, size_t width);

void gpu_col2im(Tensor& Dest,const Tensor& Source, Shape shape, size_t Kernel, size_t pad);
void gpu_sum_bias_conv(Tensor& grad_b, const Tensor& grad_input, size_t batch, size_t nb_filters, size_t height, size_t width);
void gpu_sum_batch(Tensor& Dest, Tensor& Source, size_t batch, size_t rows, size_t cols);