#pragma once
#include "model/Layer.h"
#include "model/Model.h"
#include "model/Layer_conv/LayerConv2D/LayerConv2DFuncKernels.cuh"

namespace LayerConv2DGPU {
    void forward(Tensor& output, Tensor& input, 
        const Tensor& _W, const Tensor& _b,
        size_t nb_filters, size_t kernel,size_t pad,
        Shape shape_input);

    void backward(Tensor& grad_W, Tensor& grad_b,Tensor& grad_output,
        Tensor& grad_input, Tensor& last_input, Tensor& _W,
        size_t nb_filters, size_t kernel, size_t pad,
        Shape shape_input);
}