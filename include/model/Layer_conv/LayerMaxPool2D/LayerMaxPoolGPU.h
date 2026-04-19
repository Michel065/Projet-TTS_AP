#pragma once
#include "model/Layer.h"
#include "model/Layer_conv/LayerMaxPool2D/LayerMaxPoolFuncKernels.cuh"

namespace LayerMaxPoolGPU {
    void forward(Tensor& output, Tensor& _mask, Tensor& input, size_t _taille_batch, Shape shape_input, Shape shape_output);

    void backward(Tensor& grad_out, Tensor& _mask, Tensor& grad_in, size_t _taille_batch, Shape shape_input, Shape shape_output);
}