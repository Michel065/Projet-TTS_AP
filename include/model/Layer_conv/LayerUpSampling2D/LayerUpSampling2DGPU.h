#pragma once
#include "model/Layer.h"
#include "model/Model.h"
#include "model/Layer_conv/LayerUpSampling2D/LayerUpSampling2DFuncKernels.cuh"

namespace LayerUpSampling2DGPU {
    void forward(Tensor& output, Tensor& input, size_t _taille_batch, Shape shape_input, Shape shape_output);

    void backward(Tensor& grad_output,Tensor& grad_input, size_t _taille_batch, Shape shape_input, Shape shape_output);
}