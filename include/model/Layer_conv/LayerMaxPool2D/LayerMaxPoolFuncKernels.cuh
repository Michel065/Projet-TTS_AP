#pragma once
#include "model/Layer_conv/LayerMaxPool2D/LayerMaxPoolKernels.cuh"

void gpu_MaxPool2D_div2(Tensor& output, Tensor& _mask, Tensor& input, size_t _taille_batch, Shape shape_input, Shape shape_output);

void gpu_MaxPool2D_mul2(Tensor& output, Tensor& _mask, Tensor& input, size_t _taille_batch, Shape shape_input, Shape shape_output);