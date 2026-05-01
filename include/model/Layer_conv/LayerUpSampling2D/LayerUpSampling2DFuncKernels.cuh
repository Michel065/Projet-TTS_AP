#pragma once
#include "model/Layer_conv/LayerUpSampling2D/LayerUpSampling2DKernels.cuh"

void gpu_UpSampling2D_mul2(Tensor& output, Tensor& input, size_t _taille_batch, Shape shape_input, Shape shape_output);

void gpu_UpSampling2D_div2(Tensor& output, Tensor& input, size_t _taille_batch, Shape shape_input, Shape shape_output);