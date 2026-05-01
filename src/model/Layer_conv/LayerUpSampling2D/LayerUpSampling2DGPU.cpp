#include "model/Layer_conv/LayerUpSampling2D/LayerUpSampling2DGPU.h"

void LayerUpSampling2DGPU::forward(Tensor& output, Tensor& input, size_t _taille_batch, Shape shape_input, Shape shape_output){
    gpu_UpSampling2D_mul2(output,input,_taille_batch,shape_input,shape_output);
}







void LayerUpSampling2DGPU::backward(Tensor& grad_output,Tensor& grad_input, size_t _taille_batch, Shape shape_input, Shape shape_output){  
    gpu_UpSampling2D_div2(grad_output,grad_input,_taille_batch,shape_input,shape_output);
}