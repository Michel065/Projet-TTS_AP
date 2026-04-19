#include "model/Layer_conv/LayerMaxPool2D/LayerMaxPoolGPU.h"

//ce fichier sert en soit a rien mais niveau decoupe du code c'est mieux

void LayerMaxPoolGPU::forward(Tensor& output, Tensor& _mask, Tensor& input, size_t _taille_batch, Shape shape_input, Shape shape_output){
    gpu_MaxPool2D_div2(output,_mask,input,_taille_batch,shape_input,shape_output);
}


void LayerMaxPoolGPU::backward(Tensor& grad_out, Tensor& _mask, Tensor& grad_in, size_t _taille_batch, Shape shape_input, Shape shape_output){
    gpu_MaxPool2D_mul2(grad_out,_mask,grad_in,_taille_batch,shape_input,shape_output);
}
