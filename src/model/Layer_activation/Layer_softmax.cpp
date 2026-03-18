#include "model/Layer_activation/Layer_softmax.h"

Tensor LayerSoftMax::softmax(const Tensor& x){
    Tensor shifted = x - x.max_per_row();
    Tensor exp_x = shifted.exp();
    Tensor sum_exp = exp_x.sum_per_row();
    return exp_x / sum_exp;
}

void LayerSoftMax::build(){
    set_output_shape(_shape_input);
    print_couche_msg("Build termine.",Color::GREEN);
}

Tensor LayerSoftMax::forward(const Tensor& input){
    _last_output = softmax(input);
    return _last_output;
}

Tensor LayerSoftMax::backward(const Tensor& grad){
    return grad;
}