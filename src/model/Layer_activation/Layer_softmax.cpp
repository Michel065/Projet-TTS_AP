#include "model/Layer_activation/Layer_softmax.h"

Tensor LayerSoftMax::softmax(const Tensor& x){
    if(x.get_shape()[1] == 1){
        return x; //cas ou pas d'autre val alors pas de soft max possible
    }
    Tensor shifted = x - x.max_per_row();
    Tensor exp_x = shifted.exp();
    float eps = 1e-6f;
    Tensor sum_exp = exp_x.sum_per_row()+eps;
    return exp_x / sum_exp;
}

void LayerSoftMax::build(){
    set_output_shape(_shape_input);
    print_couche_msg("Build termine.",Color::GREEN);
}

Tensor LayerSoftMax::forward(Tensor& input){
    _last_output = softmax(input);
    return _last_output;
}

Tensor LayerSoftMax::backward(const Tensor& grad){
    return grad;
}

void LayerSoftMax::to_json(json& j) const {}

void LayerSoftMax::load_json(const json& j) {}