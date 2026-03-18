#include "model/Layer_activation/Layer_sigmoid.h"

Tensor LayerSigmoid::sigmoid(const Tensor& x){
    return 1.0 / (1 + (-x).exp());
}

Tensor LayerSigmoid::sigmoid_grad(){
    return _last_output * (1.0 - _last_output);
}

void LayerSigmoid::build(){
    set_output_shape(_shape_input);
    print_couche_msg("Build termine.",Color::GREEN);
}

Tensor LayerSigmoid::forward(const Tensor& input){
    _last_output = sigmoid(input);
    return _last_output;
}

Tensor LayerSigmoid::backward(const Tensor& grad){
    return grad * sigmoid_grad();
}
