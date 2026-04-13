#include "model/Layer_activation/Layer_relu.h"

Tensor LayerRelu::relu(Tensor& x){
    return x.max();
}

Tensor LayerRelu::relu_grad(){
    return (_last_output > 0.0f);
}

void LayerRelu::build(){
    set_output_shape(_shape_input);
    print_couche_msg("Build termine.",Color::GREEN);
}

Tensor LayerRelu::forward(Tensor& input){
    _last_output = relu(input);
    return _last_output;
}

Tensor LayerRelu::backward(const Tensor& grad){
    return grad * relu_grad();
}

void LayerRelu::to_json(json& j) const {}

void LayerRelu::load_json(const json& j) {}