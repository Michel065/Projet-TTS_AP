#include "model/Layer_dense/Layer_identity.h"
#include "model/Model.h"

LayerIdentity::LayerIdentity() : Layer("Identite"){}

void LayerIdentity::build(){
    _shape_output = _shape_input;
    print_couche_msg("Build termine.",Color::GREEN);
}

Tensor LayerIdentity::forward(Tensor& input){
    return input;
}


Tensor LayerIdentity::backward(Tensor& grad){
    return grad;
}

void LayerIdentity::to_json(json& j) const{}

void LayerIdentity::load_json(const json& j) {}