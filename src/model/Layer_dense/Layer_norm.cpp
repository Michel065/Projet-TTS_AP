#include "model/Layer_dense/Layer_norm.h"

LayerNormalisation::LayerNormalisation(float min, float max,TypeNormalisation type_norm): Layer("Normalisation"), _min(min), _max(max),_type_norm(type_norm){
    if(_max == _min)
        Throw_Error("max ne peut pas etre egal a min (LayerNormalisation)");
}


Tensor LayerNormalisation::calc_defaut(const Tensor& input){
    return (input - _min) / (_max - _min);
}

Tensor LayerNormalisation::calc_defaut_grad(const Tensor& grad){
    return grad / (_max - _min);
}

Tensor LayerNormalisation::calc_alternative(const Tensor& input){
    return 2.0f * (input - _min) / (_max - _min) - 1.0f;
}

Tensor LayerNormalisation::calc_alternative_grad(const Tensor& grad){
    return grad * (2.0f / (_max - _min));
}

void LayerNormalisation::build(){
    set_output_shape(_shape_input);
    print_couche_msg("Build termine.", Color::GREEN);
}

Tensor LayerNormalisation::forward(const Tensor& input){
    switch(_type_norm){
        case TypeNormalisation::DEFAULT:
            return calc_defaut(input);

        case TypeNormalisation::ALTERNATIVE:
            return calc_alternative(input);
    }

    Throw_Error("TypeNormalisation inconnu");
    return input;
}

Tensor LayerNormalisation::backward(const Tensor& grad){
    switch(_type_norm){
        case TypeNormalisation::DEFAULT:
            return calc_defaut_grad(grad);

        case TypeNormalisation::ALTERNATIVE:
            return calc_alternative_grad(grad);
    }

    Throw_Error("TypeNormalisation inconnu");
    return grad;
}