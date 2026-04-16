#include "model/Layer_dense/Layer_norm.h"
#include "model/Model.h"

LayerNormalisation::LayerNormalisation(): Layer("Normalisation"){}

LayerNormalisation::LayerNormalisation(xt::xarray<float> min, xt::xarray<float> max,TypeNormalisation type_norm): Layer("Normalisation"),_type_norm(type_norm){
    _device = DeviceType::CPU;

    _vmin = min;
    _vmax = max;
    if(xt::all(xt::equal(_vmax, _vmin)))
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
    _min = Tensor(_device,_vmin).reshape(Shape({1,_vmin.size()}));
    _max = Tensor(_device,_vmax).reshape(Shape({1,_vmin.size()}));
    print_couche_msg("Build termine.", Color::GREEN);
}

void LayerNormalisation::get_from_model(){
    if(_model == nullptr)
        return;
    _device = _model->get_device();
}

Tensor LayerNormalisation::forward(Tensor& input){
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

void LayerNormalisation::to_json(json& j) const{
    j["min_liste"] = _min;
    j["max_liste"] = _max;
    j["type_normalisation"] = _type_norm;
}

void LayerNormalisation::load_json(const json& j) {
    j.at("min_liste").get_to(_min);
    j.at("max_liste").get_to(_max);
    j.at("type_normalisation").get_to(_type_norm);
}