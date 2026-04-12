#include "model/Layer_conv/LayerFlatten.h"
#include "model/Model.h"

LayerFlatten::LayerFlatten() : Layer("Flatten") {}

void LayerFlatten::build(){
    if(_shape_input.len()!= 3){
        Throw_Error("Dimensions non valides "+_shape_input.print());
        return;
    }

    _total=1;
    for(size_t i = 0; i < (size_t)_shape_input.len(); i++){
        _total *= _shape_input[i];
    }

    set_output_shape(Shape({_total}));

    print_couche_msg("Build termine.", Color::GREEN);
}

Tensor LayerFlatten::forward(Tensor& input){
    _shape_input_batch_save = input.get_shape();

    size_t batch = _shape_input_batch_save[0];
    size_t total = 1;

    for(size_t i = 1; i < (size_t)_shape_input_batch_save.len(); i++){
        total *= _shape_input_batch_save[i];
    }

    Tensor output = input;
    output.reshape(Shape({batch, total}));
    return output;
}

Tensor LayerFlatten::backward(const Tensor& grad){
    Tensor output = grad;
    output.reshape(_shape_input_batch_save);
    return output;
}

void LayerFlatten::to_json(json& j) const{
}

void LayerFlatten::load_json(const json& j){
}