#include "model/Layer_conv/LayerUnflatten/LayerUnflatten.h"
#include "model/Model.h"

LayerUnflatten::LayerUnflatten() : Layer("UnFlatten") {}

LayerUnflatten::LayerUnflatten(Shape new_shape) : Layer("UnFlatten") {
    _shape_out = new_shape;
}

void LayerUnflatten::get_from_model(){
    /*
    Je considere qu'il n'y a qu'un flatten par model;
    */
    if(_model == nullptr)
        return;
        
    if(_shape_out.len() != 0){
        return;
    }

    const Layer* flatten = _model->find_layer("Flatten");
    _shape_out = flatten->get_input_shape();
}

void LayerUnflatten::build(){
    if(_shape_input.len()!= 1){
        Throw_Error("Dimensions non valides "+_shape_input.print());
        return;
    }
    
    if(_shape_out.len() != 3){
        Throw_Error("Shape de sortie invalide "+_shape_out.print());
        return;
    }

    set_output_shape(_shape_out);
    print_couche_msg("Build termine.", Color::GREEN);
}

Tensor LayerUnflatten::forward(Tensor& input){
    _shape_input_batch_save = input.get_shape();
    size_t batch = _shape_input_batch_save[0];
    input.reshape(Shape({batch, _shape_output[0], _shape_output[1], _shape_output[2]}));
    return input;
}

Tensor LayerUnflatten::backward(Tensor& grad){
    grad.reshape(_shape_input_batch_save);
    return grad;
}

void LayerUnflatten::to_json(json& j) const{}

void LayerUnflatten::load_json(const json& j){}