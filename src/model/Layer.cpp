#include "model/Layer.h"

void Layer::set_input_shape(Shape shape_input){
    _shape_input = shape_input;
    build();
}

void Layer::set_output_shape(Shape shape_output){
    _shape_output = shape_output;
}

Shape Layer::get_output_shape(){
    return _shape_output;
}

Shape Layer::get_input_shape() const{
    return _shape_input;
}

void Layer::print_couche_msg(std::string msg,Color couleur){
    Print_Color(couleur, "Couche "+nom_couche+"[",_shape_input.print(),";",_shape_output.print(),"]:"+msg);
}

void Layer::set_model(Model* model_global){
    _model=model_global;
}

void Layer::print(){
    Print_Color(Color::PINK,"Couche ", nom_couche, "[",_shape_input.print(), ";",_shape_output.print(), "] : ",_nb_params, " params");
}

int Layer::get_nbr_params(){
    return _nb_params;
}

std::string Layer::get_name(){
    return nom_couche;
}

json Layer::to_json_layer() const {
    return {
        {"type", nom_couche},
        {"shape_input", _shape_input},
        {"shape_output", _shape_output}
    };
}

void Layer::load_json_layer(const json& j) {
    j.at("shape_input").get_to(_shape_input);
    j.at("shape_output").get_to(_shape_output);
}