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

Shape Layer::get_input_shape(){
    return _shape_input;
}

void Layer::print_couche_msg(std::string msg,Color couleur){
    Print_Color(couleur, "Couche "+nom_couche+"[",_shape_input.print(),";",_shape_output.print(),"]:"+msg);
}

void Layer::init_eta(float eta){
    _eta=eta;
}

void Layer::print(){
    Print_Color(Color::PINK,"Couche ", nom_couche, "[",_shape_input.print(), ";",_shape_output.print(), "] : ",_nb_params, " params");
}