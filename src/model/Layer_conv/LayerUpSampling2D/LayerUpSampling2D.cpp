#include "model/Layer_conv/LayerUpSampling2D/LayerUpSampling2D.h"
#include "model/Model.h"

LayerUpSampling2D::LayerUpSampling2D() : Layer("Conv2D"){}

void LayerUpSampling2D::build(){
    if( _shape_input.len()!=3){   
        Throw_Error("Dimensions non valides."+_shape_input.print());
        return;
    } 
    set_output_shape(Shape({_shape_input[0], _shape_input[1] * 2, _shape_input[2] * 2})); // channel, H*2 , W*2
    print_couche_msg("Build termine.",Color::GREEN);
}

Tensor LayerUpSampling2D::forward(Tensor& input){
    Shape in_sh = input.get_shape();
    _taille_batch = in_sh[0];

    Tensor output(input.get_device(),Shape({_taille_batch, _shape_output[0], _shape_output[1], _shape_output[2]}), false);
    if(input.is_cpu()){
        Throw_Error("Methode LayerUpSampling2DCPU::forward pas implemnté !!! a faire i besoin");
    }else if(input.is_gpu()){
        LayerUpSampling2DGPU::forward(output, input, _taille_batch, _shape_input, _shape_output);
    }
    return output;
}


Tensor LayerUpSampling2D::backward(Tensor& grad){
    Tensor grad_out(grad.get_device(),Shape({_taille_batch,_shape_input[0],_shape_input[1],_shape_input[2]}), false);
    if(grad.is_cpu()){
        Throw_Error("Methode LayerUpSampling2DCPU::backward pas implemnté !!! a faire i besoin");
    }else if(grad.is_gpu()){
        LayerUpSampling2DGPU::backward(grad_out,grad, _taille_batch, _shape_input, _shape_output);
    } 
    return grad_out;
}

void LayerUpSampling2D::to_json(json& j) const{}

void LayerUpSampling2D::load_json(const json& j) {}