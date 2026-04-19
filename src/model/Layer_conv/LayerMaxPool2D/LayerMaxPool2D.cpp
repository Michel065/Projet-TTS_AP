#include "model/Layer_conv/LayerMaxPool2D/LayerMaxPool2D.h"
#include "model/Model.h"

LayerMaxPool2D::LayerMaxPool2D() : Layer("MaxPool2D") {
}

void LayerMaxPool2D::build(){
    if(_shape_input.len() != 3){
        Throw_Error("Dimensions non valides."+_shape_input.print());
        return;
    }
    set_output_shape(Shape({_shape_input[0], _shape_input[1]/2, _shape_input[2]/2}));
    print_couche_msg("Build termine.", Color::GREEN);
}

Tensor LayerMaxPool2D::forward(Tensor& input){
    Shape in_sh = input.get_shape();
    _taille_batch = in_sh[0];
    // output : {batch , channeles, H//2 , W//2}
    Tensor output(input.get_device(),Shape({_taille_batch, _shape_output[0], _shape_output[1], _shape_output[2]}), false);
    _mask = Tensor(input.get_device(),in_sh, false);// pour replacer la val lors du back

    if(input.is_cpu()){
        LayerMaxPoolCPU::forward(output, _mask, input,_taille_batch,_shape_output);
    }else if(input.is_gpu()){
        LayerMaxPoolGPU::forward(output, _mask, input,_taille_batch,_shape_input,_shape_output);
    }
    return output;
}

Tensor LayerMaxPool2D::backward(Tensor& grad){
    Tensor grad_out(grad.get_device(),Shape({_taille_batch,_shape_input[0],_shape_input[1],_shape_input[2]}), false);
    if(grad.is_cpu()){
        LayerMaxPoolCPU::backward(grad_out, _mask, grad,_taille_batch,_shape_output);
    }else if(grad.is_gpu()){
        LayerMaxPoolGPU::backward(grad_out, _mask, grad,_taille_batch,_shape_input,_shape_output);
    }
    return grad_out;
}

void LayerMaxPool2D::to_json(json& j) const{}

void LayerMaxPool2D::load_json(const json& j){}