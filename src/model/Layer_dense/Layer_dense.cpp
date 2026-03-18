#include "model/Layer_dense/Layer_dense.h"

LayerDense::LayerDense(size_t output_size) : Layer("Dense"){
    set_output_shape(Shape({output_size}));
}

void LayerDense::build(){
    if( _shape_input.len()!=1 || _shape_output.len()!=1 ){   
        Throw_Error("Dimensions non valides. entre:"+_shape_input.print()+" ou sortie:"+_shape_output.print()+" ",Color::RED);
        return;
    } 
    Shape shape_poid({_shape_input[0],_shape_output[0]});
    Shape shape_b({1,_shape_output[0]});

    _W = Tensor(shape_poid,true);
    _b = Tensor(shape_b,false);

    //pour le print
    _nb_params = (_shape_input[0] * _shape_output[0]) + _shape_output[0];

    print_couche_msg("Build termine.",Color::GREEN);
}

Tensor LayerDense::forward(const Tensor& input){
    _last_input = input;
    Tensor output_val = input.prod_mat(_W)+_b;
    return output_val;
}


Tensor LayerDense::backward(const Tensor& grad){
    Tensor grad_W;
    Tensor grad_b;
    Tensor grad_prec;

    int batch_size = _last_input.shape[0];
    grad_W = _last_input.transpose().prod_mat(grad)/batch_size;
    grad_b = grad.sum_axis(0, true)/batch_size;
    grad_prec = grad.prod_mat(_W.transpose());

    _W -= grad_W * _eta;
    _b -= grad_b * _eta;
    return grad_prec;
}