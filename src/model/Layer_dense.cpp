#include "model/layer_dense.h"

LayerDense::LayerDense(size_t output_size){
    Shape s({output_size});
    set_output_shape(s);

    nom_couche = "dense";
}

void LayerDense::build(){
    if( _shape_input.len()!=1 || _shape_output.len()!=1 ){   
        print_couche_msg("Erreur dimensions non valides. entre:"+_shape_input.print()+" ou sortie:"+_shape_output.print()+" ",Color::RED);
        return;
    } 
    Shape shape_poid({_shape_input[0],_shape_output[0]});
    Shape shape_b({1,_shape_output[0]});

    _W = new Tensor(shape_poid);
    _b = new Tensor(shape_b);
    _W->init_alea();  

    print_couche_msg("Build termine.",Color::GREEN);
}

Tensor LayerDense::forward(const Tensor& input){
    return Tensor();
}


Tensor LayerDense::backward(const Tensor& grad){
    return Tensor();
}