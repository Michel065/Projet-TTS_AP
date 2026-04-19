#include "model/Layer_conv/Layer_norm_image.h"
#include "model/Model.h"

LayerNormalisationImage::LayerNormalisationImage(): Layer("Normalisation"){}

void LayerNormalisationImage::build(){
    set_output_shape(_shape_input);
    print_couche_msg("Build termine.", Color::GREEN);
}

Tensor LayerNormalisationImage::forward(Tensor& input){
    input /= 255.0f;
    return input;
}

Tensor LayerNormalisationImage::backward(Tensor& grad){
    return grad; // vide puisque logiquement ca sert a rien de faire, pas de suite normalement
}

void LayerNormalisationImage::to_json(json& j) const{}

void LayerNormalisationImage::load_json(const json& j) {}