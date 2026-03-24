#pragma once
#include "model/Layer.h"

enum class TypeNormalisation {
    DEFAULT,//  0 , 1
    ALTERNATIVE, //-1,1
};

class LayerNormalisation : public Layer {
public:
    LayerNormalisation(std::vector<float> min, std::vector<float> max,TypeNormalisation type_norm = TypeNormalisation::ALTERNATIVE );

    Tensor calc_defaut(const Tensor& input);
    Tensor calc_defaut_grad(const Tensor& grad);

    Tensor calc_alternative(const Tensor& input);
    Tensor calc_alternative_grad(const Tensor& grad);

    void build() override;
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

private:
    Tensor _min;
    Tensor _max;
    TypeNormalisation _type_norm;
};