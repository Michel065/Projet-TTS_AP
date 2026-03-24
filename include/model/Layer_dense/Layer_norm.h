#pragma once
#include "model/Layer.h"

enum class TypeNormalisation {
    DEFAULT,//  0 , 1
    ALTERNATIVE, //-1,1
};

class LayerNormalisation : public Layer {
public:
    LayerNormalisation(float min, float max,TypeNormalisation type_norm = TypeNormalisation::ALTERNATIVE );

    Tensor calc_defaut(const Tensor& input);
    Tensor calc_defaut_grad(const Tensor& grad);

    Tensor calc_alternative(const Tensor& input);
    Tensor calc_alternative_grad(const Tensor& grad);

    void build() override;
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

private:
    float _min;
    float _max;
    TypeNormalisation _type_norm;
};