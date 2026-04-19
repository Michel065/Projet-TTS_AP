#pragma once
#include "model/Layer.h"

enum class TypeNormalisation {
    DEFAULT,//  0 , 1
    ALTERNATIVE, //-1,1
};

class LayerNormalisation : public Layer {
public:
    LayerNormalisation();
    LayerNormalisation(xt::xarray<float> min, xt::xarray<float> max,TypeNormalisation type_norm = TypeNormalisation::ALTERNATIVE );

    Tensor calc_defaut(const Tensor& input);
    Tensor calc_defaut_grad(const Tensor& grad);

    Tensor calc_alternative(const Tensor& input);
    Tensor calc_alternative_grad(const Tensor& grad);

    void build() override;
    void get_from_model() override;
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& grad) override;
    
    void to_json(json& j) const override;
    void load_json(const json& j) override;

private:
    DeviceType _device;
    xt::xarray<float> _vmin,_vmax;

    Tensor _min;
    Tensor _max;
    TypeNormalisation _type_norm;
    
    inline static AutoRegisterLayer<LayerNormalisation> enregistrement{};
};