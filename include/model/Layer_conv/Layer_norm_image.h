#pragma once
#include "model/Layer.h"

class LayerNormalisationImage : public Layer {
public:
    LayerNormalisationImage();

    void build() override;
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& grad) override;
    
    void to_json(json& j) const override;
    void load_json(const json& j) override;

private:
    DeviceType _device;
    
    inline static AutoRegisterLayer<LayerNormalisationImage> enregistrement{};
};