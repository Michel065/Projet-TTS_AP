#pragma once
#include "model/Layer.h"

class LayerIdentity : public Layer {
public:
    LayerIdentity();

    void build() override;
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& grad) override;

    void to_json(json& j) const override;
    void load_json(const json& j) override;
private:
    inline static AutoRegisterLayer<LayerIdentity> enregistrement{};
};