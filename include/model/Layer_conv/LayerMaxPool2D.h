#pragma once
#include "model/Layer.h"

class LayerMaxPool2D : public Layer {
public:
    LayerMaxPool2D();

    void build() override;
    Tensor forward(Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

    void to_json(json& j) const override;
    void load_json(const json& j) override;

private:
    size_t _channels = 1;

    size_t _last_batch;
    Tensor _mask; // pour backward (position du max)

    inline static AutoRegisterLayer<LayerMaxPool2D> enregistrement{};
};