#pragma once
#include "model/Layer.h"
#include "model/Layer_conv/LayerMaxPool2D/LayerMaxPoolCPU.h"
#include "model/Layer_conv/LayerMaxPool2D/LayerMaxPoolGPU.h"

class LayerMaxPool2D : public Layer {
public:
    LayerMaxPool2D();

    void build() override;
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& grad) override;

    void to_json(json& j) const override;
    void load_json(const json& j) override;

private:
    size_t _taille_batch;
    Tensor _mask; // pour backward (position du max)

    inline static AutoRegisterLayer<LayerMaxPool2D> enregistrement{};
};