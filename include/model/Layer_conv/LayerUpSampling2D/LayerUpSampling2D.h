#pragma once
#include "model/Layer.h"
#include "model/Layer_conv/LayerUpSampling2D/LayerUpSampling2DGPU.h"

class LayerUpSampling2D : public Layer {
public:
    LayerUpSampling2D();

    void build() override;
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& grad) override;

    void to_json(json& j) const override;
    void load_json(const json& j) override;

private:
    size_t _taille_batch;
    inline static AutoRegisterLayer<LayerUpSampling2D> enregistrement{};
};