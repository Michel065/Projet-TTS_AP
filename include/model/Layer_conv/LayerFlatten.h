#pragma once
#include "model/Layer.h"

class LayerFlatten : public Layer {
public:
    LayerFlatten();

    void build() override;
    Tensor forward(Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

    void to_json(json& j) const override;
    void load_json(const json& j) override;

private:
    Shape _shape_input_batch_save;
    size_t _total = 1;
    inline static AutoRegisterLayer<LayerFlatten> enregistrement{};
};