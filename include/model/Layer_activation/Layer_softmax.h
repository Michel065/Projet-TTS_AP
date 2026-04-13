#pragma once
#include "model/Layer.h"

class LayerSoftMax : public Layer {
public:
    LayerSoftMax() : Layer("SoftMax"){}

    Tensor softmax(const Tensor& x);

    void build() override;
    Tensor forward(Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

    void to_json(json& j) const override;
    void load_json(const json& j) override;

private:
    Tensor _last_output;
    inline static AutoRegisterLayer<LayerSoftMax> enregistrement{};
};