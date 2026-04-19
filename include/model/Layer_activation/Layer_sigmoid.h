#pragma once
#include "model/Layer.h"

class LayerSigmoid : public Layer {
public:
    LayerSigmoid() : Layer("Sigmoid"){}

    Tensor sigmoid(const Tensor& x);
    Tensor sigmoid_grad();
    
    void build() override;
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& grad) override;

    void to_json(json& j) const override;
    void load_json(const json& j) override;

private:
	Tensor _last_output;
    inline static AutoRegisterLayer<LayerSigmoid> enregistrement{};
};