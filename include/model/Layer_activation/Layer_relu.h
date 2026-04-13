#pragma once
#include "model/Layer.h"

class LayerRelu : public Layer {
public:
    LayerRelu() : Layer("Relu"){}

    Tensor relu(Tensor& x);
    Tensor relu_grad();

    void build() override;
    Tensor forward(Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

    void to_json(json& j) const override;
    void load_json(const json& j) override;

private:
	Tensor _last_output;
    inline static AutoRegisterLayer<LayerRelu> enregistrement{};
};