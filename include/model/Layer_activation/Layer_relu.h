#pragma once
#include "model/Layer.h"

class LayerRelu : public Layer {
public:
    LayerRelu() : Layer("Relu"){}

    Tensor relu(const Tensor& x);
    Tensor relu_grad();

    void build() override;
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

private:
	Tensor _last_output;
};