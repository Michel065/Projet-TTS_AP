#pragma once
#include "model/Layer.h"

class LayerSigmoid : public Layer {
public:
    LayerSigmoid() : Layer("Sigmoid"){}

    Tensor sigmoid(const Tensor& x);
    Tensor sigmoid_grad();
    
    void build() override;
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

private:
	Tensor _last_output;
};