#pragma once
#include "model/Layer.h"

class LayerSoftMax : public Layer {
public:
    LayerSoftMax() : Layer("SoftMax"){}

    Tensor softmax(const Tensor& x);

    void build() override;
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

private:
    Tensor _last_output;
};