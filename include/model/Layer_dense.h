#pragma once
#include "model/Layer.h"

class LayerDense : public Layer {
public:
    LayerDense(size_t output_size);

    void build() override;
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

private:
	Tensor* _W = nullptr;
	Tensor* _b = nullptr;
};