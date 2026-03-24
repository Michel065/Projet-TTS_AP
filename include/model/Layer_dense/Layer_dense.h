#pragma once
#include "model/Layer.h"

class LayerDense : public Layer {
public:
    LayerDense(size_t output_size);

    void build() override;
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

private:
	Tensor _W;
	Tensor _b;
	Tensor _last_input; //pour le retour
    float _eta=0.f;
};