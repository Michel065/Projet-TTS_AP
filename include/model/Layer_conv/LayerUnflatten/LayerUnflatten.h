#pragma once
#include "model/Layer.h"

class LayerUnflatten : public Layer {
public:
    LayerUnflatten();
    LayerUnflatten(Shape new_shape);

    void build() override;
    void get_from_model();
    Tensor forward(Tensor& input) override;
    Tensor backward(Tensor& grad) override;

    void to_json(json& j) const override;
    void load_json(const json& j) override;

private:
    Shape _shape_out;
    Shape _shape_input_batch_save;
    inline static AutoRegisterLayer<LayerUnflatten> enregistrement{};
};