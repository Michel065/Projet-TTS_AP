#pragma once
#include "model/Layer.h"

class LayerConv2D : public Layer {
public:
    LayerConv2D(size_t nb_filters, size_t kernel);
    LayerConv2D();

    void build() override;
    void get_from_model();
    Tensor forward(Tensor& input) override;
    Tensor backward(const Tensor& grad) override;

    void to_json(json& j) const override;
    void load_json(const json& j) override;

private:
    DeviceType _device;
    size_t _nb_filters = 0;
    size_t _kernel = 0;
    size_t _padding = 0;
    size_t _nbr_channel = 0;

    Tensor _W;
    Tensor _b;
    Tensor _last_input;
    float _eta = 0.f;

    inline static AutoRegisterLayer<LayerConv2D> enregistrement{};
};