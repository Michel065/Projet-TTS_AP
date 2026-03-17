#include "model/Tensor.h"

Tensor::Tensor(Shape _shape){
    shape=_shape;
    data = xt::zeros<float>(shape.dims);
}

Tensor::Tensor(){}

void Tensor::init_alea(){

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    
    for (auto& v : data.storage())
        v = dist(gen);
}

xt::xarray<float>& Tensor::recup_data() {
    return data;
}