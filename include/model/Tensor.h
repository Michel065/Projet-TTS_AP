#pragma once

#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <random>

#include "model/Struct.h"

//Version avec xtensor
class Tensor {
public:
    Tensor();
    Tensor(Shape _shape);
    Shape shape;
    
    void init_alea();
    xt::xarray<float>& recup_data();

private:
    xt::xarray<float> data;
};