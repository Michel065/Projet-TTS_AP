#pragma once

#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <random>
#include <stdexcept>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/core/xmath.hpp>

#include "model/Struct.h"

/*
La Class tenor est juste la pour faire tampon et donc si je veux plus tard remplacer par des tansor fait a la main c facile, bon peu de chance etant donné la perte de perf mais on sait jamais
*/

class Tensor {
public:
    Tensor();
    Tensor(Shape _shape);

    Shape shape;

    void init_alea();

    xt::xarray<float>& recup_data();
    const xt::xarray<float>& recup_data() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    Tensor& operator+=(float scalar);
    Tensor& operator-=(float scalar);
    Tensor& operator*=(float scalar);
    Tensor& operator/=(float scalar);

    Tensor operator-() const;

    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

    Tensor prod_mat(const Tensor& other) const;


    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

private:
    xt::xarray<float> data;
};