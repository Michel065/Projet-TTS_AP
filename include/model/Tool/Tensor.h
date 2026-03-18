#pragma once

#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <random>
#include <stdexcept>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/core/xmath.hpp>

#include "model/Tool/Shape.h"
#include "outil/Print.h"

/*
La Class tenor est juste la pour faire tampon et donc si je veux plus tard remplacer par des tansor fait a la main c facile, bon peu de chance etant donné la perte de perf mais on sait jamais
*/

class Tensor {
public:
    Tensor();
    Tensor(const xt::xarray<float>& arr);
    Tensor(Shape _shape,bool alea=false,int val_init=0);

    Shape shape;

    //fonction classique
    void recalul_shape();

    void init_alea();
    
    Tensor prod_mat(const Tensor& other) const;
    Tensor transpose() const;//pour la propagation
    Tensor sum_axis(std::size_t axis, bool keep_dims) const;
    Tensor exp() const;
    Tensor max(float val=0.0f) const;
    Tensor sum_per_row() const;
    Tensor max_per_row() const;
    Tensor round(int decimals) const;

    //test
    void set(std::initializer_list<int> indices, float value);


    xt::xarray<float>& recup_data();
    const xt::xarray<float>& recup_data() const;


    //surcharge d'op
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

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

    // a l'exterieur car trop de param
    friend Tensor operator+(float scalar, const Tensor& t);
    friend Tensor operator-(float scalar, const Tensor& t);
    friend Tensor operator*(float scalar, const Tensor& t);
    friend Tensor operator/(float scalar, const Tensor& t);

    Tensor operator>(float scalar) const;

private:
    xt::xarray<float> data;
};
