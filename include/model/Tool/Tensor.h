#pragma once

#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <random>
#include <vector>
#include <stdexcept>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xjson.hpp>

#include "model/Tool/Shape.h"
#include "outil/Print.h"
#include "model/Json/Json_gestion.h"

/*
La Class tenor est juste la pour faire tampon et donc si je veux plus tard remplacer par des tansor fait a la main c facile, bon peu de chance etant donné la perte de perf mais on sait jamais
*/

class Tensor {
public:
    Tensor();
    Tensor(const xt::xarray<float>& arr);
    Tensor(Shape _shape,bool alea=false,int val_init=0);
    Tensor(const std::vector<std::vector<float>>& vec);
    Tensor(const std::vector<float>& vec);

    Shape shape;

    //fonction classique
    void recalul_shape();

    void init_alea();
    
    Tensor prod_mat(const Tensor& other) const;
    Tensor transpose() const;//pour la propagation
    Tensor sum_axis(std::size_t axis, bool keep_dims) const;
    Tensor exp() const;
    float moyenne() const;
    Tensor pow(float val) const;
    int size() const;
    Tensor max(float val=0.0f) const;
    Tensor sum_per_row() const;
    Tensor max_per_row() const;
    Tensor round(int decimals) const;
    std::vector<Tensor> separation_batch(int batch_size)const;
    Tensor extraction_section_axe_0(int debut, int fin) const;
    Tensor clip(float b_min,float b_max) const;
    Tensor log() const;
    bool scan_for_Nan(bool throww=true) const;
    
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

    //ajout pour la convolution
    template<typename... Args>
    float& operator()(Args... args){
        return data(args...);
    }

    template<typename... Args>
    float operator()(Args... args) const{
        return data(args...);
    }

    // pour le flatten
    Tensor& reshape(Shape format);


private:
    xt::xarray<float> data;
};

inline void from_json(const json& j, Tensor& tensor) {
    tensor = Tensor(j.get<xt::xarray<float>>());
    tensor.recalul_shape();
}

inline void to_json(json& j, const Tensor& tensor) {
    j = tensor.recup_data();
}