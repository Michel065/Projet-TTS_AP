#pragma once

#include <cstddef>
#include "model/Tool/Shape.h"
#include "model/Json/Json_gestion.h"

class Tensor;

class TensorDataBase {
public:
    Shape shape;
    
    virtual ~TensorDataBase() = default;

    virtual TensorDataBase* clone() const = 0;
    
    virtual void apply_add(const Tensor& b) = 0;
    virtual void apply_sub(const Tensor& b) = 0;
    virtual void apply_mul(const Tensor& b) = 0;
    virtual void apply_div(const Tensor& b) = 0;

    virtual void apply_add(float scalar) = 0;
    virtual void apply_sub(float scalar) = 0;
    virtual void apply_mul(float scalar) = 0;
    virtual void apply_div(float scalar) = 0;


    //methode qui modifie nos data Tensor
    virtual void init(Shape _shape, bool alea, int val_init) = 0;
    virtual void init_with_data(const xt::xarray<float>& arr) = 0;
    virtual void apply_exp() = 0;
    virtual void apply_pow(float val) = 0;
    virtual void apply_max(float val) = 0;
    virtual void apply_round(int decimals) = 0;
    virtual void apply_clip(float b_min, float b_max) = 0;
    virtual void apply_log() = 0;
    virtual void calcul_sup(float scalar) = 0;
    virtual void transpose() = 0;
    virtual void reshape(Shape format) = 0;
    virtual void shuffle(const std::vector<int>& indices) = 0;

    //methode qui créer un nouveau Tensor
    virtual Tensor matmul(const Tensor& b) const = 0;
    virtual Tensor sum_axis(std::size_t axis, bool keep_dims) const = 0;
    virtual Tensor sum_per_row() const = 0;
    virtual Tensor max_per_row() const = 0;
    virtual Tensor extraction_section_axe_0(int debut, int fin) const = 0;

    //methode autre
    virtual float moyenne() const = 0;
    virtual const xt::xarray<float> to_json() const = 0;

    //methode pour modifier un element par element ( au cas ou, pas utile en theorie)
    virtual float get(const std::vector<size_t>& indices) const = 0;
    virtual void set(const std::vector<size_t>& indices, float val) = 0;

    Shape get_shape(){
        return shape;
    }

};