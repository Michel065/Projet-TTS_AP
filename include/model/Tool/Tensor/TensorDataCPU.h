#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xjson.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xeval.hpp>

#include <random>
#include <vector>
#include <stdexcept>

#include "model/Tool/Tensor/TensorDataBase.h"

class TensorDataCPU : public TensorDataBase {
private:
    xt::xarray<float> data_cpu;

public:
    TensorDataCPU() = default;
    explicit TensorDataCPU(const xt::xarray<float>& d) : data_cpu(d) {}

    const TensorDataCPU* check_cpu(const Tensor& a) const ;

    TensorDataBase* clone() const override;
    //les ops
    void apply_add(const Tensor& b) override;
    void apply_sub(const Tensor& b) override;
    void apply_mul(const Tensor& b) override;
    void apply_div(const Tensor& b) override;

    void apply_add(float scalar) override;
    void apply_sub(float scalar) override;
    void apply_mul(float scalar) override;
    void apply_div(float scalar) override;
    
    //methode qui modifie nos data Tensor
    void init(Shape _shape, bool alea, int val_init) override;
    void init_with_data(const xt::xarray<float>& arr) override;
    void fill_alea() override;
    void apply_exp() override;
    void apply_pow(float val) override;
    void apply_max(float val) override;
    void apply_round(int decimals) override;
    void apply_clip(float b_min, float b_max) override;
    void apply_log() override;
    void calcul_sup(float scalar) override;
    void transpose() override;
    void reshape(Shape format) override;
    
    //methode qui créer un nouveau Tensor
    Tensor matmul(const Tensor& b) const override;
    Tensor sum_axis(std::size_t axis, bool keep_dims) const override;
    Tensor sum_per_row() const override;
    Tensor max_per_row() const override;
    Tensor extraction_section_axe_0(int debut, int fin) const override;

    //methode autre
    bool equal(const Tensor& b) const override;
    void recalul_shape() override;
    float moyenne() const override;
    bool scan_for_Nan(bool throww) const override;
    const xt::xarray<float>& get_format_xr() const override;
    friend std::ostream& operator<<(std::ostream& os, const TensorDataCPU* t);

    //methode pour modifier un element par element
    float get(const std::vector<size_t>& indices) const override;
    void set(const std::vector<size_t>& indices, float val) override;

    friend class Tensor;   
};