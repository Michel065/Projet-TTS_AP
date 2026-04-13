#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>
#include <xtensor/xarray.hpp>

#include "model/Tool/Tensor/TensorDataBase.h"   
#include "model/Tool/Tensor/CudaData.h"

//ajout des methodes liée au kernel
#include "model/Tool/Tensor/TensorKernels.cuh"

class TensorDataGPU : public TensorDataBase {
private:
    CudaData data_gpu;
    size_t get_total_size() const;
    const CudaData& get_data_gpu() const;
    xt::xarray<float> copy_to_cpu() const;
    void copy_from_cpu(const xt::xarray<float>& arr);  

public:
    TensorDataGPU() = default;
    ~TensorDataGPU() override;

    TensorDataGPU(const TensorDataGPU&) = delete;
    TensorDataGPU& operator=(const TensorDataGPU&) = delete;

    const TensorDataGPU* check_gpu(const Tensor& a) const;

    TensorDataBase* clone() const override;

    // les ops
    void apply_add(const Tensor& b) override;
    void apply_sub(const Tensor& b) override;
    void apply_mul(const Tensor& b) override;
    void apply_div(const Tensor& b) override;

    void apply_add(float scalar) override;
    void apply_sub(float scalar) override;
    void apply_mul(float scalar) override;
    void apply_div(float scalar) override;

    // methode qui modifie nos data Tensor
    void init(Shape _shape, bool alea, int val_init = 0) override;
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

    // methode qui crée un nouveau Tensor
    Tensor matmul(const Tensor& b) const override;
    Tensor sum_axis(std::size_t axis, bool keep_dims) const override;
    Tensor sum_per_row() const override;
    Tensor max_per_row() const override;
    Tensor extraction_section_axe_0(int debut, int fin) const override;

    // methode autre
    bool equal(const Tensor& b) const override;
    void recalul_shape() override;
    float moyenne() const override;
    bool scan_for_Nan(bool throww) const override;
    const xt::xarray<float> to_json() const override;

    // methode pour modifier un element par element
    float get(const std::vector<size_t>& indices) const override;
    void set(const std::vector<size_t>& indices, float val) override;

    friend class Tensor;

};