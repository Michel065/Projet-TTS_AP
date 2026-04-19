#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>
#include <xtensor/xarray.hpp>

#include "model/Tool/Tensor/TensorDataBase.h"   
#include "model/Tool/Tensor/CudaData.h"

//ajout des methodes liée au kernel
#include "model/Tool/Tensor/TensorFuncKernels.cuh"

class TensorDataGPU : public TensorDataBase {
private:
    CudaData<float> data_gpu;
    size_t get_total_size() const;
    xt::xarray<float> copy_to_cpu() const;
    void copy_from_cpu(const xt::xarray<float>& arr);  
    int calcul_stride(int dim_dep) const;
    bool shape_broadcastable(const Shape& b,int dim_dep=1);
    
    const TensorDataGPU* check_gpu(const Tensor& a) const;
    TensorDataGPU* check_gpu_nc(Tensor& a) const;

public:
    TensorDataGPU() = default;
    ~TensorDataGPU() override;
    TensorDataBase* clone() const override;

    
    // on bloque la copy direct pour eviter de faire des bétises du coup si je evxu une copy je suis forcé d'utilsier le clone qui est sure 
    TensorDataGPU(const TensorDataGPU&) = delete;
    TensorDataGPU& operator=(const TensorDataGPU&) = delete;

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
    void apply_exp() override;
    void apply_pow(float val) override;
    void apply_max(float val) override;
    void apply_round(int decimals) override;
    void apply_clip(float b_min, float b_max) override;
    void apply_log() override;
    void calcul_sup(float scalar) override;
    void transpose(bool batch=false) override;
    void reshape(Shape format) override;
    void shuffle(const std::vector<int>& indices) override;

    // methode qui crée un nouveau Tensor
    Tensor matmul(const Tensor& b) const override;
    Tensor sum_axis(std::size_t axis, bool keep_dims) const override;
    Tensor sum_per_row() const override;
    Tensor max_per_row() const override;
    Tensor extraction_section_axe_0(int debut, int fin) const override;

    // methode autre
    float moyenne() const override;
    const xt::xarray<float> to_json() const override;

    // methode pour modifier un element par element
    float get(const std::vector<size_t>& indices) const override;
    void set(const std::vector<size_t>& indices, float val) override;

    // methode pour get de l'exterieur pas propre mais simple pour l'instant
    const CudaData<float>& get_data_gpu() const;
    CudaData<float>& get_data_gpu();

    friend class Tensor;

};