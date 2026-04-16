#pragma once
#include <xtensor/xarray.hpp>

#include <curand.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/Tool/Tensor/CudaConfig.cuh"
#include "model/Tool/Shape.h"
#include "outil/Print.h"

template<typename T = float>
class CudaData {
private:
    T* _data = nullptr;
    size_t _size = 0;

public:
    CudaData() = default;
    ~CudaData();

    CudaData(const CudaData&) = delete;
    CudaData& operator=(const CudaData&) = delete;

    CudaData(CudaData&& other) noexcept;
    CudaData& operator=(CudaData&& other) noexcept;

    void allocate(size_t size);
    void free();

    void copy_from_cpu(const xt::xarray<T>& arr);
    void copy_from_cpu(const std::vector<T>& vec);
    xt::xarray<T> copy_to_cpu(const Shape& shape) const;

    void copy_from_gpu(const CudaData& other);

    T* data();
    const T* data() const;

    size_t size() const;
    bool empty() const;

    T get(size_t index) const;
    void set(size_t index, T val);

    void fill_zero();
    void fill_value(T val);
    void fill_random();
};


/*
Source : https://docs.nvidia.com/cuda/cuda-c-programming-guide/#
*/