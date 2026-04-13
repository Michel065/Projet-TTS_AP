#pragma once
#include <xtensor/xarray.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/Tool/Shape.h"
#include "outil/Print.h"

class CudaData {
private:
    float* _data = nullptr;
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

    void copy_from_cpu(const xt::xarray<float>& arr);
    xt::xarray<float> copy_to_cpu(const Shape& shape) const;

    void copy_from_gpu(const CudaData& other);

    float* data();
    const float* data() const;

    size_t size() const;
    bool empty() const;
};

/*
Source : https://docs.nvidia.com/cuda/cuda-c-programming-guide/#
*/