// ici et pas dans le .h sinon warning 
#include <cuda_runtime.h>

#include "model/Tool/Tensor/CudaData.h"

namespace {
    void cuda_check(cudaError_t err, const char* msg){
        if(err != cudaSuccess){
            Throw_Error(std::string(msg)," : ",cudaGetErrorString(err));
        }
    }
}

CudaData::~CudaData(){
    free();
}

CudaData::CudaData(CudaData&& other) noexcept
    : _data(other._data), _size(other._size){
    other._data = nullptr;
    other._size = 0;
}

CudaData& CudaData::operator=(CudaData&& other) noexcept{
    if(this == &other) return *this;

    free();

    _data = other._data;
    _size = other._size;

    other._data = nullptr;
    other._size = 0;

    return *this;
}

void CudaData::allocate(size_t size){ // l'allocation est fixe avec float vue que c'est pas prevue d'utiliser autre chose
    if(size == 0){
        free();
        return;
    }

    if(_data != nullptr && _size == size){
        return;
    }

    free();
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&_data), size * sizeof(float)), "cudaMalloc failed");
    _size = size;
}

void CudaData::free(){
    if(_data != nullptr){
        cudaFree(_data);
        _data = nullptr;
    }
    _size = 0;
}

void CudaData::copy_from_cpu(const xt::xarray<float>& arr){
    size_t n = arr.size();
    allocate(n);

    if(n == 0) return;

    cuda_check(
        cudaMemcpy(_data, arr.data(), n * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy HostToDevice failed"
    );
}

xt::xarray<float> CudaData::copy_to_cpu(const std::vector<size_t>& shape) const{
    xt::xarray<float> arr = xt::xarray<float>::from_shape(shape);

    if(_size == 0) return arr;
    if(arr.size() != _size){
        throw std::runtime_error("copy_to_cpu : taille shape incoherente avec _size");
    }

    cuda_check(
        cudaMemcpy(arr.data(), _data, _size * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy DeviceToHost failed"
    );

    return arr;
}

void CudaData::copy_from_gpu(const CudaData& other){
    allocate(other._size);

    if(_size == 0) return;

    cuda_check(
        cudaMemcpy(_data, other._data, _size * sizeof(float), cudaMemcpyDeviceToDevice),
        "cudaMemcpy DeviceToDevice failed"
    );
}

float* CudaData::data(){
    return _data;
}

const float* CudaData::data() const{
    return _data;
}

size_t CudaData::size() const{
    return _size;
}

bool CudaData::empty() const{
    return _data == nullptr || _size == 0;
}