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

template<typename T>
CudaData<T>::~CudaData(){
    free();
}

template<typename T>
CudaData<T>::CudaData(CudaData<T>&& other) noexcept : _data(other._data), _size(other._size){
    other._data = nullptr;
    other._size = 0;
}

template<typename T>
CudaData<T>& CudaData<T>::operator=(CudaData<T>&& other) noexcept{
    if(this == &other) return *this;

    free();

    _data = other._data;
    _size = other._size;

    other._data = nullptr;
    other._size = 0;
    return *this;
}

template<typename T>
void CudaData<T>::allocate(size_t size){ // l'allocation est fixe avec T vue que c'est pas prevue d'utiliser autre chose
    if(size == 0){
        free();
        return;
    }

    if(_data != nullptr){
        free();
        return;
    }

    if(_data != nullptr && _size == size){
        return;
    }

    free();
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&_data), size * sizeof(T)), "cudaMalloc failed");
    _size = size;
}

template<typename T>
void CudaData<T>::free(){
    if(_data != nullptr){
        cudaFree(_data);
        _data = nullptr;
    }
    _size = 0;
}

template<typename T>
void CudaData<T>::copy_from_cpu(const xt::xarray<T>& arr){
    size_t n = arr.size();
    allocate(n);
    if(n == 0) return;
    cuda_check(cudaMemcpy(_data, arr.data(), n * sizeof(T), cudaMemcpyHostToDevice),"cudaMemcpy HostToDevice failed");
}

//en plus pour faciliter indices int
template<typename T>
void CudaData<T>::copy_from_cpu(const std::vector<T>& vec){
    size_t n = vec.size();
    allocate(n);
    if(n == 0) return;
    cuda_check(cudaMemcpy(_data, vec.data(), n * sizeof(T), cudaMemcpyHostToDevice),"cudaMemcpy HostToDevice failed");
}

template<typename T>
xt::xarray<T> CudaData<T>::copy_to_cpu(const Shape& shape) const{
    xt::xarray<T> arr = xt::xarray<T>::from_shape(shape.dims);

    if(_size == 0) return arr;
    if(arr.size() != _size){
        throw std::runtime_error("copy_to_cpu : taille shape incoherente avec _size");
    }

    cuda_check(cudaMemcpy(arr.data(), _data, _size * sizeof(T), cudaMemcpyDeviceToHost),"cudaMemcpy DeviceToHost failed");
    return arr;
}

template<typename T>
void CudaData<T>::copy_from_gpu(const CudaData<T>& other){
    allocate(other._size);

    if(_size == 0) return;
    cuda_check(cudaMemcpy(_data, other._data, _size * sizeof(T), cudaMemcpyDeviceToDevice),"cudaMemcpy DeviceToDevice failed");
}

template<typename T>
T* CudaData<T>::data(){
    return _data;
}

template<typename T>
const T* CudaData<T>::data() const{
    return _data;
}

template<typename T>
size_t CudaData<T>::size() const{
    return _size;
}

template<typename T>
bool CudaData<T>::empty() const{
    return _data == nullptr || _size == 0;
}

// puique on peut pas acceder au data du gpu direct on fait une petite copy d'un elt precis ver une val local qu'on renvoie

template<typename T>
T CudaData<T>::get(size_t index) const{
    if(index >= _size)
        Throw_Error("CudaData get index hors borne");
    T val{};
    cuda_check(cudaMemcpy(&val, _data + index, sizeof(T), cudaMemcpyDeviceToHost),"CudaData get cudaMemcpy DeviceToHost failed");
    return val;
}

template<typename T>
void CudaData<T>::set(size_t index, T val){
    if(index >= _size)
        Throw_Error("CudaData::set index hors borne");

    cuda_check(cudaMemcpy(_data + index, &val, sizeof(T), cudaMemcpyHostToDevice),"CudaData set cudaMemcpy HostToDevice failed");
}





template class CudaData<float>;
template class CudaData<int>;