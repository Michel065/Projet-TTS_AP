// ici et pas dans le .h sinon warning 
#include <cuda_runtime.h>

#include "model/Tool/Tensor/CudaData.h"

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









template<typename T>
void CudaData<T>::fill_zero(){
    if(_data == nullptr)
        Throw_Error("CudaData::fill_zero data null");
    cuda_check(cudaMemset(_data, 0, _size * sizeof(T)),"cudaMemset failed");
}


__global__ void fill_value_kernel(float* data, float val, size_t n){ // kernel ici pour eviter de créer un nouveau fichier
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n) return;
    data[id] = val;
}

template<>
void CudaData<float>::fill_value(float val){
    if(_data == nullptr)
        Throw_Error("CudaData::fill_value data null");

    //version kernel
    int blocks = CudaConfig::calculs_blocks_1D(_size);
    fill_value_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(_data, val, _size);
    cuda_check_all("fill_value_kernel");
}
//le genreique:
template<typename T>
void CudaData<T>::fill_value(T val){
    Throw_Error("fill_value non implémenté pour ce type");
}




__global__ void scale_kernel(float* data, size_t n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n) return;
    data[id] = data[id] * 2.0f - 1.0f;
}

template<>
void CudaData<float>::fill_random(){
    if(_data == nullptr)
        Throw_Error("CudaData::fill_random data null");
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    // génère entre 0 1
    curandGenerateUniform(gen, _data, _size);
    curandDestroyGenerator(gen);

    //version kernel pour passé de 0 1 à -1 1
    int blocks = CudaConfig::calculs_blocks_1D(_size);
    scale_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(_data, _size);
    cuda_check_all("curandGenerateUniform");
}
//le genreique:
template<typename T>
void CudaData<T>::fill_random(){
    Throw_Error("fill_value non implémenté pour ce type");
}










template class CudaData<float>;
template class CudaData<int>;