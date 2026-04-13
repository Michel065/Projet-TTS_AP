#include "model/Tool/Tensor/Tensor.h"
#include "model/Tool/Tensor/TensorDataCPU.h"
#include "model/Tool/Tensor/TensorDataBase.h"

//pour simplifier la suite ajout de clone 
Tensor& Tensor::operator=(const Tensor& other){
    if(this == &other) return *this;
    delete _data;
    _data = nullptr;
    device = other.device;
    _data = (other._data != nullptr) ? other._data->clone() : nullptr;
    return *this;
}

// et transfere
Tensor& Tensor::operator=(Tensor&& other) noexcept{
    if(this == &other) return *this;
    delete _data;
    device = other.device;
    _data = other._data;
    other._data = nullptr;
    return *this;
}

Tensor Tensor::operator+(const Tensor& other) const{
    Tensor res = *this;
    res.get_data()->apply_add(other);
    return res;
}

Tensor Tensor::operator-(const Tensor& other) const{
    check_data();
    Tensor res = *this;
    res.get_data()->apply_sub(other);
    return res;
}

Tensor Tensor::operator*(const Tensor& other) const{
    check_data();
    Tensor res = *this;
    res.get_data()->apply_mul(other);
    return res;
}

Tensor Tensor::operator/(const Tensor& other) const{
    check_data();
    Tensor res = *this;
    res.get_data()->apply_div(other);
    return res;
}

Tensor Tensor::operator+(float scalar) const{
    check_data();
    Tensor res = *this;
    res.get_data()->apply_add(scalar);
    return res;
}

Tensor Tensor::operator-(float scalar) const{
    check_data();
    Tensor res = *this;
    res.get_data()->apply_sub(scalar);
    return res;
}

Tensor Tensor::operator*(float scalar) const{
    check_data();
    Tensor res = *this;
    res.get_data()->apply_mul(scalar);
    return res;
}

Tensor Tensor::operator/(float scalar) const{
    check_data();
    if(scalar == 0.f) Throw_Error("Division par zero");
    Tensor res = *this;
    res.get_data()->apply_div(scalar);
    return res;
}



// revoir ici possibement un probleme ondoit trouver une sol pour ne chosir si on copy ou pas
Tensor& Tensor::operator+=(float scalar){
    check_data();
    _data->apply_add(scalar);
    return *this;
}

Tensor& Tensor::operator-=(float scalar){
    check_data();
    _data->apply_sub(scalar);
    return *this;
}

Tensor& Tensor::operator*=(float scalar){
    check_data();
    _data->apply_mul(scalar);
    return *this;
}

Tensor& Tensor::operator/=(float scalar){
    check_data();
    if(scalar == 0.f) Throw_Error("Division by zero");
    _data->apply_div(scalar);
    return *this;
}

Tensor& Tensor::operator+=(const Tensor& other){
    check_data();
    _data->apply_add(other);
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other){
    check_data();
    _data->apply_sub(other);
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other){
    check_data();
    _data->apply_mul(other);
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other){
    check_data();
    _data->apply_div(other);
    return *this;
}

Tensor operator+(float scalar, const Tensor& t){
    Tensor res = t;
    res.get_data()->apply_add(scalar);
    return res;
}

Tensor operator-(float scalar, const Tensor& t){
    if(t.get_data() == nullptr) Throw_Error("data de Tensor est nullptr");
    Tensor res = t;
    res.get_data()->apply_mul(-1);
    res.get_data()->apply_add(scalar);
    return res;
}

Tensor operator*(float scalar, const Tensor& t){
    Tensor res = t;
    res.get_data()->apply_mul(scalar);
    return res;
}

Tensor operator/(float scalar, const Tensor& t){
    if(t.get_data() == nullptr) Throw_Error("data de Tensor est nullptr");
    Tensor res = t;
    res.get_data()->apply_pow(-1);   // 1 / t
    res.get_data()->apply_mul(scalar); // scalar * (1 / t)
    return res;
}

Tensor Tensor::operator>(float scalar) const {
    check_data();
    Tensor res = *this;
    res.get_data()->calcul_sup(scalar);
    return res;
}

bool Tensor::operator==(const Tensor& other) const{
    check_data();
    return _data->get_shape().dims == other.get_shape().dims && _data->equal(other);
}

bool Tensor::operator!=(const Tensor& other) const{
    return !(*this == other);
}

std::string get_device_str(const DeviceType& d){
    switch(d){
        case DeviceType::CPU: return std::string("CPU");
        case DeviceType::GPU: return std::string("GPU");
        default: return std::string("Unknown"); 
    }
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor type:" << get_device_str(t.get_device()) << " shape=" << t.get_shape().print() << "\n";
    os << t.get_data_format_xr();
    return os;
}
