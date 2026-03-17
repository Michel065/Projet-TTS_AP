#include "model/Tensor.h"

Tensor::Tensor(Shape _shape){
    shape = _shape;
    data = xt::zeros<float>(shape.dims);
}

Tensor::Tensor(){}

void Tensor::init_alea(){
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    for(auto& v : data.storage()){
        v = dist(gen);
    }
}

xt::xarray<float>& Tensor::recup_data(){
    return data;
}

const xt::xarray<float>& Tensor::recup_data() const{
    return data;
}

Tensor Tensor::operator+(const Tensor& other) const{
    if(shape.dims != other.shape.dims){
        throw std::runtime_error("Shape mismatch");
    }

    Tensor res(shape);
    res.data = data + other.data;
    return res;
}

Tensor Tensor::operator-(const Tensor& other) const{
    if(shape.dims != other.shape.dims){
        throw std::runtime_error("Shape mismatch");
    }

    Tensor res(shape);
    res.data = data - other.data;
    return res;
}

Tensor Tensor::operator*(const Tensor& other) const{
    if(shape.dims != other.shape.dims){
        throw std::runtime_error("Shape mismatch");
    }

    Tensor res(shape);
    res.data = data * other.data;
    return res;
}

Tensor Tensor::operator/(const Tensor& other) const{
    if(shape.dims != other.shape.dims){
        throw std::runtime_error("Shape mismatch");
    }

    Tensor res(shape);
    res.data = data / other.data;
    return res;
}

Tensor& Tensor::operator+=(const Tensor& other){
    if(shape.dims != other.shape.dims){
        throw std::runtime_error("Shape mismatch");
    }

    data += other.data;
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other){
    if(shape.dims != other.shape.dims){
        throw std::runtime_error("Shape mismatch");
    }

    data -= other.data;
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other){
    if(shape.dims != other.shape.dims){
        throw std::runtime_error("Shape mismatch");
    }

    data *= other.data;
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other){
    if(shape.dims != other.shape.dims){
        throw std::runtime_error("Shape mismatch");
    }

    data /= other.data;
    return *this;
}

Tensor Tensor::operator+(float scalar) const{
    Tensor res(shape);
    res.data = data + scalar;
    return res;
}

Tensor Tensor::operator-(float scalar) const{
    Tensor res(shape);
    res.data = data - scalar;
    return res;
}

Tensor Tensor::operator*(float scalar) const{
    Tensor res(shape);
    res.data = data * scalar;
    return res;
}

Tensor Tensor::operator/(float scalar) const{
    if(scalar == 0.f){
        throw std::runtime_error("Division by zero");
    }

    Tensor res(shape);
    res.data = data / scalar;
    return res;
}

Tensor& Tensor::operator+=(float scalar){
    data += scalar;
    return *this;
}

Tensor& Tensor::operator-=(float scalar){
    data -= scalar;
    return *this;
}

Tensor& Tensor::operator*=(float scalar){
    data *= scalar;
    return *this;
}

Tensor& Tensor::operator/=(float scalar){
    if(scalar == 0.f){
        throw std::runtime_error("Division by zero");
    }

    data /= scalar;
    return *this;
}

Tensor Tensor::operator-() const{
    Tensor res(shape);
    res.data = -data;
    return res;
}

bool Tensor::operator==(const Tensor& other) const{
    return shape.dims == other.shape.dims && xt::all(xt::equal(data, other.data));
}

bool Tensor::operator!=(const Tensor& other) const{
    return !(*this == other);
}

Tensor Tensor::prod_mat(const Tensor& other) const {
    if(shape.dims.size() != 2 || other.shape.dims.size() != 2)
        throw std::runtime_error("prod_mat requires 2D tensors");

    if(shape.dims[1] != other.shape.dims[0])
        throw std::runtime_error("prod_mat shape mismatch");

    Shape res_shape({shape.dims[0], other.shape.dims[1]});
    Tensor res(res_shape);

    for(size_t i = 0; i < shape.dims[0]; i++){
        for(size_t j = 0; j < other.shape.dims[1]; j++){

            float sum = 0.f;

            for(size_t k = 0; k < shape.dims[1]; k++){
                sum += data(i,k) * other.data(k,j);
            }

            res.recup_data()(i,j) = sum;
        }
    }

    return res;
}


std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor shape=" << t.shape.print() << "\n";
    os << t.data;
    return os;
}