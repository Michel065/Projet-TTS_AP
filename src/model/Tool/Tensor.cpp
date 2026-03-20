#include "model/Tool/Tensor.h"

Tensor::Tensor(Shape _shape,bool alea,int val_init){
    shape = _shape;
    if(!alea && val_init>0){
        data = xt::ones<float>(shape.dims)*val_init;
    }else{
        data = xt::zeros<float>(shape.dims);
    }
    if(alea)init_alea();
}

Tensor::Tensor(){}

Tensor::Tensor(const xt::xarray<float>& arr){
    data = arr;
    recalul_shape();
}

void Tensor::init_alea(){
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    for(auto& v : data.storage()){
        v = dist(gen);
    }
}

//inutile normalement on est pas supposé avoir le droit de recup la structure interne directement
xt::xarray<float>& Tensor::recup_data(){
    return data;
}
const xt::xarray<float>& Tensor::recup_data() const{
    return data;
}



//surcharge
Tensor Tensor::operator+(const Tensor& other) const{
    Tensor res(shape);
    res.data = data + other.data;
    return res;
}

Tensor Tensor::operator-(const Tensor& other) const{
    Tensor res(shape);
    res.data = data - other.data;
    return res;
}

Tensor Tensor::operator*(const Tensor& other) const{
    Tensor res(shape);
    res.data = data * other.data;
    return res;
}

Tensor Tensor::operator/(const Tensor& other) const{
    Tensor res(shape);
    res.data = data / other.data;
    return res;
}

Tensor& Tensor::operator+=(const Tensor& other){
    data += other.data;
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other){
    data -= other.data;
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other){
    data *= other.data;
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other){
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
        Throw_Error("Division par zero");
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
        Throw_Error("Division by zero");
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
    if(shape.len() != 2 || other.shape.len() != 2)
        Throw_Error("Produit matriciel 2D uniquement");

    if(shape[1] != other.shape[0]){
        Throw_Error("Produit matriciel impossible : Shapes incompatibles");
    }

    Shape res_shape({shape[0], other.shape[1]});
    Tensor res(res_shape);

    for(size_t i = 0; i < shape[0]; i++){
        for(size_t j = 0; j < other.shape[1]; j++){

            float sum = 0.f;

            for(size_t k = 0; k < shape[1]; k++){
                sum += data(i,k) * other.data(k,j);
            }

            res.recup_data()(i,j) = sum;
        }
    }

    return res;
}

Tensor Tensor::transpose() const{
    Tensor res;
    res.data = xt::eval(xt::transpose(data));
    res.recalul_shape();
    return res;
}

Tensor Tensor::sum_axis(std::size_t axis, bool keep_dims) const{
    Tensor res;
    auto tmp = xt::sum(data, {axis});
    if(keep_dims){
        res.data = xt::expand_dims(tmp, axis);
    }else{
        res.data = tmp;
    }
    res.recalul_shape();
    return res;
}


void Tensor::recalul_shape(){
    shape.dims.clear();
    for(auto s : data.shape()){
        shape.dims.push_back(s);
    }
}


Tensor Tensor::exp() const {
    Tensor res = *this;
    res.data = xt::exp(res.data);
    return res;
}

// Juste pour faciliter le print
std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor shape=" << t.shape.print() << "\n";
    os << t.data;
    return os;
}
//______________________


Tensor operator+(float scalar, const Tensor& t){
    Tensor res = t;
    res += scalar;
    return res;
}

Tensor operator-(float scalar, const Tensor& t){
    Tensor res = t;
    res.data = scalar - res.data;
    return res;
}

Tensor operator*(float scalar, const Tensor& t){
    Tensor res = t;
    res *= scalar;
    return res;
}

Tensor operator/(float scalar, const Tensor& t){
    Tensor res = t;
    res.data = scalar / res.data;
    return res;
}

Tensor Tensor::max(float val) const {
    Tensor res = *this;
    res.data = xt::maximum(res.data,val);
    return res;
}

Tensor Tensor::operator>(float scalar) const {
    Tensor res = *this;
    res.data = (data > scalar);
    return res;
}

Tensor Tensor::sum_per_row() const {
    Tensor res(shape);
    res.data = xt::sum(data, {1}, xt::keep_dims);
    return res;
}

Tensor Tensor::max_per_row() const {
    Tensor res(shape);
    res.data = xt::amax(data, {1}, xt::keep_dims);
    return res;
}

Tensor Tensor::round(int decimals) const {
    Tensor res(shape);
    float factor = std::pow(10.0f, decimals);
    res.data = xt::round(data * factor) / factor;
    return res;
}

void Tensor::set(std::initializer_list<int> indices, float value){
    std::vector<size_t> idx;
    for(int i : indices){
        idx.push_back(static_cast<size_t>(i));
    }
    data.element(idx.begin(), idx.end()) = value;
}


float Tensor::moyenne() const {
    return xt::mean(data)();
}

Tensor Tensor::pow(float val) const {
    Tensor result(shape);
    result.data = xt::pow(data, val);
    return result;
}
int Tensor::size() const{
   return shape.size(); 
}