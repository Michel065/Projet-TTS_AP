#include "model/Tool/TensorData/TensorDataCPU.h"
#include "model/Tool/Tensor.h"

TensorDataBase* TensorDataCPU::clone() const{
    return new TensorDataCPU(*this);
}

const TensorDataCPU* TensorDataCPU::check_cpu(const Tensor& a) const {
    if(!a.is_cpu())
        Throw_Error("TensorDataCPU: Tensor non cpu");
    auto da = dynamic_cast<TensorDataCPU*>(a.get_data());
    if(!da)
        Throw_Error("TensorDataCPU: cast impossible");
    return da;
}







//les ops
void TensorDataCPU::apply_add(const Tensor& b) {    
    auto db = check_cpu(b);
    data_cpu += db->data_cpu;
}

void TensorDataCPU::apply_sub(const Tensor& b) {    
    auto db = check_cpu(b);
    data_cpu -= db->data_cpu;
}

void TensorDataCPU::apply_mul(const Tensor& b) {    
    auto db = check_cpu(b);
    data_cpu *= db->data_cpu;
}

void TensorDataCPU::apply_div(const Tensor& b) {    
    auto db = check_cpu(b);
    data_cpu /= db->data_cpu;
}

void TensorDataCPU::apply_add(float scalar) {
    data_cpu += scalar;
}

void TensorDataCPU::apply_sub(float scalar) {
    data_cpu -= scalar;
}

void TensorDataCPU::apply_mul(float scalar) {
    data_cpu *= scalar;
}

void TensorDataCPU::apply_div(float scalar) {
    data_cpu /= scalar;
}








// methode qui modifie datacpu
void TensorDataCPU::init(Shape shape, bool alea, int val_init){
    if(!alea && val_init>0){
        data_cpu = xt::ones<float>(shape.dims)*val_init;
    }else{
        data_cpu = xt::zeros<float>(shape.dims);
    }
    if(alea)fill_alea();
}

void TensorDataCPU::init_with_data(const xt::xarray<float>& arr) {
    data_cpu=arr;
}

void TensorDataCPU::fill_alea(){
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for(auto& v : data_cpu.storage()){
        v = dist(gen);
    }
}

void TensorDataCPU::apply_exp(){
    data_cpu = xt::exp(data_cpu);
}

void TensorDataCPU::apply_pow(float val){
    data_cpu = xt::pow(data_cpu, val);
}

void TensorDataCPU::apply_max(float val){
    data_cpu = xt::maximum(data_cpu,val);
}

void TensorDataCPU::apply_round(int decimals){
    float factor = std::pow(10.0f, decimals);
    data_cpu = xt::round(data_cpu * factor) / factor;
}

void TensorDataCPU::apply_clip(float b_min, float b_max){
    data_cpu = xt::clip(data_cpu, b_min, b_max);
}

void TensorDataCPU::apply_log(){
    data_cpu = xt::log(data_cpu);
}   

void TensorDataCPU::calcul_sup(float scalar) {
    data_cpu = (data_cpu > scalar);
}

void TensorDataCPU::transpose() {
    data_cpu = xt::eval(xt::transpose(data_cpu));
}

void TensorDataCPU::reshape(Shape format) {
    data_cpu.reshape(format.dims);
}







//methode qui créer un nouveau Tensor
Tensor TensorDataCPU::matmul(Shape shape_a,const Tensor& b) const {// toujours facon naive sur CPU
    check_cpu(b);
    Tensor res(DeviceType::CPU,Shape({shape_a[0], b.shape[1]}));
    for(size_t i = 0; i < shape_a[0]; i++){
        for(size_t j = 0; j < b.shape[1]; j++){
            float sum = 0.f;
            for(size_t k = 0; k < shape_a[1]; k++){
                sum += data_cpu(i,k) * b.get({k,j});
            }
            res.set({i,j},sum);
        }
    }
    res.recalul_shape();
    return res;
}

Tensor TensorDataCPU::sum_axis(std::size_t axis, bool keep_dims) const {
    Tensor res(DeviceType::CPU);
    auto tmp = xt::sum(data_cpu, {axis});
    if(keep_dims){
        res.set_data(new TensorDataCPU(xt::expand_dims(tmp, axis)));
    }else{
        res.set_data(new TensorDataCPU(tmp));
    }
    res.recalul_shape();
    return res;
}

Tensor TensorDataCPU::sum_per_row() const {
    Tensor res(DeviceType::CPU);
    res.set_data(new TensorDataCPU(xt::sum(data_cpu, {1}, xt::keep_dims)));
    res.recalul_shape();
    return res;
}

Tensor TensorDataCPU::max_per_row() const {
    Tensor res(DeviceType::CPU);
    res.set_data(new TensorDataCPU(xt::amax(data_cpu, {1}, xt::keep_dims)));
    res.recalul_shape();
    return res;
}

Tensor TensorDataCPU::extraction_section_axe_0(Shape shape_a, int debut, int fin) const{
    if(shape_a.len() == 0)
        Throw_Error("Tensor vide");
    int nbr_val_tensor = shape_a[0];
    if(debut < 0 || fin < 0 || debut > fin || fin > nbr_val_tensor)
        Throw_Error("Indices invalides dans extraction_section_axe_0");

    xt::xstrided_slice_vector slices;
    slices.push_back(xt::range(debut, fin));
    for(int d = 1; d < shape_a.len(); d++)
        slices.push_back(xt::all());

    Tensor res(DeviceType::CPU);
    res.set_data(new TensorDataCPU(xt::eval(xt::strided_view(data_cpu, slices))));
    res.recalul_shape();
    return res;
}







//methode autre
bool TensorDataCPU::equal(const Tensor& b) const {
    auto db = check_cpu(b);
    return xt::all(xt::equal(data_cpu, db->data_cpu));
}

std::vector<size_t> TensorDataCPU::recalul_shape() const {
    std::vector<size_t> shape;
    for(auto s : data_cpu.shape()){
        shape.push_back(s);
    }
    return shape;
}

float TensorDataCPU::moyenne() const{
    return xt::mean(data_cpu)();
}

bool TensorDataCPU::scan_for_Nan(bool throww) const {
    if (xt::any(xt::isnan(data_cpu))) {
        if(throww){
            Print("");
            Throw_Error("TensorDataCPU a un Nan dans ca valeur");
        }
        return true;
    }
    return false;
}

const xt::xarray<float>& TensorDataCPU::get_format_xr() const {
    return data_cpu;
}

std::ostream& operator<<(std::ostream& os, const TensorDataCPU* t) {
    os << t->data_cpu;
    return os;
}










//methode pour modifier un element par element
float TensorDataCPU::get(const std::vector<size_t>& indices) const{
    return data_cpu.element(indices.begin(), indices.end());
}

void TensorDataCPU::set(const std::vector<size_t>& indices, float val){
    data_cpu.element(indices.begin(), indices.end()) = val;
}