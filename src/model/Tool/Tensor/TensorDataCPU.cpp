#include "model/Tool/Tensor/TensorDataCPU.h"
#include "model/Tool/Tensor/Tensor.h"

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
void TensorDataCPU::init(Shape _shape, bool alea, int val_init){
    if(!alea && val_init != 0){
        data_cpu = xt::ones<float>(_shape.dims)*val_init;
    }else{
        data_cpu = xt::zeros<float>(_shape.dims);
    }
    if(alea)fill_alea();
    shape=_shape;
}

void TensorDataCPU::init_with_data(const xt::xarray<float>& arr) {
    data_cpu=arr;
    recalul_shape();
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
    xt::xarray<float> tmp = xt::transpose(data_cpu);
    data_cpu = tmp;
    recalul_shape();
}

void TensorDataCPU::reshape(Shape format) {
    size_t size_old = 1;
    for(size_t i = 0; i < (size_t)shape.len(); i++){
        size_old *= shape[i];
    }

    size_t size_new = 1;
    for(size_t i = 0; i < (size_t)format.len(); i++){
        size_new *= format[i];
    }

    if(size_old != size_new){
        Throw_Error("Tensor ", shape.print(), " reshape impossible vers ", format.print());
    }
    data_cpu.reshape(format.dims);
    shape=format;
}

void TensorDataCPU::shuffle(const std::vector<int>& indices){
    xt::xarray<float> tmp = data_cpu;
    for(int i = 0; i < (int)indices.size(); i++){
        xt::view(data_cpu, i) = xt::view(tmp, indices[i]);
    }
}







//methode qui créer un nouveau Tensor
Tensor TensorDataCPU::matmul(const Tensor& b) const {// toujours facon naive sur CPU
    check_cpu(b);
    Shape shape_b = b.get_shape();

    if(shape.len() != 2 || shape_b.len() != 2){
        Print("shape a",shape.print());
        Print("shape b",shape_b.print());
        Throw_Error("Produit matriciel 2D uniquement");
    }

    if(shape[1] != shape_b[0]){
        Throw_Error("Produit matriciel impossible : Shapes incompatibles");
    }

    Tensor res(DeviceType::CPU,Shape({shape[0], shape_b[1]}));
    for(size_t i = 0; i < shape[0]; i++){
        for(size_t j = 0; j < shape_b[1]; j++){
            float sum = 0.f;
            for(size_t k = 0; k < shape[1]; k++){
                sum += data_cpu(i,k) * b.get({k,j});
            }
            res.set({i,j},sum);
        }
    }
    auto dres = dynamic_cast<TensorDataCPU*>(res.get_data());
    dres->recalul_shape();
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
    auto dres = dynamic_cast<TensorDataCPU*>(res.get_data());
    dres->recalul_shape();
    return res;
}

Tensor TensorDataCPU::sum_per_row() const {
    Tensor res(DeviceType::CPU);
    res.set_data(new TensorDataCPU(xt::sum(data_cpu, {1}, xt::keep_dims)));
    auto dres = dynamic_cast<TensorDataCPU*>(res.get_data());
    dres->recalul_shape();
    return res;
}

Tensor TensorDataCPU::max_per_row() const {
    Tensor res(DeviceType::CPU);
    res.set_data(new TensorDataCPU(xt::amax(data_cpu, {1}, xt::keep_dims)));
    auto dres = dynamic_cast<TensorDataCPU*>(res.get_data());
    dres->recalul_shape();
    return res;
}

Tensor TensorDataCPU::extraction_section_axe_0(int debut, int fin) const{
    if(shape.len() == 0)
        Throw_Error("Tensor vide");
    int nbr_val_tensor = shape[0];
    if(debut < 0 || fin < 0 || debut > fin || fin > nbr_val_tensor)
        Throw_Error("Indices invalides dans extraction_section_axe_0");

    xt::xstrided_slice_vector slices;
    slices.push_back(xt::range(debut, fin));
    for(int d = 1; d < shape.len(); d++)
        slices.push_back(xt::all());

    auto vue = xt::strided_view(data_cpu, slices);
    xt::xarray<float> tmp = vue;
    Tensor res(DeviceType::CPU, tmp);
    auto dres = dynamic_cast<TensorDataCPU*>(res.get_data());
    dres->recalul_shape();
    return res;
}







//methode autre
void TensorDataCPU::recalul_shape() {
    std::vector<size_t> shape_tmp;
    for(auto s : data_cpu.shape()){
        shape_tmp.push_back(s);
    }
    shape.dims = shape_tmp;
}

float TensorDataCPU::moyenne() const{
    return xt::mean(data_cpu)();
}

const xt::xarray<float> TensorDataCPU::to_json() const{
    return data_cpu;
}










//methode pour modifier un element par element
float TensorDataCPU::get(const std::vector<size_t>& indices) const{
    return data_cpu.element(indices.begin(), indices.end());
}

void TensorDataCPU::set(const std::vector<size_t>& indices, float val){
    data_cpu.element(indices.begin(), indices.end()) = val;
}