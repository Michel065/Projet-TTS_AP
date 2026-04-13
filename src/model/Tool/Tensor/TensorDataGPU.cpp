#include "model/Tool/Tensor/TensorDataGPU.h"
#include "model/Tool/Tensor/Tensor.h"

TensorDataGPU::~TensorDataGPU() = default;

const TensorDataGPU* TensorDataGPU::check_gpu(const Tensor& a) const{
    if(!a.is_gpu())
        Throw_Error("TensorDataGPU: Tensor non cpu");
    auto da = dynamic_cast<TensorDataGPU*>(a.get_data());
    if(!da)
        Throw_Error("TensorDataGPU: cast impossible");
    return da;
}


TensorDataBase* TensorDataGPU::clone() const{
    return nullptr;
}

void TensorDataGPU::apply_add(const Tensor& b){
}

void TensorDataGPU::apply_sub(const Tensor& b){
}

void TensorDataGPU::apply_mul(const Tensor& b){
}

void TensorDataGPU::apply_div(const Tensor& b){
}

void TensorDataGPU::apply_add(float scalar){
}

void TensorDataGPU::apply_sub(float scalar){
}

void TensorDataGPU::apply_mul(float scalar){
}

void TensorDataGPU::apply_div(float scalar){
}

void TensorDataGPU::init(Shape _shape, bool alea, int val_init){
}

void TensorDataGPU::init_with_data(const xt::xarray<float>& arr){
}

void TensorDataGPU::fill_alea(){
}

void TensorDataGPU::apply_exp(){
}

void TensorDataGPU::apply_pow(float val){
}

void TensorDataGPU::apply_max(float val){
}

void TensorDataGPU::apply_round(int decimals){
}

void TensorDataGPU::apply_clip(float b_min, float b_max){
}

void TensorDataGPU::apply_log(){
}

void TensorDataGPU::calcul_sup(float scalar){
}

void TensorDataGPU::transpose(){
}

void TensorDataGPU::reshape(Shape format){
}

Tensor TensorDataGPU::matmul(const Tensor& b) const{
    return Tensor();
}

Tensor TensorDataGPU::sum_axis(std::size_t axis, bool keep_dims) const{
    return Tensor();
}

Tensor TensorDataGPU::sum_per_row() const{
    return Tensor();
}

Tensor TensorDataGPU::max_per_row() const{
    return Tensor();
}

Tensor TensorDataGPU::extraction_section_axe_0(int debut, int fin) const{
    return Tensor();
}

bool TensorDataGPU::equal(const Tensor& b) const{
    return false;
}

void TensorDataGPU::recalul_shape(){
}

float TensorDataGPU::moyenne() const{
    return 0.0f;
}

bool TensorDataGPU::scan_for_Nan(bool throww) const{
    return false;
}

const xt::xarray<float>& TensorDataGPU::get_format_xr() const{
    static xt::xarray<float> dummy;
    return dummy;
}

std::ostream& operator<<(std::ostream& os, const TensorDataGPU* t){
    os << "TensorDataGPU";
    return os;
}

float TensorDataGPU::get(const std::vector<size_t>& indices) const{
    return 0.0f;
}

void TensorDataGPU::set(const std::vector<size_t>& indices, float val){
}

size_t TensorDataGPU::get_total_size() const{
    return 0;
}

void TensorDataGPU::alloc_gpu(size_t n){
}

void TensorDataGPU::free_gpu(){
}

xt::xarray<float> TensorDataGPU::copy_to_cpu() const{
    return xt::xarray<float>();
}

void TensorDataGPU::copy_from_cpu(const xt::xarray<float>& arr){
}