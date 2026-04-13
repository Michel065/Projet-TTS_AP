#include "model/Tool/Tensor/TensorDataGPU.h"
#include "model/Tool/Tensor/Tensor.h"

TensorDataGPU::~TensorDataGPU() = default;

const TensorDataGPU* TensorDataGPU::check_gpu(const Tensor& a) const{
    if(!a.is_gpu())
        Throw_Error("TensorDataGPU: Tensor non gpu");
    auto da = dynamic_cast<const TensorDataGPU*>(a.get_data());
    if(!da)
        Throw_Error("TensorDataGPU: cast impossible");
    return da;
}


TensorDataBase* TensorDataGPU::clone() const{
    return nullptr;
}


// les ops
void TensorDataGPU::apply_add(const Tensor& b){
    auto db = check_gpu(b);
    if(shape.dims != b.get_shape().dims){
        Throw_Error("Shape differente, TensorDataGPU add impossble, actuellement");
    }
    gpu_add(data_gpu.data(),db->get_data_gpu().data(),get_total_size());
}

void TensorDataGPU::apply_sub(const Tensor& b){
    auto db = check_gpu(b);
    if(shape.dims != b.get_shape().dims){
        Throw_Error("Shape differente, TensorDataGPU sub impossble, actuellement");
    }
    gpu_sub(data_gpu.data(),db->get_data_gpu().data(),get_total_size());
}

void TensorDataGPU::apply_mul(const Tensor& b){
    auto db = check_gpu(b);
    if(shape.dims != b.get_shape().dims){
        Throw_Error("Shape differente, TensorDataGPU mul impossble, actuellement");
    }
    gpu_mul(data_gpu.data(),db->get_data_gpu().data(),get_total_size());
}

void TensorDataGPU::apply_div(const Tensor& b){
    auto db = check_gpu(b);
    if(shape.dims != b.get_shape().dims){
        Throw_Error("Shape differente, TensorDataGPU div impossble, actuellement");
    }
    gpu_div(data_gpu.data(),db->get_data_gpu().data(),get_total_size());
}

void TensorDataGPU::apply_add(float scalar){
    gpu_add(data_gpu.data(),scalar,get_total_size());
}

void TensorDataGPU::apply_sub(float scalar){
    gpu_sub(data_gpu.data(),scalar,get_total_size());
}

void TensorDataGPU::apply_mul(float scalar){
    gpu_mul(data_gpu.data(),scalar,get_total_size());
}

void TensorDataGPU::apply_div(float scalar){
    if(scalar == 0){
        Throw_Error("Division TensorDataGPU impossible scalar == 0.");
    }
    gpu_div(data_gpu.data(),scalar,get_total_size());
}







// methode qui modifie nos data Tensor
void TensorDataGPU::init(Shape _shape, bool alea, int val_init){
    shape = _shape;
    xt::xarray<float> tmp;
    if(!alea && val_init > 0){
        tmp = xt::ones<float>(_shape.dims) * val_init;
    }else{
        tmp = xt::zeros<float>(_shape.dims);
    }
    data_gpu.copy_from_cpu(tmp);
    if(alea)
        fill_alea();
}

void TensorDataGPU::init_with_data(const xt::xarray<float>& arr){
    copy_from_cpu(arr);
}

void TensorDataGPU::fill_alea(){
    xt::xarray<float> tmp = xt::zeros<float>(shape.dims);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    for(auto& v : tmp.storage()){
        v = dist(gen);
    }

    data_gpu.copy_from_cpu(tmp);
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

const xt::xarray<float> TensorDataGPU::to_json() const{
    return copy_to_cpu();
}






// methode pour modifier un element par element
float TensorDataGPU::get(const std::vector<size_t>& indices) const{
    return 0.0f;
}

void TensorDataGPU::set(const std::vector<size_t>& indices, float val){
}










// private
size_t TensorDataGPU::get_total_size() const{
    return shape.size();
}

const CudaData& TensorDataGPU::get_data_gpu() const{
    return data_gpu;
}

xt::xarray<float> TensorDataGPU::copy_to_cpu() const{
    return data_gpu.copy_to_cpu(shape);
}

void TensorDataGPU::copy_from_cpu(const xt::xarray<float>& arr){
    shape.dims.assign(arr.shape().begin(), arr.shape().end());
    data_gpu.copy_from_cpu(arr);
}
