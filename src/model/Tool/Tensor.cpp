#include "model/Tool/TensorData/TensorDataCPU.h"
#include "model/Tool/TensorData/TensorDataBase.h"
#include "model/Tool/Tensor.h"

//tous les ops dans Tensor.cpp
Tensor::Tensor(DeviceType _device) : shape(), device(_device), _data(nullptr) {
    init_data_struct();
}

Tensor::Tensor(DeviceType _device, Shape _shape, bool alea, int val_init) : shape(_shape), device(_device), _data(nullptr){
    init_data_struct();
    _data->init(_shape,alea,val_init);
}

Tensor::Tensor(const Tensor& other){
    device = other.device;
    shape = other.shape;
    _data = (other._data != nullptr) ? other._data->clone() : nullptr;
}

Tensor::Tensor(Tensor&& other) noexcept{
    device = other.device;
    shape = other.shape;
    _data = other._data;
    other._data = nullptr;
}
Tensor::~Tensor(){
    delete _data;
}

//constructeur avec data:
Tensor::Tensor(DeviceType _device,const xt::xarray<float>& arr) : shape(), device(_device), _data(nullptr){
    init_data_struct();
    _data->init_with_data(arr);
    recalul_shape();
}








//methode qui change data tensor
Tensor& Tensor::exp() {
    _data->apply_exp();
    return *this;
}

Tensor& Tensor::pow(float val) {
    check_data();
    _data->apply_pow(val);
    return *this;
}

Tensor& Tensor::max(float val) {
    check_data();
    _data->apply_max(val);
    return *this;
}

Tensor& Tensor::round(int decimals) {
    check_data();
    _data->apply_round(decimals);
    return *this;
}

Tensor& Tensor::clip(float b_min,float b_max) {
    check_data();
    _data->apply_clip(b_min,b_max);
    return *this;
}

Tensor& Tensor::log() {
    check_data();
    _data->apply_log();
    return *this;
}

Tensor& Tensor::reshape(Shape format){
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
    _data->reshape(format);
    shape = format;
    return *this;
}

Tensor& Tensor::transpose(){
    check_data();
    _data->transpose();
    recalul_shape();
    return *this;
}







//methode qui créer de nouveau Tensor
Tensor Tensor::prod_mat(const Tensor& other) const {
    check_data();
    if(shape.len() != 2 || other.shape.len() != 2)
        Throw_Error("Produit matriciel 2D uniquement");

    if(shape[1] != other.shape[0]){
        Throw_Error("Produit matriciel impossible : Shapes incompatibles");
    }
    return _data->matmul(shape,other);
}









//methode personalisé pour trnasforamtion des couches
Tensor Tensor::sum_axis(std::size_t axis, bool keep_dims) const{
    Tensor res =_data->sum_axis(axis,keep_dims);
    return res;
}

Tensor Tensor::sum_per_row() const {
    Tensor res =_data->sum_per_row();
    return res;
}

Tensor Tensor::max_per_row() const {
    Tensor res =_data->max_per_row();
    return res;
}

std::vector<Tensor> Tensor::separation_batch(int batch_size) const{
    std::vector<Tensor> liste;
    if(batch_size <= -1)
        Throw_Error("batch_size invalide >= 0");

    if(batch_size == 0){
        liste.push_back(*this);
        return liste;
    }

    if(shape.len() == 0)
        Throw_Error("Tensor invalide (Aucune dimension).");

    int nbr_val_tensor = shape[0];
    if(nbr_val_tensor == 0)
        return liste;

    int nbr_de_batch = (nbr_val_tensor + batch_size - 1) / batch_size;
    liste.reserve(nbr_de_batch);
    for(int i = 0; i < nbr_val_tensor; i += batch_size){
        int debut = i;
        int fin = std::min(i + batch_size, nbr_val_tensor);
        liste.push_back(extraction_section_axe_0(debut, fin));
    }
    return liste;
}

Tensor Tensor::extraction_section_axe_0(int debut, int fin) const{
    check_data();
    return _data->extraction_section_axe_0(shape,debut,fin);
}







//methode evaluation
bool Tensor::scan_for_Nan(bool throww) const {
    check_data();
    return _data->scan_for_Nan(throww);
}

bool Tensor::is_cpu() const{
    return device == DeviceType::CPU;
}

bool Tensor::is_gpu() const{
    return !is_cpu();
}







//methode void
void Tensor::recalul_shape(){
    check_data();
    shape = _data->recalul_shape();
}

void Tensor::to_cpu(){
    if(device == DeviceType::CPU)
        return;

    //logique a faire de conv
    device = DeviceType::CPU;
}

void Tensor::to_gpu(){
    if(device == DeviceType::GPU)
        return;

    //logique a faire de conv plus tard
    device = DeviceType::GPU;
}

void Tensor::init_data_struct(){
    if(is_cpu()){
        _data = new TensorDataCPU();
        return;
    }
    if(is_gpu()){
        //_data = new TensorDataGPU();
        return;
    }
    Throw_Error("Initialisation Impossible Tensor");
}








// autre
float Tensor::moyenne() const {
    check_data();
    return _data->moyenne();
}


int Tensor::size() const{
   return shape.size(); 
}

DeviceType Tensor::get_device() const{
    return device;
}






//methode pour modifier un element par element
float Tensor::get(const std::vector<size_t>& indices) const{
    check_data();
    return _data->get(indices);
}

void Tensor::set(const std::vector<size_t>& indices, float val){
    check_data();
    _data->set(indices, val);
}






//private
//info lie a TensorDataBase
TensorDataBase* Tensor::get_data()const {
    return _data;
}

const xt::xarray<float>& Tensor::get_data_format_xr()const{
    check_data();
    return _data->get_format_xr();
}


void Tensor::set_data(TensorDataBase* data){
    _data = data;
}

void Tensor::check_data() const{
    if(_data == nullptr) Throw_Error("data de Tensor est nullptr");
}