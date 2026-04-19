#include "model/Tool/Tensor/TensorDataGPU.h"
#include "model/Tool/Tensor/Tensor.h"



// private
size_t TensorDataGPU::get_total_size() const{
    return shape.size();
}

const CudaData<float>& TensorDataGPU::get_data_gpu() const{
    return data_gpu;
}

CudaData<float>& TensorDataGPU::get_data_gpu(){
    return data_gpu;
}

xt::xarray<float> TensorDataGPU::copy_to_cpu() const{
    return data_gpu.copy_to_cpu(shape);
}

void TensorDataGPU::copy_from_cpu(const xt::xarray<float>& arr){
    shape.dims.assign(arr.shape().begin(), arr.shape().end());
    data_gpu.copy_from_cpu(arr);
}

int TensorDataGPU::calcul_stride(int dim_dep) const{
    int stride = 1;
    for(int i = dim_dep; i < shape.len(); i++)
        stride *= (int)shape[i];
    return stride;
}

bool TensorDataGPU::shape_broadcastable(const Shape& b_shape,int dim_dep){
    bool broadcastable = true;
    for(int i = dim_dep; i < shape.len(); i++){
        if(shape[i] != b_shape[i]){
            broadcastable = false;
            break;
        }
    }
    return broadcastable;
}






//public
TensorDataGPU::~TensorDataGPU(){
    data_gpu.free();
}

TensorDataBase* TensorDataGPU::clone() const{
    TensorDataGPU* res = new TensorDataGPU();
    res->shape = shape;
    res->data_gpu.copy_from_gpu(data_gpu);
    return res;
}

const TensorDataGPU* TensorDataGPU::check_gpu(const Tensor& a) const{
    if(!a.is_gpu())
        Throw_Error("TensorDataGPU: Tensor non gpu");
    auto da = dynamic_cast<const TensorDataGPU*>(a.get_data());
    if(!da)
        Throw_Error("TensorDataGPU: cast impossible");
    return da;
}

TensorDataGPU* TensorDataGPU::check_gpu_nc(Tensor& a) const{
    if(!a.is_gpu())
        Throw_Error("TensorDataGPU: Tensor non gpu");
    auto da = dynamic_cast<TensorDataGPU*>(a.get_data());
    if(!da)
        Throw_Error("TensorDataGPU: cast impossible");
    return da;
}










// les ops
void TensorDataGPU::apply_add(const Tensor& b){
    auto db = check_gpu(b);
    Shape b_shape = b.get_shape();
    if(shape == b_shape){
        gpu_add(data_gpu.data(),db->get_data_gpu().data(),get_total_size());
        return;
    }
    if(shape.len() == b_shape.len() && b_shape[0] == 1){
        if(shape_broadcastable(b_shape,1)){
            gpu_add_broadcast_axis0(data_gpu.data(),db->get_data_gpu().data(),(int)shape[0],calcul_stride(1));
            return;
        }
    }
    Throw_Error("Shape non pris en compte, TensorDataGPU add impossble, actuellement");
}

void TensorDataGPU::apply_sub(const Tensor& b){
    auto db = check_gpu(b);
    Shape b_shape = b.get_shape();
    if(shape == b_shape){
        gpu_sub(data_gpu.data(),db->get_data_gpu().data(),get_total_size());
        return;
    }
    if(shape.len() == b_shape.len() && b_shape[0] == 1){
        if(shape_broadcastable(b_shape,1)){
            gpu_sub_broadcast_axis0(data_gpu.data(),db->get_data_gpu().data(),(int)shape[0],calcul_stride(1));
            return;
        }
    }
    Throw_Error("Shape non pris en compte, TensorDataGPU sub impossble, actuellement");
}

void TensorDataGPU::apply_mul(const Tensor& b){
    auto db = check_gpu(b);
    Shape b_shape = b.get_shape();
    if(shape == b_shape){
        gpu_mul(data_gpu.data(),db->get_data_gpu().data(),get_total_size());
        return;
    }
    if(shape.len() == b_shape.len() && b_shape[0] == 1){
        if(shape_broadcastable(b_shape,1)){
            gpu_mul_broadcast_axis0(data_gpu.data(),db->get_data_gpu().data(),(int)shape[0],calcul_stride(1));
            return;
        }
    }
    Throw_Error("Shape non pris en compte, TensorDataGPU mul impossble, actuellement");
}

void TensorDataGPU::apply_div(const Tensor& b){
    auto db = check_gpu(b);
    Shape b_shape = b.get_shape();
    if(shape == b_shape){
        gpu_div(data_gpu.data(),db->get_data_gpu().data(),get_total_size());
        return;
    }
    if(shape.len() == b_shape.len() && b_shape[0] == 1){
        if(shape_broadcastable(b_shape,1)){
            gpu_div_broadcast_axis0(data_gpu.data(),db->get_data_gpu().data(),(int)shape[0],calcul_stride(1));
            return;
        }
    }
    Throw_Error("Shape non pris en compte, TensorDataGPU div impossble, actuellement");
}

void TensorDataGPU::apply_add(float scalar){
    gpu_add_scalar(data_gpu.data(),scalar,get_total_size());
}

void TensorDataGPU::apply_sub(float scalar){
    gpu_sub_scalar(data_gpu.data(),scalar,get_total_size());
}

void TensorDataGPU::apply_mul(float scalar){
    gpu_mul_scalar(data_gpu.data(),scalar,get_total_size());
}

void TensorDataGPU::apply_div(float scalar){
    if(scalar == 0){
        Throw_Error("Division TensorDataGPU impossible scalar == 0.");
    }
    gpu_div_scalar(data_gpu.data(),scalar,get_total_size());
}










// methode qui modifie nos data Tensor
void TensorDataGPU::init(Shape _shape, bool alea, int val_init){
    shape = _shape;
    data_gpu.allocate(get_total_size());
    if(alea){
        data_gpu.fill_random();
    }else if(val_init != 0){
        data_gpu.fill_value(val_init);
    }else{
        data_gpu.fill_zero();
    }
}

void TensorDataGPU::init_with_data(const xt::xarray<float>& arr){
    copy_from_cpu(arr);
}

void TensorDataGPU::apply_exp(){
    gpu_exp(data_gpu.data(),get_total_size());
}

void TensorDataGPU::apply_pow(float val){
    gpu_pow(data_gpu.data(),val,get_total_size());
}

void TensorDataGPU::apply_max(float val){
    gpu_max(data_gpu.data(),val,get_total_size());
}

void TensorDataGPU::apply_round(int decimals){
    gpu_round(data_gpu.data(),decimals,get_total_size());
}

void TensorDataGPU::apply_clip(float b_min, float b_max){
    gpu_clip(data_gpu.data(),b_min,b_max,get_total_size());
}

void TensorDataGPU::apply_log(){
    gpu_log(data_gpu.data(),get_total_size());
}

void TensorDataGPU::calcul_sup(float scalar){
    gpu_sup(data_gpu.data(),scalar,get_total_size());
}

void TensorDataGPU::transpose(bool batch){
    if(shape.len() == 1){
        shape.dims = {shape.dims[0], 1};
        return;
    }

    if(shape.len() == 2){
        CudaData<float> tmp;
        tmp.allocate(get_total_size());
        gpu_transpose(tmp.data(), data_gpu.data(), shape.dims[0], shape.dims[1], 1);
        data_gpu = std::move(tmp);
        std::swap(shape.dims[0], shape.dims[1]);
        return;
    }

    if(shape.len() == 3 && batch){
        CudaData<float> tmp;
        tmp.allocate(get_total_size());
        gpu_transpose(tmp.data(), data_gpu.data(), shape.dims[1], shape.dims[2], shape.dims[0]);
        data_gpu = std::move(tmp);
        std::swap(shape.dims[1], shape.dims[2]);
        return;
    }

    Throw_Error("Dimension invalide pour transposeGPU ", shape.print());
}

void TensorDataGPU::reshape(Shape format){
    if(format.size() != shape.size()){
        Throw_Error("reshape GPU impossible, taille totale differente");
    }
    shape = format;
}

void TensorDataGPU::shuffle(const std::vector<int>& indices){
    if(shape.len() == 0)
        Throw_Error("shuffle impossible : tensor vide");

    int axis0_size = (int)shape[0];

    if((int)indices.size() != axis0_size)
        Throw_Error("shuffle GPU : taille indices invalide");

    int stride = calcul_stride(1);

    CudaData<int> d_indices;
    d_indices.copy_from_cpu(indices);

    CudaData tmp;
    tmp.allocate(get_total_size());

    gpu_shuffle_axis0(tmp.data(), data_gpu.data(), d_indices.data(), axis0_size, stride);
    data_gpu = std::move(tmp);
}

Tensor TensorDataGPU::matmul(const Tensor& b) const{
    auto db = check_gpu(b);
    Shape b_shape = b.get_shape();

    if(shape.len() != 2 && shape.len() != 3){
        Throw_Error("matmul GPU : le tensor de gauche doit etre 2D");
    }

    int rows = shape[0];
    int trans = shape[1];

    const float* A = data_gpu.data();
    const float* B = db->get_data_gpu().data();

    if(shape.len() == 2 && b_shape.len() == 2){
        int cols = b_shape[1];
        if(trans != (int)b_shape[0]){
            Throw_Error("matmul GPU impossible : dimensions incompatibles");
        }
        Tensor result(DeviceType::GPU, Shape({(size_t)rows, (size_t)cols}));
        auto dres = check_gpu_nc(result);
        float* C = dres->get_data_gpu().data();
        gpu_matmul(C, A, B, rows, trans, cols);
        return result;
    }

    if(shape.len() == 2 && b_shape.len() == 3){
        int batch = b_shape[0];
        int b_trans = b_shape[1];
        int cols = b_shape[2];
        if(trans != b_trans){
            Throw_Error("matmul GPU broadcast impossible : dimensions invalide");
        }
        Tensor result(DeviceType::GPU, Shape({(size_t)batch, (size_t)rows, (size_t)cols}));
        auto dres = check_gpu_nc(result);
        gpu_broadcast_matmul(dres->get_data_gpu().data(), A, B, batch, rows, trans, cols, false);
        return result;
    }

    if(shape.len() == 3 && b_shape.len() == 2){
        int batch = shape[0];
        int a_rows = shape[1];
        int a_trans = shape[2];
        int cols = b_shape[1];
        if(a_trans != (int)b_shape[0]){
            Throw_Error("matmul GPU broadcast impossible : dimensions invalide");
        }
        Tensor result(DeviceType::GPU, Shape({(size_t)batch, (size_t)a_rows, (size_t)cols}));
        auto dres = check_gpu_nc(result);
        gpu_broadcast_matmul(dres->get_data_gpu().data(), A, B, batch, a_rows, a_trans, cols, true);
        return result;
    }

    if(shape.len() == 3 && b_shape.len() == 3){
        int batch_a = shape[0];
        int a_rows  = shape[1];
        int a_trans = shape[2];

        int batch_b = b_shape[0];
        int b_trans = b_shape[1];
        int cols    = b_shape[2];

        if(batch_a != batch_b){
            Throw_Error("matmul GPU batch impossible : batch differents");
        }
        if(a_trans != b_trans){
            Throw_Error("matmul GPU batch impossible : dimensions incompatibles");
        }

        Tensor result(DeviceType::GPU, Shape({(size_t)batch_a, (size_t)a_rows, (size_t)cols}));
        auto dres = check_gpu_nc(result);

        gpu_broadcast_all_matmul(dres->get_data_gpu().data(),A, B,batch_a, a_rows, a_trans, cols);

        return result;
    }

    Throw_Error("matmul GPU : shape non valide, pour l'instant");
    return Tensor();
}

Tensor TensorDataGPU::sum_axis(std::size_t axis, bool keep_dims) const{
    if(shape.len() != 2)
        Throw_Error("sum_axis GPU nécessite un tensor 2D");

    if(axis > 1)
        Throw_Error("sum_axis GPU axis invalide pour tensor 2D");

    if(axis == 1){
        Tensor tmp = sum_per_row();
        if(keep_dims)
            return tmp;

        return tmp.reshape(Shape({shape[0]}));
    }
    
    Tensor res;
    if(keep_dims){
        res = Tensor(DeviceType::GPU, Shape({1, shape[1]}));
    }else{
        res = Tensor(DeviceType::GPU, Shape({shape[1]}));
    }
    auto dres = check_gpu_nc(res);

    gpu_sum_axis0(dres->get_data_gpu().data(),data_gpu.data(),(int)shape[0],(int)shape[1]);
    return res;
}

Tensor TensorDataGPU::sum_per_row() const{
    if(shape.len() != 2)
        Throw_Error("sum_per_row GPU nécessite un tensor 2D");

    Tensor res(DeviceType::GPU, Shape({(size_t)shape[0], 1}));
    auto dres = check_gpu_nc(res);

    gpu_sum_per_row(dres->get_data_gpu().data(),data_gpu.data(),(int)shape[0],(int)shape[1]);
    return res;
}

Tensor TensorDataGPU::max_per_row() const{
    if(shape.len() != 2)
        Throw_Error("max_per_row GPU nécessite un tensor 2D");

    Tensor res(DeviceType::GPU, Shape({shape[0], 1}));
    auto dres = check_gpu_nc(res);

    gpu_max_per_row(dres->get_data_gpu().data(),data_gpu.data(),(int)shape[0],(int)shape[1]);
    return res;
}

Tensor TensorDataGPU::extraction_section_axe_0(int debut, int fin) const{
    if(shape.len() == 0)
        Throw_Error("Tensor vide");

    int nbr_val_tensor = (int)shape[0];
    if(debut < 0 || fin < 0 || debut > fin || fin > nbr_val_tensor)
        Throw_Error("Indices invalides dans extraction_section_axe_0");

    int new_axis0 = fin - debut;

    Shape new_shape = shape;
    new_shape[0] = new_axis0;

    Tensor res(DeviceType::GPU, new_shape);
    auto dres = check_gpu_nc(res);

    int stride = 1;
    for(int i = 1; i < shape.len(); i++){ // on calcule le nbr d'element par ligne sauf la dim 0 car batch
        stride *= (int)shape[i];
    }
    gpu_extraction_section_axe_0(dres->get_data_gpu().data(),data_gpu.data(),debut,fin,stride);
    return res;
}

float TensorDataGPU::moyenne() const{ // methode pecifique qui fusione mais uniquemen avec mat 2D
    int total = (int)get_total_size();

    Tensor sum_rows = sum_per_row();              
    Tensor total_sum = sum_rows.sum_axis(0, false);
    if(total_sum.get_shape().len() >1){
        Throw_Error("moyenne gpu shape invlaide Maj necessaire :",shape.print());
    }
    float val = total_sum.get_data()->get({0});
    return val / total;
}


const xt::xarray<float> TensorDataGPU::to_json() const{
    return copy_to_cpu();
}






// methode pour modifier un element par element
float TensorDataGPU::get(const std::vector<size_t>& indices) const{
    if(indices.size() != (size_t)shape.len())
        Throw_Error("TensorDataGPU::get indices invalides");
    size_t index = 0;
    size_t decal = 1;
    for(int i = (int)shape.len() - 1; i >= 0; i--){
        if(indices[i] >= shape[i])
            Throw_Error("TensorDataGPU::get index hors borne");
        index += indices[i] * decal;
        decal *= shape[i];
    }
    return data_gpu.get(index);
}

void TensorDataGPU::set(const std::vector<size_t>& indices, float val){
    if(indices.size() != (size_t)shape.len())
        Throw_Error("TensorDataGPU::get indices invalides");
    size_t index = 0;
    size_t decal = 1;
    for(int i = shape.len() - 1; i >= 0; i--){
        if(indices[i] >= shape[i])
            Throw_Error("TensorDataGPU::get index hors borne");
        index += indices[i] * decal;
        decal *= shape[i];
    }
    return data_gpu.set(index,val);
}








