#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xjson.hpp>

#include <random>
#include <vector>
#include <stdexcept>

#include "model/Tool/Shape.h"
#include "outil/Print.h"
#include "model/Json/Json_gestion.h"

//modifcation pour integration de CPU GPUs
#include "model/Tool/Tensor/TensorDataGPU.h"

class TensorDataBase;
class TensorDataCPU;
class TensorDataGPU;

enum class DeviceType {
    CPU,
    GPU
};

class Tensor {
public:
    //def dans Tensor.cpp
    Tensor(DeviceType _device = DeviceType::CPU);
    Tensor(DeviceType _device, Shape _shape, bool alea=false,int val_init=0);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    ~Tensor();

    //constructeur avec data:
    Tensor(DeviceType _device,const xt::xarray<float>& arr);

    //tous les ops dans TensorOps.cpp
    //utile:
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    //genere des tensors operator
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;
    friend Tensor operator+(float scalar, const Tensor& t);
    friend Tensor operator-(float scalar, const Tensor& t);
    friend Tensor operator*(float scalar, const Tensor& t);
    friend Tensor operator/(float scalar, const Tensor& t);
    //modifie celui actuel operator
    Tensor& operator+=(float scalar);
    Tensor& operator-=(float scalar);
    Tensor& operator*=(float scalar);
    Tensor& operator/=(float scalar);
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    //comparatif operator
    Tensor operator>(float scalar) const;
    void operator>=(float scalar);

    //methode utile
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
    friend void to_json(json& j, const Tensor& tensor);
    friend void from_json(const json& j, Tensor& tensor);
    
    //def dans Tensor.cpp
    //methode qui change data tensor
    Tensor& exp();
    Tensor& pow(float val);
    Tensor& max(float val=0.0f);
    Tensor& round(int decimals);
    Tensor& clip(float b_min,float b_max);
    Tensor& log();
    Tensor& reshape(Shape format);
    Tensor& transpose(bool batch = false);//pour la propagation

    //methode qui créer de nouveau Tensor
    Tensor prod_mat(const Tensor& other) const;

    //methode personalisé pour trnasforamtion des couches
    Tensor sum_axis(std::size_t axis, bool keep_dims) const;
    Tensor sum_per_row() const;
    Tensor max_per_row() const;
    void shuffle(const std::vector<int> &indices);
    std::vector<Tensor> separation_batch(int batch_size)const;
    Tensor extraction_section_axe_0(int debut, int fin) const;

    //methode evaluation
    bool is_cpu() const;
    bool is_gpu() const;

    //methode void
    void to_cpu();//methode pour faire la distinction
    void to_gpu();
    void init_data_struct();

    // autre
    float moyenne() const;
    int size() const;
    DeviceType get_device() const;
    Shape get_shape() const;

    //methode pour modifier un element par element
    float get(const std::vector<size_t>& indices) const;
    void set(const std::vector<size_t>& indices, float val);


    TensorDataBase* get_data()const; // pas propre provisoire
private:
    DeviceType device = DeviceType::CPU;
    
    //info lie a TensorDataBase
    TensorDataBase* _data=nullptr;
    
    void set_data(TensorDataBase* data);
    void check_data() const;
    
    // ajout des class en friend
    friend class TensorDataBase;
    friend class TensorDataCPU;
    friend class TensorDataGPU;
};



inline void from_json(const json& j, Tensor& tensor){
    tensor = Tensor(j.at("device_type").get<DeviceType>(), j.at("data").get<xt::xarray<float>>());
}

inline void to_json(json& j, const Tensor& tensor){
    j = json{
        {"device_type", tensor.get_device()},
        {"data", tensor.get_data()->to_json()}
    };
}


//methode equivaelente a ce qui est deja dans mais c plus simple une de base a l'exterieur de tensor pour que ce oit plus simple et léve une erreur en ca de non valididité, juste donc une methode pour check rapide et raise. 
inline void check_is_gpu(const Tensor& tens){ // pratique mais pas propre, a changer si j'ai le temps
    if(!tens.is_gpu()){
        Throw_Error("Tensor non GPU, utilisation de Cuda impossible");
    }
}

inline float* check_and_get_if_is_gpu(const Tensor& tens){ // pratique mais pas propre, a changer si j'ai le temps
    if(!tens.is_gpu()){
        Throw_Error("Tensor non GPU, utilisation de Cuda impossible");
    }
    TensorDataGPU* data = dynamic_cast<TensorDataGPU*>(tens.get_data());
    float* data_float = data->get_data_gpu().data();
    return data_float;
}


inline void debug_check_tensor_non_vide_rec(Tensor& tensor,const Shape& s,std::vector<size_t>& indices,size_t dim,int& nbr){
    if(dim == (size_t)s.len()){
        float val = tensor.get(indices);
        if(val != 0.0f){
            nbr++;
        }
        return;
    }

    for(size_t i = 0; i < s[dim]; i++){
        indices[dim] = i;
        debug_check_tensor_non_vide_rec(tensor, s, indices, dim + 1, nbr);
    }
}

inline void debug_check_tensor_non_vide(Tensor& tensor, std::string nom = "Tensor"){
    Shape s = tensor.get_shape();
    Print("get_shape ", s.print());

    if(s.len() == 0){
        Print(nom, " shape vide");
        return;
    }
    std::vector<size_t> indices(s.len(), 0);
    int nbr = 0;
    debug_check_tensor_non_vide_rec(tensor, s, indices, 0, nbr);
    if(nbr != 0)
        Print(nom, " PAS VIDE !!! nbr valeur : ", nbr);
    else
        Print(nom, " VIDE !!!");
}