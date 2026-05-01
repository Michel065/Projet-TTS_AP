#pragma once
#include "model/Tool/Tensor/Tensor.h"
#include "model/Tool/Shape.h"
#include "outil/Print.h"
#include "model/Json/Json_gestion.h"

class Model;

class Layer {
public:
    Layer(std::string nom_couche="Inconue"):nom_couche(nom_couche){}
    virtual ~Layer() {}

    virtual void set_input_shape(Shape shape_input);
    void set_output_shape(Shape shape_output);
    
    Shape get_output_shape();
    Shape get_input_shape() const;

    void set_model(Model* model_global);
    virtual void get_from_model() {};

    virtual void build() = 0;

    virtual Tensor forward(Tensor& input) = 0;
    virtual Tensor backward(Tensor& grad) = 0;

    void print_couche_msg(std::string msg,Color couleur = Color::DEFAULT);

    void print();
    int get_nbr_params();
    std::string get_name();

    virtual void to_json(json& j) const = 0;
    virtual void load_json(const json& j) = 0;

    json to_json_layer() const;
    void load_json_layer(const json& j);
    
protected:
	Shape _shape_input;
	Shape _shape_output;
	std::string nom_couche;
    size_t _nb_params = 0;

    Model* _model=nullptr;
};

inline void from_json(const json& j, Layer*& layer) {
    std::string type = j.at("type");
    layer = LayerConstructorListe::create(type);
    layer->load_json_layer(j);
    layer->load_json(j);
}

inline void to_json(json& j, const Layer* layer) {
    j = layer->to_json_layer();
    layer->to_json(j);
}