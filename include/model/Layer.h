#pragma once
#include "model/Tool/Tensor.h"
#include "model/Tool/Shape.h"
#include "outil/Print.h"

class Model;

class Layer {
public:
    Layer(std::string nom_couche="Inconue"):nom_couche(nom_couche){}
    virtual ~Layer() {}

    virtual void set_input_shape(Shape shape_input);
    void set_output_shape(Shape shape_output);
    
    Shape get_output_shape();
    Shape get_input_shape();

    void set_model(Model* model_global);

    virtual void build() = 0;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad) = 0;

    void print_couche_msg(std::string msg,Color couleur = Color::DEFAULT);

    void print();
    int get_nbr_params();


protected:
	Shape _shape_input;
	Shape _shape_output;
	std::string nom_couche;
    size_t _nb_params = 0;

    Model* _model=nullptr;
};