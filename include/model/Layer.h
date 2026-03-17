#pragma once
#include "model/Tensor.h"
#include "model/Struct.h"
#include "tool/Print.h"

class Layer {
public:
    virtual ~Layer() {}

    virtual void set_input_shape(Shape shape_input);
    void set_output_shape(Shape shape_output);
    
    Shape get_output_shape();
    Shape get_input_shape();

    void set_id(int id);

    virtual void build() = 0;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad) = 0;

    void print_couche_msg(std::string msg,Color couleur = Color::DEFAULT);

protected:
	Shape _shape_input;
	Shape _shape_output;
	std::string nom_couche = "Inconue";
};