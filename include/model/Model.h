#pragma once
#include <iostream>
#include <vector>
#include <string>

#include "model/Layer.h"

class Model {

public:
	Model(std::string model_name,Shape input_shape,float eta=0.01f);
	
	void add(Layer* layer);
	Tensor forward(Tensor input);
	void backward(Tensor grad);

	void fit(Tensor input,Tensor y,int epochs=50);
	
	void print();

private:
    std::vector<Layer*> _layers;
	std::string _model_name;
	Shape _input_shape;
	float _eta;
};