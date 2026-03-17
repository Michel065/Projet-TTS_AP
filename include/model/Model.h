#pragma once
#include <iostream>
#include <vector>
#include <string>

#include "model/Layer.h"

class Model {

public:
	Model(std::string model_name);
	
	void add(Layer* layer);
	
	void print();

	Tensor forward(Tensor x);

private:
    std::vector<Layer*> _layers;
	std::string _model_name;
};