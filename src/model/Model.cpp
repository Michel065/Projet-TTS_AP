#include <iostream>
#include <string>

#include "model/Model.h"
#include "tool/Print.h"

Model::Model(std::string model_name){
	_model_name=model_name;
}

void Model::print(){
	Print("Model : " + _model_name);
}