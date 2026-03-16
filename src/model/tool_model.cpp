#include <iostream>
#include <string>

#include "model/tool_model.h"
#include "tool/print.h"

Model::Model(std::string model_name){
	_model_name=model_name;
}

void Model::print(){
	Print("Model : " + _model_name);
}