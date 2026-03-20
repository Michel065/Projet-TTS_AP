#include <iostream>
#include <string>

#include "model/Model.h"
#include "outil/Print.h"

Model::Model(std::string model_name,Shape input_shape,float eta){
	_model_name=model_name;
	_input_shape=input_shape;
	_eta=eta;
}

void Model::print(){
	Print("Model : " + _model_name);
	for(auto& layer : _layers){
        layer->print();
    }
}

void Model::add(Layer* layer){
	if(!_layers.empty()){
        layer->set_input_shape(_layers.back()->get_output_shape());
    }
	else{
		layer->set_input_shape(_input_shape);
	}
	layer->init_eta(_eta);
	_layers.push_back(layer);
}

Tensor Model::forward(Tensor input){
	Tensor tmp = input;
	for(auto& layer : _layers){
        tmp = layer->forward(tmp);
    }
	return tmp;
}

void Model::backward(Tensor grad){
	Tensor tmp = grad;
	for (auto it = _layers.rbegin(); it != _layers.rend(); ++it) {
		tmp = (*it)->backward(tmp);
	}
}

void Model::fit(Tensor input,Tensor y,int epochs){
	Tensor Y_pred,grad_init;
	for(int i=0;i<epochs;i++){
		Print("Epochs : ",i+1,"/",epochs);
		Y_pred = forward(input);
	
		int taille_sortie = Y_pred.size();

        float loss = calcul_loss(Y_pred, y);
        _train_loss_history.push_back(loss);
		
		//calculs init
		grad_init = (Y_pred-y)*(2.0f / taille_sortie);
		
		backward(grad_init);
	}
}

float Model::calcul_loss(Tensor Y_pred, Tensor y_reel){
    if(Y_pred.shape.dims != y_reel.shape.dims)
		Throw_Error("Dimensions y_pred et y_reel non valides.(calcul_loss())",Color::RED);
    return (Y_pred - y_reel).pow(2).moyenne();
}

void Model::create_graph_loss_entrainement(){
	create_graphs_loss(_train_loss_history);
}