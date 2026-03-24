#include <iostream>
#include <string>

#include "model/Model.h"
#include "outil/Print.h"

Model::Model(ModelConfig model_config){
	_model_name=model_config.model_name;
	_input_shape=model_config.input_shape;
	_eta=model_config.eta;
	_loss_function=model_config.loss_function;
}

void Model::print(){
	int somme=0;
	Print(Color::PINK,"\nModel : " + _model_name);
	for(auto& layer : _layers){
        layer->print();
		somme+=layer->get_nbr_params();
    }
	Print(Color::PINK,"Nombre de params entraible:",somme);
}

void Model::add(Layer* layer){
	layer->set_model(this);
	if(!_layers.empty()){
        layer->set_input_shape(_layers.back()->get_output_shape());
    }
	else{
		layer->set_input_shape(_input_shape);
	}
	_layers.push_back(layer);
}

Tensor Model::forward(const Tensor& input){
    Tensor tmp = input;
    Tensor tmp_save = input;
    for(size_t i = 0; i < _layers.size(); ++i){
		tmp_save = tmp;
        tmp = _layers[i]->forward(tmp);
    }
    return tmp;
}

void Model::backward(Tensor grad){
	Tensor tmp = grad;
	for (auto it = _layers.rbegin(); it != _layers.rend(); ++it) {
		tmp = (*it)->backward(tmp);
	}
}

void Model::fit(Tensor input,Tensor y,int epochs,int batch_size){
	std::vector<Tensor> input_split = input.separation_batch(batch_size);
	std::vector<Tensor> y_split = y.separation_batch(batch_size);
	int nbr_split = input_split.size();
	Tensor Y_pred;
	float loss_moy=0,loss_tmp=0;
	early_stop=false;

	for(int i=0;i<epochs;i++){		
		loss_moy=0;
		for(int id_it=0; id_it<nbr_split; id_it++){
			Y_pred = forward(input_split[id_it]);
			loss_tmp = _loss_function->calcul_loss(Y_pred, y_split[id_it]);
			loss_moy += loss_tmp;
			backward(_loss_function->calcul_grad(Y_pred, y_split[id_it]));
			if(_type_aff == 1)
				Print_over("Epochs : ",i+1,"/", epochs, " iteration : ", id_it+1, "/", nbr_split," loss train : ",std::round(loss_tmp * 10000.0f) / 10000.0f);
		}
		_train_loss_history.push_back(loss_moy/nbr_split);
		
		if(_type_aff == 0)
			Print_over("Epochs : ", i + 1, "/", epochs," loss train : ",std::round(_train_loss_history.back() * 10000.0f) / 10000.0f);

		run_callback();
		if(early_stop){
			Print("\nArret anticipe du training.");
			break;
		}

	}
	early_stop=false;
}

void Model::create_graph_loss_entrainement(){
	create_graphs_loss(_train_loss_history);
}

void Model::fit(const std::vector<std::vector<float>>& X,const std::vector<std::vector<float>>& y,int epochs,int batch_size)
{
    fit(Tensor(X), Tensor(y), epochs, batch_size);
}

void Model::fit(const std::vector<float>& X,const std::vector<float>& y,int epochs,int batch_size)
{
    fit(Tensor(X), Tensor(y), epochs, batch_size);
}

Tensor Model::predict(Tensor input){
	return forward(input);
}

float Model::get_eta(){
	return _eta;
}

void Model::set_affichge_level(int val){
	_type_aff = val;
}

void Model::add_callback(Callback* callback){
	callback->set_Model(this);
	_callbacks.push_back(callback);
}

void Model::run_callback(){
	for(auto& callback : _callbacks){
        callback->on_epoch_end();
    }
}

std::vector<float>& Model::get_history(){
	return _train_loss_history;
}
