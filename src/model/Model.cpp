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
	int somme=0;
	Print("Model : " + _model_name);
	for(auto& layer : _layers){
        layer->print();
		somme+=layer->get_nbr_params();
    }
	Print(Color::PINK,"Nombre de params entraible:",somme);
}

void Model::add(Layer* layer){
	if(!_layers.empty()){
        layer->set_input_shape(_layers.back()->get_output_shape());
    }
	else{
		layer->set_input_shape(_input_shape);
	}
	layer->set_model(this);
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

void Model::fit(Tensor input,Tensor y,int epochs,int batch_size){
	std::vector<Tensor> input_split = input.separation_batch(batch_size);
	std::vector<Tensor> y_split = y.separation_batch(batch_size);
	int nbr_split = input_split.size();
	Tensor Y_pred;
	float loss_moy=0;
	for(int i=0;i<epochs;i++){
		loss_moy=0;
		for(int id_it=0; id_it<nbr_split; id_it++){
			Print_over("Epochs : ", i + 1, "/", epochs, " iteration:", id_it + 1, "/", nbr_split);
			Y_pred = forward(input_split[id_it]);
			loss_moy += calcul_loss(Y_pred, y_split[id_it]);
			backward(calcul_grad_init(Y_pred, y_split[id_it]));
		}
		_train_loss_history.push_back(loss_moy/nbr_split);
	}
}

float Model::calcul_loss(Tensor Y_pred, Tensor y_reel){
    if(Y_pred.shape.dims != y_reel.shape.dims)
		Throw_Error("Dimensions y_pred et y_reel non valides.(calcul_loss())");
    return (Y_pred - y_reel).pow(2).moyenne();
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

Tensor Model::calcul_grad_init(Tensor Y_pred,Tensor y_reel){
    if(Y_pred.shape.dims != y_reel.shape.dims)
		Throw_Error("Dimensions y_pred et y_reel non valides.(calcul_grad_init())");
    return (Y_pred - y_reel) * (2.0f / Y_pred.size());

}

Tensor Model::predict(Tensor input){
	return forward(input);
}

float Model::get_eta(){
	return _eta;
}