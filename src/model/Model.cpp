#include <iostream>
#include <string>
#include <chrono>

#include "model/Model.h"
#include "outil/Print.h"

Model::Model(std::string path){
	load_path(path);
}

Model::Model(ModelConfig model_config){
	_model_name=model_config.model_name;
	_input_shape=model_config.input_shape;
	_eta=model_config.eta;
	_loss_function=model_config.loss_function;
	_device=model_config.device;
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
	layer->get_from_model();
	if(!_layers.empty()){
        layer->set_input_shape(_layers.back()->get_output_shape());
    }
	else{
		layer->set_input_shape(_input_shape);
	}
	_layers.push_back(layer);
}

Tensor Model::forward(Tensor& input){
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

float Model::update_time_estimation(float temps_it, int total_it, int actuel_it) {
    avg_time = (avg_time * count + temps_it) / (count + 1);
    count++;
    return avg_time * (total_it - actuel_it);
}

void Model::fit(Tensor input,Tensor y,int epochs,int batch_size){
	std::vector<Tensor> input_split = input.separation_batch(batch_size);
	std::vector<Tensor> y_split = y.separation_batch(batch_size);
	int nbr_split = input_split.size();
	Tensor Y_pred;
	float loss_moy=0,loss_tmp=0;
	early_stop=false;

	if(_loss_function == nullptr){
		Throw_Error("Fonction loss mnquante. (fit)");
	}


	//ajout d'une prediction du temps suite a l'ajout du CNN
	avg_time = 0;
	count = 0;
	
	for(int i=0;i<epochs;i++){		
		loss_moy=0;
		for(int id_it=0; id_it<nbr_split; id_it++){
			auto debut_it = std::chrono::high_resolution_clock::now();

			Y_pred = forward(input_split[id_it]);
			loss_tmp = _loss_function->calcul_loss(Y_pred, y_split[id_it]);
			loss_moy += loss_tmp;
			backward(_loss_function->calcul_grad(Y_pred, y_split[id_it]));
			



			//prediction du temps calculs
			auto fin_it = std::chrono::high_resolution_clock::now();
			float temps_it = std::chrono::duration<float>(fin_it - debut_it).count();
			int total_it = epochs * nbr_split;
			int actuel_it = i * nbr_split + id_it + 1;
			float temps_restant = update_time_estimation(temps_it, total_it, actuel_it); //total



			if(_type_aff == 1)
				Print_over("Epochs : ",i+1,"/", epochs, " iteration : ", id_it+1, "/", nbr_split," loss train : ",std::round(loss_tmp * 10000.0f) / 10000.0f," temps restant (fin du train): ", std::round(temps_restant), "s");
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

void Model::create_graph_loss_entrainement(bool full){
	create_graphs_loss_screen(_train_loss_history,full);
}

Tensor Model::predict(Tensor input){
	return forward(input);
}

float Model::get_eta() const{
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

std::string Model::get_name_model() const{
	return _model_name;
}

Shape Model::get_shape_input() const{
	return _input_shape;
}

void Model::set_loss_function(Loss* loss_function){
	_loss_function = loss_function;
}

const std::vector<Layer*>& Model::get_layers() const {
	return _layers;
}

void Model::save(std::string path) {
    json j = this;

    std::ofstream file(path);
    if (!file.is_open())
        Throw_Error("Impossible d'ouvrir le fichier en ecriture : ", path);

    file << j.dump();
    file.close();
	Print("Modele sauvegarde path:",path);
}

void Model::reformat(ModelConfig model_config){
	_model_name=model_config.model_name;
	_input_shape=model_config.input_shape;
	_eta=model_config.eta;
	_loss_function=model_config.loss_function;
}

void Model::load_path(std::string path) {
    std::ifstream file(path);
    if (!file.is_open())
        Throw_Error("Impossible d'ouvrir le fichier en lecture : ", path);

    json j;
    file >> j;
    file.close();

    from_json(j, this);
	Print("Modele charge  de path:",path);
}

void Model::add_from_save(Layer* layer){
	layer->set_model(this);
	layer->get_from_model();
	_layers.push_back(layer);
}

DeviceType  Model::get_device() const{
	return _device;
}