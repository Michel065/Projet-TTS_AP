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
    for(size_t i = 0; i < _layers.size(); ++i){
		//Print("forward i:",i, " name:",_layers[i]->get_name());
        tmp = _layers[i]->forward(tmp);
    }
    return tmp;
}

Tensor Model::backward(Tensor grad){
	Tensor tmp = grad;
	//int i=0;
	for (auto it = _layers.rbegin(); it != _layers.rend(); ++it) {
		//Print("backward i:",i);
		//i++;
		tmp = (*it)->backward(tmp);
	}
	return tmp;
}

float Model::update_time_estimation(float temps_it, int total_it, int actuel_it) {
    avg_time = (avg_time * count + temps_it) / (count + 1);
    count++;
    return avg_time * (total_it - actuel_it);
}

const std::vector<int> Model::genere_indices_shuffle(int n){
    std::vector<int> indices(n);
    for(int i = 0; i < n; i++)
        indices[i] = i;

    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));
	return indices;
}

float round_esti(float val, int deci = 5){
	float dix = std::pow(10.0f, deci);
	return std::round(val * dix) / dix;
}

void Model::fit(Tensor input,Tensor y,int epochs,int batch_size,bool shuffle){	
	Tensor Y_pred;
	float loss_moy=0,loss_tmp=0;
	early_stop=false;

	if(_loss_function == nullptr){
		Throw_Error("Fonction loss mnquante. (fit)");
	}


	//ajout d'une prediction du temps suite a l'ajout du CNN
	avg_time = 0;
	count = 0;

	auto debut_train = std::chrono::high_resolution_clock::now();	
	for(int i=0;i<epochs;i++){
		auto t_epoch_start = std::chrono::high_resolution_clock::now();

		if(_type_aff == 2)
			Print("\n[Epoch ", i + 1, "] debut");
		//melange des datas:
		if(shuffle){
			const std::vector<int> indices = genere_indices_shuffle(input.get_shape()[0]);
			input.shuffle(indices);
			y.shuffle(indices);
		}

		auto t_after_shuffle = std::chrono::high_resolution_clock::now();
		if(_type_aff == 2){
			float dt = std::chrono::duration<float>(t_after_shuffle - t_epoch_start).count();
			Print("[Epoch ", i + 1, "] temps shuffle : ", round_esti(dt), "s");
		}
		
		std::vector<Tensor> input_split = input.separation_batch(batch_size);
		std::vector<Tensor> y_split = y.separation_batch(batch_size);
		int nbr_split = input_split.size();

		auto t_after_split = std::chrono::high_resolution_clock::now();
		if(_type_aff == 2){
			float dt = std::chrono::duration<float>(t_after_split - t_after_shuffle).count();
			Print("[Epoch ", i + 1, "] temps split : ", round_esti(dt), "s");
		}

		// entrainement pricipale
		loss_moy=0;
		float temps_restant=0;
   		
		auto t_train_loop_start = std::chrono::high_resolution_clock::now();
		for(int id_it=0; id_it<nbr_split; id_it++){
			auto debut_it = std::chrono::high_resolution_clock::now();
			
    		//Print("Dans le X source du fit id_it:", id_it);
			//debug_check_tensor_non_vide_batch(input_split[id_it], 0, "X source");

			Y_pred = forward(input_split[id_it]);

			//Print("Dans le Y pred du fit id_it:", id_it);
			//debug_check_tensor_non_vide_batch(Y_pred, 0, "Y_pred");

			loss_tmp = _loss_function->calcul_loss(Y_pred, y_split[id_it]);
			loss_moy += loss_tmp;
			backward(_loss_function->calcul_grad(Y_pred, y_split[id_it]));

			//debug
			/*if(id_it>50)
				return ;
			*/
			//prediction du temps calculs
			float temps_it = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - debut_it).count();
			temps_restant = update_time_estimation(temps_it, epochs * nbr_split, i * nbr_split + id_it + 1);
			
			if(epochs % 10 == 9){// sauvegarde de temps en temps
				save("./models/"+_model_name+".json",false);
			}

			if(_type_aff == 1) 
				Print_over("Epochs : ",i+1,"/",epochs," iteration : ",id_it+1,"/",nbr_split," loss train : ",round_esti(loss_moy/(id_it+1))," temps restant (fin du train): ",std::round(temps_restant),"s");
		}

		auto t_train_loop_end = std::chrono::high_resolution_clock::now();
		if(_type_aff == 2){
			float dt = std::chrono::duration<float>(t_train_loop_end - t_train_loop_start).count();
			Print("[Epoch ", i + 1, "] temps boucle train : ", round_esti(dt), "s");
		}

		_train_loss_history.push_back(loss_moy/nbr_split);
		
		auto t_after_loss = std::chrono::high_resolution_clock::now();
		if(_type_aff == 2){
			float dt = std::chrono::duration<float>(t_after_loss - t_train_loop_end).count();
			Print("[Epoch ", i + 1, "] temps post-loss : ", round_esti(dt), "s");
		}

		if(_type_aff == 0){
			Print_over("Epochs : ", i + 1, "/", epochs," loss train : ",round_esti(_train_loss_history.back())," temps restant (fin du train): ", std::round(temps_restant), "s");
		}
    	auto t_before_callback = std::chrono::high_resolution_clock::now();
		run_callback();
   	 	auto t_after_callback = std::chrono::high_resolution_clock::now();

		if(_type_aff == 2){
			float dt_cb = std::chrono::duration<float>(t_after_callback - t_before_callback).count();
			float dt_total = std::chrono::duration<float>(t_after_callback - t_epoch_start).count();

			Print("[Epoch ", i + 1, "] temps callback : ", round_esti(dt_cb), "s");
			Print("[Epoch ", i + 1, "] temps total epoch : ", round_esti(dt_total), "s");
		}

		
		if(early_stop){
			Print("\nArret anticipe du training.");
			break;
		}

	}
	early_stop=false;
	auto fin_train = std::chrono::high_resolution_clock::now();	
	float temps_total = std::chrono::duration<float>(fin_train - debut_train).count();
	Print("\nTemps total d'entrainement :",temps_total,"s.");
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

void Model::save(std::string path,bool aff) {
    json j = this;

    std::ofstream file(path);
    if (!file.is_open())
        Throw_Error("Impossible d'ouvrir le fichier en ecriture : ", path);

    file << j.dump();
    file.close();
	if(aff)
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


const Layer* Model::find_layer(std::string name){
	for(auto& layer : _layers){
		if(layer->get_name() == name){
			return layer;
		}
    }
	return NULL;
}
