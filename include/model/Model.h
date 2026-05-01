#pragma once
#include <iostream>
#include <vector>
#include <string>

#include "model/Layer.h"
#include "model/Loss/Loss.h"
#include "model/Loss/LossBinaryCrossEntropy.h"
#include "model/Tool/graphs_tool.h"
#include "model/Callback/Callback.h"

struct ModelConfig {
    std::string model_name="model";
	Shape input_shape;
	float eta=0.01f;
	Loss* loss_function = new LossBinaryCrossEntropy();
	DeviceType device = DeviceType::CPU;
};

class Model {
public:
	Model(std::string path);
	Model(ModelConfig model_config);
	
	void add(Layer* layer);
	Tensor forward(Tensor& input);
	void backward(Tensor grad);
	Tensor predict(Tensor input);
	void fit(Tensor input,Tensor y,int epochs=50,int batch_size=64,bool shuffle=true);
	void create_graph_loss_entrainement(bool full=false);
	void set_affichge_level(int val=0);
	float get_eta() const;
	DeviceType get_device() const;
	std::vector<float>& get_history();
	void print();
	void add_callback(Callback* callback);
	void run_callback();
	std::string get_name_model() const;
	void set_loss_function(Loss* loss_function);
	void save(std::string path,bool aff=true);

	bool early_stop=false;
private:
    std::vector<Layer*> _layers;
	std::string _model_name;
	Shape _input_shape;
    std::vector<float> _train_loss_history;
	float _eta;
	Loss* _loss_function;
	int _type_aff=0;
    std::vector<Callback*> _callbacks;
	DeviceType _device;

	const std::vector<Layer*>& get_layers() const;
	Shape get_shape_input() const;
	void load_path(std::string path);
	void reformat(ModelConfig model_config);
	
	void add_from_save(Layer* layer);
	const std::vector<int> genere_indices_shuffle(int n);

	friend void to_json(json& j, const Model* model);
	friend void from_json(const json& j, Model* model);

	// pour l'estimation du temps
	float update_time_estimation(float temps_it, int total_it, int actuel_it);
	float avg_time = 0;
	int count = 0;
};

inline void from_json(const json& j, Model* model) {
    std::string name_model;
    Shape shape_input;
    float eta;
    std::vector<Layer*> layers;

    j.at("name_model").get_to(name_model);
    j.at("shape_input").get_to(shape_input);
    j.at("eta").get_to(eta);
	
    j.at("layers").get_to(layers);

    model->reformat({.model_name = name_model, .input_shape = shape_input, .eta = eta});

    for (Layer* layer : layers) {
        model->add_from_save(layer);
    }
}

inline void to_json(json& j, const Model* model) {
    j = {
        {"name_model", model->get_name_model()},
        {"shape_input", model->get_shape_input()},
        {"eta", model->get_eta()},
        {"layers", model->get_layers()}
    };
}