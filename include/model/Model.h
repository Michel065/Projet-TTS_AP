#pragma once
#include <iostream>
#include <vector>
#include <string>

#include "model/Layer.h"
#include "model/Loss/Loss.h"
#include "model/Loss/LossBinaryCrossEntropy.h"
#include "model/Tool/graphs_tool.h"
#include "model/Callback/Callback.h"

class Model {
public:
	Model(std::string model_name,Shape input_shape,float eta=0.01f,Loss* loss_function = new LossBinaryCrossEntropy());
	
	void add(Layer* layer);
	Tensor forward(const Tensor& input);
	void backward(Tensor grad);
	Tensor predict(Tensor input);
	void fit(Tensor input,Tensor y,int epochs=50,int batch_size=64);
	void fit(const std::vector<std::vector<float>>& X,const std::vector<std::vector<float>>& y,int epochs=50,int batch_size=64);
	void fit(const std::vector<float>& X,const std::vector<float>& y,int epochs=50,int batch_size=64);
	void create_graph_loss_entrainement();
	void set_affichge_level(int val=0);
	float get_eta();
	std::vector<float>& get_history();
	void print();
	void add_callback(Callback* callback);
	void run_callback();


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
};