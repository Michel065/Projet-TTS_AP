#pragma once
#include "test_data.h"


#include "outil/Print.h"

#include "model/Model.h"
#include "model/Callback/CallbackEarlyStopLoss.h"

// imports des base pour nos models et donc les tests avancé
#include "model/Layer_ALL.h"
#include "model/Loss/Loss_ALL.h"



void print_exemeple_image(Tensor& images,size_t index);
void evaluate_cnn(Model& model,Tensor X, Tensor y);


void test_non_lineaire(DeviceType device = DeviceType::GPU);
void test_load();
void test_CNN(DeviceType device = DeviceType::CPU);
void test_CNN_load(DeviceType device = DeviceType::CPU);

void test_UpSampling(DeviceType device = DeviceType::GPU); // GPU par defaut puisque CPU pas fait
