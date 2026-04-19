#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cstdint>

#include "outil/Print.h"
#include "model/Model.h"
#include "model/Layer_dense/Layer_dense.h"
#include "model/Layer_activation/Layer_sigmoid.h"
#include "model/Layer_activation/Layer_relu.h"
#include "model/Layer_activation/Layer_softmax.h"
#include "model/Tool/Shape.h"

//pour les datasets via csv
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <chrono>



void get_data_lineaire(Tensor& X, Tensor& y, Tensor& x_test);
void get_data_non_lineaire(Tensor& X, Tensor& y, Tensor& x_test,size_t n=500,DeviceType device = DeviceType::CPU);

void load_mnist_csv(Tensor& X, Tensor& y, const std::string& path,DeviceType device= DeviceType::CPU);
void get_data_CNN(Tensor& X_train,Tensor& y_train,Tensor& X_test,Tensor& y_test,DeviceType device = DeviceType::CPU);
void print_exemeple_image(Tensor& images,size_t index);
void evaluate_cnn(Model& model,Tensor X, Tensor y);

void test_perf_cpu_gpu_simple(size_t N = 2000);
