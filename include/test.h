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

void get_data_lineaire(Tensor& X, Tensor& y, Tensor& x_test);
void get_data_non_lineaire(Tensor& X, Tensor& y, Tensor& x_test,size_t n=500);
void get_data_CNN(Tensor& X, Tensor& y, Tensor& x_test, Tensor& y_test);
