#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "outil/Print.h"
#include "test.h"
#include "model/Model.h"
#include "model/Layer_dense/Layer_dense.h"
#include "model/Layer_dense/Layer_norm.h"
#include "model/Layer_activation/Layer_sigmoid.h"
#include "model/Layer_activation/Layer_relu.h"
#include "model/Layer_activation/Layer_softmax.h"
#include "model/Tool/Shape.h"
#include "model/Callback/CallbackEarlyStopLoss.h"

void test_actu(){
    Tensor X, y, x_test;
    get_data_non_lineaire(X, y, x_test,1000);

    size_t nbr_neur_in = (size_t)X.shape[1];
    int nbr_neur_out = y.shape[1];

    Print("construction model.");
    Model model({.input_shape = Shape({nbr_neur_in}), .eta = 0.11});
    model.add(new LayerNormalisation({-3,-3},{3,3}));// c le min et le max a la main
    model.add(new LayerDense(20));
    model.add(new LayerRelu());
    model.add(new LayerDense(20));
    model.add(new LayerRelu());
    model.add(new LayerDense(5));
    model.add(new LayerRelu());
    model.add(new LayerDense(nbr_neur_out));
    model.add(new LayerSigmoid());

    model.add_callback(new CallbackEarlyStopLoss({.patience = 5}));

    //model.set_affichge_level(1);

    Print("entrainement.");
    model.fit(X,y,500,4);
    
    Print("Test:");
    Tensor y_test = model.predict(x_test).round(2)*100;
    Print("Prediction :",y_test);/**/

    //model.print();
    //model.create_graph_loss_entrainement();
}

int main() {
    test_actu();
    return 0;
}