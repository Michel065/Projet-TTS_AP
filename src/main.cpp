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
#include "model/Layer_conv/LayerConv2D.h"
#include "model/Layer_conv/LayerFlatten.h"
#include "model/Layer_conv/LayerMaxPool2D.h"

void test_non_lineaire(DeviceType device = DeviceType::GPU){
    Tensor X, y, x_test;
    get_data_non_lineaire(X, y, x_test,1000,device);
    size_t nbr_neur_in = (X.get_shape()[1]);
    int nbr_neur_out = y.get_shape()[1];

    Print("construction model.");
    Model model({.input_shape = Shape({nbr_neur_in}), .eta = 0.1 ,.device=device});
    model.add(new LayerNormalisation({-3,-3},{3,3}));// c le min et le max a la main
    model.add(new LayerDense(20));
    model.add(new LayerRelu());
    model.add(new LayerDense(20));
    model.add(new LayerRelu());
    model.add(new LayerDense(5));
    model.add(new LayerRelu());
    model.add(new LayerDense(nbr_neur_out));
    model.add(new LayerSigmoid());
    model.add_callback(new CallbackEarlyStopLoss({.patience = 7}));
    model.set_loss_function(new LossBinaryCrossEntropy());

    //model.set_affichge_level(1);

    Print("entrainement.");
    model.fit(X,y,150,4,false);
    
    Print("Test:");
    Tensor y_test = model.predict(x_test).round(2)*100;
    Print("Prediction :",y_test);

    //model.print();
    model.create_graph_loss_entrainement();
    //model.save("./models/model.json");
    
}

void test_load(){
    Tensor X, y, x_test;
    get_data_non_lineaire(X, y, x_test,1000);

    Model model("./models/model.json");
    model.set_loss_function(new LossBinaryCrossEntropy());
    

    Print("Test:");
    Tensor y_test = model.predict(x_test).round(2)*100;
    
    Print("Prediction :",y_test);
}

void test_CNN(){
    Tensor X, y, x_test, y_test;
    get_data_CNN(X, y, x_test,y_test,DeviceType::GPU);

    Print("construction model.");
    Model model({.input_shape = Shape({1,28,28}), .eta = 0.11, .device=DeviceType::GPU});

    model.add(new LayerConv2D(8,3));
    model.add(new LayerRelu());
    model.add(new LayerMaxPool2D());
 
    model.add(new LayerConv2D(16,3));
    model.add(new LayerRelu());
    model.add(new LayerMaxPool2D());

    model.add(new LayerFlatten());

    model.add(new LayerDense(64));
    model.add(new LayerRelu());
    model.add(new LayerDense(10));
    model.add(new LayerSigmoid());
    //model.add_callback(new CallbackEarlyStopLoss({.patience = 5}));

    model.set_affichge_level(1);

    Print("entrainement.");
    model.fit(X,y,1,4);
    
    Print("Test:");
    Tensor pred = model.predict(x_test);
    for(size_t i = 0; i < x_test.get_shape()[0]; i++){
        size_t p_class = 0;
        float max_pred = pred.get({i,0});
        for(size_t j = 1; j < 10; j++){
            if(pred.get({i,j}) > max_pred){
                max_pred = pred.get({i,j});
                p_class = j;
            }
        }
        size_t r_class = 0;
        for(size_t j = 0; j < 10; j++){
            if(y_test.get({i,j}) == 1.0f){
                r_class = j;
                break;
            }
        }
        Print("Image ", i, " pred:", p_class, " vrai:", r_class);
    }

    model.print();
    model.create_graph_loss_entrainement();
    model.save("./models/model.json");
    
}

int main() {
    test_non_lineaire(DeviceType::CPU);
    return 0;
}