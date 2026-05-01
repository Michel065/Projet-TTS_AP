#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "outil/Print.h"
#include "test.h"
#include "model/Model.h"
#include "model/Callback/CallbackEarlyStopLoss.h"

// imports des base pour nos models
#include "model/Layer_ALL.h"
#include "model/Loss/Loss_ALL.h"

void test_UpSampling(DeviceType device){
    Tensor X, y, x_test, y_test;
    Print("Chargement des datas:");
    get_data_CNN(X, y, x_test,y_test,device);
    Print("Chargement des datas Fini. X(",X.get_shape()[0],") y(",x_test.get_shape()[0],")");

    X/=255.0;
    y = X;

    Print("construction model.");
    Model model({.input_shape = Shape({1,28,28}), .eta = 1, .device=device}); // image deja noralisé
    model.add(new LayerConv2D(32,3));
    model.add(new LayerRelu());
    model.add(new LayerMaxPool2D());
 
    model.add(new LayerConv2D(16,3));
    model.add(new LayerRelu());
    model.add(new LayerMaxPool2D());

    model.add(new LayerFlatten()); // transition dense

    model.add(new LayerDense(10));
    model.add(new LayerRelu());
    model.add(new LayerDense(10));
    
    model.add(new LayerUnflatten()); // retour conv  attention marche que si deja LayerFlatten
    
    model.add(new LayerUpSampling2D());
    model.add(new LayerConv2D(16,3));
    model.add(new LayerRelu());
 
    model.add(new LayerUpSampling2D());
    model.add(new LayerConv2D(32,3));
    model.add(new LayerRelu());

    //preparation de la sortie
    model.add(new LayerConv2D(1,3));
    model.add(new LayerSigmoid());

    model.set_loss_function(new LossMSE());
    model.add_callback(new CallbackEarlyStopLoss({.patience = 5}));

    model.set_affichge_level(1);

    model.print();

    Print("entrainement.");
    model.fit(X,y,75,128);

    model.create_graph_loss_entrainement();
    model.save("./models/model_cnn_upscaling.json",false);
}

int main() {
    test_UpSampling(DeviceType::GPU);
    
    return 0;
}