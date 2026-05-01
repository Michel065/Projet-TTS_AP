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

void test_non_lineaire(DeviceType device = DeviceType::GPU){
    Tensor X, y, x_test;
    get_data_non_lineaire(X, y, x_test,2000,device);
    size_t nbr_neur_in = (X.get_shape()[1]);
    int nbr_neur_out = y.get_shape()[1];

    Print("construction model.");
    Model model({.input_shape = Shape({nbr_neur_in}), .eta = 2 ,.device=device});
    model.add(new LayerNormalisation({-3,-3},{3,3}));// c le min et le max a la main
    model.add(new LayerDense(20));
    model.add(new LayerRelu());
    model.add(new LayerDense(20));
    model.add(new LayerRelu());
    model.add(new LayerDense(5));
    model.add(new LayerRelu());
    model.add(new LayerDense(nbr_neur_out));
    model.add(new LayerSigmoid());
    model.set_loss_function(new LossBinaryCrossEntropy());

    model.add_callback(new CallbackEarlyStopLoss({.patience = 15}));

    //model.set_affichge_level(2);

    Print("entrainement.");
    model.fit(X,y,1500,256);
    
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

void test_CNN(DeviceType device = DeviceType::CPU){
    Tensor X, y, x_test, y_test;
    Print("Chargement des datas:"); // je met un print car pas otpi tres long.
    get_data_CNN(X, y, x_test,y_test,device);
    Print("Chargement des datas Fini. X(",X.get_shape()[0],") y(",x_test.get_shape()[0],")");
     
    /*
    int nbr_image_train=50;
   
    X=X.extraction_section_axe_0(0,nbr_image_train);
    y=y.extraction_section_axe_0(0,nbr_image_train);
    
    x_test=x_test.extraction_section_axe_0(0,nbr_image_train);
    y_test=y_test.extraction_section_axe_0(0,nbr_image_train);
    Print("reduce X(",X.get_shape()[0],") y(",x_test.get_shape()[0],")");
    */

    Print("construction model.");
    Model model({.input_shape = Shape({1,28,28}), .eta = 1, .device=device});
    model.add(new LayerNormalisationImage());

    model.add(new LayerConv2D(2,3));
    model.add(new LayerRelu());
    model.add(new LayerMaxPool2D());
 
    model.add(new LayerConv2D(16,3));
    model.add(new LayerRelu());
    model.add(new LayerMaxPool2D());

    model.add(new LayerFlatten());

    model.add(new LayerDense(10));
    model.add(new LayerRelu());
    model.add(new LayerDense(10));
    model.add(new LayerSoftMax());
    model.set_loss_function(new LossCrossEntropy());
    model.add_callback(new CallbackEarlyStopLoss({.patience = 5}));

    model.set_affichge_level(1);

    // var utiliser pour debug pour cacher tous les prints qui me servent a debug
    
    //model.print();

    Print("entrainement.");
    model.fit(X,y,75,128);

    Print("Test:");
    evaluate_cnn(model,x_test,y_test);
    
    /*
    Print("Test:");
    Tensor pred = model.predict(x_test);
    for(size_t i = 0; i < x_test.get_shape()[0]/4; i++){
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
        //print_exemeple_image(x_test,i);
    }*/

    model.create_graph_loss_entrainement();
    model.save("./models/model_cnn.json",false);
}

void test_CNN_load(DeviceType device = DeviceType::CPU){
    Tensor X, y, x_test, y_test;
    Print("Chargement des datas:");
    get_data_CNN(X, y, x_test,y_test,device);
    Print("Chargement des datas Fini. X(",X.get_shape()[0],") y(",x_test.get_shape()[0],")");
    
    /*int nbr_image_train=50;
    x_test=x_test.extraction_section_axe_0(0,nbr_image_train);
    y_test=y_test.extraction_section_axe_0(0,nbr_image_train);
    Print("reduce X(",X.get_shape()[0],") y(",x_test.get_shape()[0],")");
    */

    Print("construction model.");
    
    Model model("./models/model_cnn.json");
    model.set_loss_function(new LossCrossEntropy());
    
    Print("Test:");
    evaluate_cnn(model,x_test,y_test);
}

int main() {
    test_CNN(DeviceType::GPU);
    //test_CNN_load(DeviceType::GPU);
    
    return 0;
}