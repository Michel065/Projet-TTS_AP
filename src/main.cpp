#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "outil/Print.h"
#include "test.h"
#include "model/Model.h"
#include "model/Auto_Encoder.h"
#include "model/Callback/CallbackEarlyStopLoss.h"

// imports des base pour nos models
#include "model/Layer_ALL.h"
#include "model/Loss/Loss_ALL.h"


void test_AutoEncoder_Cifar_v3(DeviceType device){
    Tensor X, x_test;
    Print("Chargement des datas:");
    get_data_Cifar_10_train(X, device);
    Print("Chargement des datas Fini. X(", X.get_shape()[0], ")");

    float eta = 0.001;
    Print("construction encoder.");
    Model* encoder = new Model({.model_name = "Encoder", .input_shape = Shape({3,32,32}), .eta = eta, .device=device});
    encoder->add(new LayerConv2D(16,3));
    encoder->add(new LayerRelu());
    encoder->add(new LayerMaxPool2D());

    encoder->add(new LayerConv2D(16,3));
    encoder->add(new LayerRelu());
    encoder->add(new LayerMaxPool2D());

    encoder->add(new LayerFlatten());
    encoder->add(new LayerIdentity()); // latent = 1024

    Print("construction decoder.");
    Model* decoder = new Model({.model_name = "Decoder", .input_shape = Shape({1024}), .eta = eta, .device=device}); //16*8*8 = 2024

    decoder->add(new LayerUnflatten(Shape({16,8,8})));

    decoder->add(new LayerUpSampling2D());
    decoder->add(new LayerConv2D(16,3));
    decoder->add(new LayerRelu());

    decoder->add(new LayerUpSampling2D());
    decoder->add(new LayerConv2D(16,3));
    decoder->add(new LayerRelu());

    decoder->add(new LayerConv2D(3,3));
    decoder->add(new LayerSigmoid());


    // petit print
    encoder->print();
    decoder->print();

    Auto_Encoder ae(encoder, decoder);
    ae.set_loss_function(new LossMSE());
    ae.add_callback(new CallbackEarlyStopLoss({.patience = 4}));

    Print("entrainement.");
    ae.fit(X, 50, 128);

    //sauvegarde
    ae.save("./models/model_AE.json");


    Print("Changement test!");// avec les datas du train, cequi et plus coherent en theorie
    int nbr_val = 2;
    x_test = X.extraction_section_axe_0(0, nbr_val);
    Tensor Y_test_pred = ae.predict(x_test);
    for(size_t i = 0; i < (size_t)nbr_val; i++){
        Print("Image ",i);
        print_exemple_images_bi(x_test, Y_test_pred, i, false);
        Print("");
        Print("");
        Print("");
        Print("");
        Print("");
        Print("");
        Print("");
        Print("");
    }
}

void test_AutoEncoder_Cifar_v3_load(DeviceType device){
    Tensor X, x_test;

    Print("Chargement des datas:");
    get_data_Cifar_10_train(X, device);
    Print("Chargement des datas Fini. X(", X.get_shape()[0], ")");

    Print("Chargement AutoEncoder.");
    Auto_Encoder ae("./models/model_AE.json");
    ae.set_loss_function(new LossMSE());

    Print("Changement test !");
    int nbr_val = 1;
    x_test = X.extraction_section_axe_0(0, nbr_val);

    Tensor Y_test_pred = ae.predict(x_test);

    for(size_t i = 0; i < (size_t)nbr_val; i++){
        Print("Image ", i);
        print_exemple_images_bi(x_test, Y_test_pred, i, false);

        Print("");
        Print("");
        Print("");
    }
}

int main() {
    //test_AutoEncoder_Cifar_v3(DeviceType::GPU);
    test_CNN(DeviceType::GPU);
    //test_non_lineaire(DeviceType::GPU);
    return 0;
}