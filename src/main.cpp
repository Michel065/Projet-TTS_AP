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

void test_UpSampling(DeviceType device){
    Tensor X,XX, x_test;
    Print("Chargement des datas:");
    get_data_Cifar_10(X,x_test,device);
    
    //X = X.extraction_section_axe_0(0,10000);
    X /= 255.0;
    XX = X;
    int nbr_val=1;
    x_test = x_test.extraction_section_axe_0(0,nbr_val);
    x_test /= 255.0;
    Print("Chargement des datas Fini. X(",X.get_shape()[0],") y(",x_test.get_shape()[0],")");



    Print("construction model.");
    Model model({.input_shape = Shape({3,32,32}), .eta = 1, .device=device});

    model.add(new LayerFlatten());

    model.add(new LayerDense(512));
    model.add(new LayerRelu());

    model.add(new LayerDense(256));
    model.add(new LayerRelu());

    model.add(new LayerDense(512));
    model.add(new LayerRelu());

    model.add(new LayerUnflatten());
    model.add(new LayerSigmoid());

    model.set_loss_function(new LossMSE());
    model.add_callback(new CallbackEarlyStopLoss({.patience = 5}));

    model.set_affichge_level(1);

    model.print();

    Print("entrainement.");
    model.fit(X,XX,150,128);

    model.create_graph_loss_entrainement();
    //model.save("./models/model_cnn_upscaling.json",false);

    Tensor Y_test_pred = model.forward(x_test);
    debug_check_tensor_non_vide(Y_test_pred);
    for(size_t i=0;i<(size_t)nbr_val;i++){
		debug_check_tensor_non_vide_batch(Y_test_pred, i, "Y_pred");
		debug_check_tensor_non_vide_batch(x_test, i, "Y_reel");
        print_exemple_images_bi(x_test,Y_test_pred,i,false);
    }

}

void test_AutoEncoder_Cifar(DeviceType device){
    Tensor X, x_test;

    Print("Chargement des datas:");
    get_data_Cifar_10(X, x_test, device);

    X /= 255.0f;

    int nbr_val = 1;
    x_test = x_test.extraction_section_axe_0(0, nbr_val);
    x_test /= 255.0f;

    Print("Chargement des datas Fini. X(", X.get_shape()[0], ") test(", x_test.get_shape()[0], ")");

    Print("construction encoder.");
    Model* encoder = new Model({.input_shape = Shape({3,32,32}), .eta = 1, .device=device});

    encoder->add(new LayerConv2D(32,3));
    encoder->add(new LayerRelu());
    encoder->add(new LayerMaxPool2D());

    encoder->add(new LayerConv2D(16,3));
    encoder->add(new LayerRelu());
    encoder->add(new LayerMaxPool2D());

    encoder->add(new LayerFlatten());

    encoder->add(new LayerDense(256));
    encoder->add(new LayerRelu());

    Print("construction decoder.");
    Model* decoder = new Model({.input_shape = Shape({256}), .eta = 0.1, .device=device});

    decoder->add(new LayerDense(16 * 8 * 8));
    decoder->add(new LayerRelu());

    decoder->add(new LayerUnflatten(Shape({16,8,8})));

    decoder->add(new LayerUpSampling2D());
    decoder->add(new LayerConv2D(16,3));
    decoder->add(new LayerRelu());

    decoder->add(new LayerUpSampling2D());
    decoder->add(new LayerConv2D(32,3));
    decoder->add(new LayerRelu());

    decoder->add(new LayerConv2D(3,3));
    decoder->add(new LayerSigmoid());

    encoder->print();
    decoder->print();

    Auto_Encoder ae(encoder, decoder);
    ae.set_loss_function(new LossMSE());

    Print("entrainement.");
    ae.fit(X, 15, 128);

    Tensor Y_test_pred = ae.predict(x_test);

    debug_check_tensor_non_vide(Y_test_pred);

    for(size_t i = 0; i < (size_t)nbr_val; i++){
        debug_check_tensor_non_vide_batch(Y_test_pred, i, "Y_pred");
        debug_check_tensor_non_vide_batch(x_test, i, "Y_reel");
        print_exemple_images_bi(x_test, Y_test_pred, i, false);
    }

    delete encoder;
    delete decoder;
}

int main() {
    test_AutoEncoder_Cifar(DeviceType::GPU);
    
    return 0;
}