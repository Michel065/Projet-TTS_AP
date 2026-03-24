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

void test_actu(){
    Tensor X, y, x_test;
    get_data_non_lineaire(X, y, x_test, 1500);

    int nbr_neur_in = X.shape[1];
    int nbr_neur_out = y.shape[1];

    /*
    Tensor X, y, x_test;
    get_data_non_lineaire(X, y, x_test, 500);
    */

    Print("construction model.");
    Model model("test",Shape({(size_t)nbr_neur_in}),0.01);
    model.add(new LayerDense(40));
    model.add(new LayerRelu());
    model.add(new LayerDense(20));
    model.add(new LayerRelu());
    model.add(new LayerDense(20));
    model.add(new LayerRelu());
    model.add(new LayerDense(5));
    model.add(new LayerRelu());
    model.add(new LayerDense(nbr_neur_out));
    model.add(new LayerSoftMax());

    Print("entrainement.");
    model.fit(X,y,500,64);
    
    Print("Test:");
    Tensor y_test = model.predict(x_test).round(3)*100;
    Print("Prediction :",y_test);
    //model.print();
    //model.create_graph_loss_entrainement();
}

int main() {
    test_actu();
    return 0;
}