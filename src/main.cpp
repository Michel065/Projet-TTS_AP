#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "outil/Print.h"
#include "test.h"
#include "model/Model.h"
#include "model/Layer_dense/Layer_dense.h"
#include "model/Layer_activation/Layer_sigmoid.h"
#include "model/Layer_activation/Layer_relu.h"
#include "model/Layer_activation/Layer_softmax.h"
#include "model/Tool/Shape.h"

void test_actu(){
    Tensor X,y;
    X = Tensor({
        {-1.0, -1.0},
        {-1.2, -0.8},
        {-0.8, -1.1},
        {-1.1, -1.3},

        {1.0, 1.0},
        {1.2, 0.9},
        {0.8, 1.1},
        {1.1, 1.3}
    });

    y = Tensor({
        {1.0, 0.0},
        {1.0, 0.0},
        {1.0, 0.0},
        {1.0, 0.0},

        {0.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0}
    });

    Tensor x_test = Tensor({
        {0.9, 1.0},
        {-0.9, -1.0}
    });

    Print("construction model.");
    Model model("test",Shape({2}),0.01);
    model.add(new LayerDense(10));
    model.add(new LayerRelu());
    model.add(new LayerDense(20));
    model.add(new LayerRelu());
    model.add(new LayerDense(10));
    model.add(new LayerRelu());
    model.add(new LayerDense(2));
    model.add(new LayerSoftMax());

    Print("entrainement.");
    model.fit(X,y,500);
    /*
    Print("Test:");
    Tensor y_test = model.forward(x_test).round(3)*100;
    Print("Prediction :",y_test);
    */
    model.create_graph_loss_entrainement();
}

int main() {
    //test_actu();
    test_RNN();
    return 0;
}