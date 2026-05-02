#include "model/Loss/LossMSE.h"

float LossMSE::calcul_loss(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.get_shape().dims != y_true.get_shape().dims)
        Throw_Error("Dimensions invalides (LossMSE)");
    return (y_pred - y_true).pow(2).moyenne();
}

Tensor LossMSE::calcul_grad(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.get_shape().dims != y_true.get_shape().dims)
        Throw_Error("Dimensions invalides (LossMSE grad)");
    float batch_size = (float)y_pred.get_shape()[0];
    return (y_pred - y_true) * (2.0f / batch_size);
}

/*
float LossMSE::calcul_loss(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.get_shape().dims != y_true.get_shape().dims)
        Throw_Error("Dimensions invalides (LossMSE)");

    Tensor diff = y_pred - y_true;
    Tensor poids = y_true * 5.0f + 1.0f;
    return (poids * diff.pow(2)).moyenne();
}

Tensor LossMSE::calcul_grad(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.get_shape().dims != y_true.get_shape().dims)
        Throw_Error("Dimensions invalides (LossMSE grad)");
    float total = (float)y_pred.get_shape().size();
    Tensor poids = y_true * 5.0f + 1.0f;
    return poids * (y_pred - y_true) * (2.0f / total);
}*/