#include "model/Loss/LossMSE.h"

float LossMSE::calcul_loss(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.get_shape().dims != y_true.get_shape().dims)
        Throw_Error("Dimensions invalides (LossMSE)");
    return (y_pred - y_true).pow(2).moyenne();
}

Tensor LossMSE::calcul_grad(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.get_shape().dims != y_true.get_shape().dims)
        Throw_Error("Dimensions invalides (LossMSE grad)");
    int batch_size = y_pred.get_shape()[0];
    return (y_pred - y_true) * (2.0f / batch_size);
}