#include "model/Loss/LossBinaryCrossEntropy.h"

float LossBinaryCrossEntropy::calcul_loss(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.get_shape().dims != y_true.get_shape().dims)
        Throw_Error("Dimensions invalides (LossBinaryCrossEntropy)");
    Tensor y_p = y_pred;
    y_p.clip(eps, 1.0f - eps);

    Tensor log_y_p = y_p;
    log_y_p.log();

    Tensor log_y_1p = (1.0f - y_p).log();
    Tensor loss = 0 - (y_true * log_y_p + (1.0f - y_true) * log_y_1p);
    return loss.moyenne();
}

Tensor LossBinaryCrossEntropy::calcul_grad(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.get_shape().dims != y_true.get_shape().dims)
        Throw_Error("Dimensions invalides (LossBinaryCrossEntropy grad)");
    int batch_size = y_pred.get_shape()[0];
    Tensor y_p = y_pred;
    y_p.clip(eps, 1.0f - eps);
    Tensor grad = (y_p - y_true) / (y_p * (1.0f - y_p));
    return grad / batch_size;
}