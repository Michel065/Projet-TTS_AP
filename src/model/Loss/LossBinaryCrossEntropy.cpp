#include "model/Loss/LossBinaryCrossEntropy.h"

float LossBinaryCrossEntropy::calcul_loss(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.shape.dims != y_true.shape.dims)
        Throw_Error("Dimensions invalides (LossBinaryCrossEntropy)");
    Tensor y_p = y_pred.clip(eps, 1.0f - eps);
    Tensor loss = -(y_true * y_p.log() + (1.0f - y_true) * (1.0f - y_p).log());
    return loss.moyenne();
}

Tensor LossBinaryCrossEntropy::calcul_grad(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.shape.dims != y_true.shape.dims)
        Throw_Error("Dimensions invalides (LossBinaryCrossEntropy grad)");
    int batch_size = y_pred.shape[0];
    Tensor y_p = y_pred.clip(eps, 1.0f - eps);
    Tensor grad = (y_p - y_true) / (y_p * (1.0f - y_p));
    return grad / batch_size;
}