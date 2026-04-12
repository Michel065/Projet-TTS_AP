#include "model/Loss/LossCrossEntropy.h"

float LossCrossEntropy::calcul_loss(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.shape.dims != y_true.shape.dims)
        Throw_Error("Dimensions invalides (LossCrossEntropy)");
    Tensor y_p_l = y_pred;
    y_p_l.clip(eps, 1.0f - eps);
    y_p_l.log();
    Tensor loss = 0-(y_true * y_p_l);
    return loss.sum_per_row().moyenne();
}

Tensor LossCrossEntropy::calcul_grad(const Tensor& y_pred, const Tensor& y_true){
    if(y_pred.shape.dims != y_true.shape.dims)
        Throw_Error("Dimensions invalides (LossCrossEntropy grad)");
    int batch_size = y_pred.shape[0];
    return (y_pred - y_true) / batch_size;
}