#pragma once
#include "model/Loss/Loss.h"

class LossCrossEntropy : public Loss {
public:
    float calcul_loss(const Tensor& y_pred, const Tensor& y_true) override;
    Tensor calcul_grad(const Tensor& y_pred, const Tensor& y_true) override;
};