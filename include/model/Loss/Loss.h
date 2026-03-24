#pragma once
#include <iostream>
#include "model/Tool/Tensor.h"

class Loss {
public:
    virtual float calcul_loss(const Tensor& y_pred, const Tensor& y_true) = 0;
    virtual Tensor calcul_grad(const Tensor& y_pred, const Tensor& y_true) = 0;
    virtual ~Loss() = default;

protected:
    float eps = 1e-6f; // ca bug si c'est trop bas
};

/*
Rappel:
    Loss          =>     Activation
    MSE           =>     aucune / sigmoid
    BCE           =>     sigmoid
    CrossEntropy  =>     CrossEntropy
 */