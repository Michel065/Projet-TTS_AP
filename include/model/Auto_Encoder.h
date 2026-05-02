#pragma once

#include "model/Model.h"
#include "model/Loss/Loss.h"
#include "model/Loss/LossMSE.h"

class Auto_Encoder {
private:
    Model* _encoder;
    Model* _decoder;
    Loss* _loss_function;

    std::vector<float> _train_loss_history;

public:
    Auto_Encoder(Model* encoder, Model* decoder);

    Tensor encode(Tensor& X);
    Tensor decode(Tensor& Z);
    Tensor forward(Tensor& X);
    Tensor predict(Tensor& X);

    void set_loss_function(Loss* loss);
    void fit(Tensor X, int epochs, int batch_size, bool shuffle = true);

    std::vector<float>& get_history();
};