#include "model/Auto_Encoder.h"
#include "outil/Print.h"

Auto_Encoder::Auto_Encoder(Model* encoder, Model* decoder){
    _encoder = encoder;
    _decoder = decoder;
    _loss_function = new LossMSE();
}

void Auto_Encoder::set_loss_function(Loss* loss){
    _loss_function = loss;
}

Tensor Auto_Encoder::encode(Tensor& X){
    return _encoder->forward(X);
}

Tensor Auto_Encoder::decode(Tensor& Z){
    return _decoder->forward(Z);
}

Tensor Auto_Encoder::forward(Tensor& X){
    Tensor Z = encode(X);
    return decode(Z);
}

Tensor Auto_Encoder::predict(Tensor& X){
    return forward(X);
}

void Auto_Encoder::fit(Tensor X, int epochs, int batch_size, bool shuffle){
    if(_loss_function == nullptr){
        Throw_Error("Fonction loss manquante. Auto_Encoder::fit");
    }

    for(int epoch = 0; epoch < epochs; epoch++){
        if(shuffle){
            std::vector<int> indices(X.get_shape()[0]);
            for(int i = 0; i < (int)indices.size(); i++)
                indices[i] = i;

            std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));
            X.shuffle(indices);
        }

        std::vector<Tensor> X_split = X.separation_batch(batch_size);
        float loss_moy = 0.0f;

        for(size_t id_it = 0; id_it < X_split.size(); id_it++){
            Tensor input = X_split[id_it];

            Tensor latent = _encoder->forward(input);
            Tensor Y_pred = _decoder->forward(latent);

            float loss = _loss_function->calcul_loss(Y_pred, input);
            loss_moy += loss;

            Tensor grad = _loss_function->calcul_grad(Y_pred, input);

            Tensor grad_latent = _decoder->backward(grad);
            _encoder->backward(grad_latent);
            Print_over("Epochs : ", epoch + 1, "/", epochs," iteration : ", id_it + 1, "/", X_split.size()," loss train : ", loss_moy / (id_it + 1));
        }

        _train_loss_history.push_back(loss_moy / X_split.size());
        Print("");
    }
}

std::vector<float>& Auto_Encoder::get_history(){
    return _train_loss_history;
}