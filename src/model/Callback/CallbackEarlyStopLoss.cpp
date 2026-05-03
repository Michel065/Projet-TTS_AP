#include "model/Callback/CallbackEarlyStopLoss.h"
#include "model/Model.h"
#include "model/Auto_Encoder.h"

std::vector<float>& CallbackEarlyStopLoss::get_losss(){
    if(_model != nullptr){
        return _model->get_history();
    }
    else if(_model_AE != nullptr){
        return _model_AE->get_history();
    }
    Throw_Error("Model non definie (CallbackEarlyStopLoss)");
    static std::vector<float> vide = {0.0f}; // c pas bien mais c rapide
    return vide;
}

void CallbackEarlyStopLoss::stop_trainning(){
    if(_model != nullptr){
        _model->early_stop = true;
        return;
    }
    else if(_model_AE != nullptr){
        _model_AE->early_stop = true;
        return;
    }
    Throw_Error("Model non definie (CallbackEarlyStopLoss)");
}

void CallbackEarlyStopLoss::on_epoch_end(){
    std::vector<float>& loss_history = get_losss();

    if(loss_history.size() < (size_t)(patience + 1))
        return; // pas assez de val on coupe

    int debut = (int)loss_history.size() - patience - 1;
    for(int i = debut; i < (int)loss_history.size() - 1; i++){
        float delta = std::abs( loss_history[i+1] - loss_history[i] );
        if(delta > epsilon)
            return;//on coupe que si c pas bon
    }
    stop_trainning();//si on est la c'est que on a plus de patience
}