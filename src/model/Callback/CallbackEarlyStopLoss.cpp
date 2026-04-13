#include "model/Callback/CallbackEarlyStopLoss.h"
#include "model/Model.h"

void CallbackEarlyStopLoss::on_epoch_end(){
    if(_model == nullptr){
        Throw_Error("Model non definie (CallbackEarlyStopLoss)");

    }
    std::vector<float>& loss_history = _model->get_history();

    if(loss_history.size() < (size_t)(patience + 1))
        return; // pas assez de val on coupe

    int debut = (int)loss_history.size() - patience - 1;
    for(int i = debut; i < (int)loss_history.size() - 1; i++){
        float delta = std::abs( loss_history[i+1] - loss_history[i] );
        if(delta > epsilon)
            return;//on coupe que si c pas bon
    }
    _model->early_stop = true;//si on est la c'est que on a plus de patience
}