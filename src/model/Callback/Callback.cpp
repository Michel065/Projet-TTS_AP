#include "model/Callback/Callback.h"

void Callback::set_Model(Model* model){
    _model = model;
}

void Callback::set_Model_auto_encoder(Auto_Encoder* model){
    _model_AE = model;
}