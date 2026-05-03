#pragma once
#include <vector>
#include "outil/Print.h"

class Model;
class Auto_Encoder;

class Callback {
public:
    void set_Model(Model* model);
    void set_Model_auto_encoder(Auto_Encoder* model); // ajout de derniere minute

    virtual void on_epoch_end() = 0;
    virtual ~Callback() = default;
protected:
    Model* _model = nullptr;
    Auto_Encoder* _model_AE = nullptr;
};