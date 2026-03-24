#pragma once
#include <vector>
#include "outil/Print.h"

class Model;

class Callback {
public:
    void set_Model(Model* model);

    virtual void on_epoch_end() = 0;
    virtual ~Callback() = default;
protected:
    Model* _model = nullptr;
};