#include "model/Callback/Callback.h"
/*
struct EarlyStopConfig {  ca a reflechir trop bien pour chosir quelle param a implementer
    float epsilon = 0.0001f;
    int patience = 3;
};
/*
CallbackEarlyStopLoss(EarlyStopConfig config)
    : epsilon(config.epsilon), patience(config.patience) {}

    et du coup : CallbackEarlyStopLoss({.patience = 5});
*/


class CallbackEarlyStopLoss : public Callback {
public:
    CallbackEarlyStopLoss(int patience=5, float epsilon=0.001f): epsilon(epsilon), patience(patience) {}
    void on_epoch_end() override;

private:
    float epsilon;
    int patience;
};