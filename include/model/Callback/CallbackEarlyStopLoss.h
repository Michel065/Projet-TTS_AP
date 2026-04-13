#include "model/Callback/Callback.h"

struct EarlyStopConfig {
    float epsilon = 0.001f;
    int patience = 5;
};

class CallbackEarlyStopLoss : public Callback {
public:
    CallbackEarlyStopLoss(EarlyStopConfig config): epsilon(config.epsilon), patience(config.patience) {}
    void on_epoch_end() override;

private:
    float epsilon;
    int patience;
};