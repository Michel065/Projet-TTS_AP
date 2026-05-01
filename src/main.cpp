#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "outil/Print.h"
#include "test.h"
#include "model/Model.h"
#include "model/Callback/CallbackEarlyStopLoss.h"

// imports des base pour nos models
#include "model/Layer_ALL.h"
#include "model/Loss/Loss_ALL.h"

int main() {
    test_UpSampling();
    
    return 0;
}