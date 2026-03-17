#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "tool/Print.h"
#include "model/Model.h"
#include "model/Layer_dense.h"
#include "model/Struct.h"

int main() {

    Tensor M1(Shape({2,3}));
    Tensor M2(Shape({3,2}));

    M1.init_alea();
    M2.init_alea();

    Print(M1);
    Print(M2);

    Tensor M3 = M1.prod_mat(M2);
    Print(M3);

    return 0;
}