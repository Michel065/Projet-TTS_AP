#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "tool/Print.h"
#include "model/Model.h"
#include "model/Layer_dense.h"
#include "model/Struct.h"


int main() {

    //Model model = Model("test");
    
    LayerDense l = LayerDense(10);


    Tensor a(Shape{{2, 3}});
    Tensor b(Shape{{2, 3}});

    auto c = a.recup_data() + b.recup_data();
    std::cout << c << std::endl;

    Shape taille({5});
    l.set_input_shape(taille);

    return 0;
}