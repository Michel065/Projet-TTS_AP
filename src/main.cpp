#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "tool/print.h"
#include "model/tool_model.h"


int main() {
    Print("Debut Code ...");

    Model model = Model("test");

    model.print();

    Print("Fin Code.");
    return 0;
}