#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


int test() {
    std::cout << "Hello world" << std::endl;

    std::ifstream file("sample.lab");
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        long start, end;
        std::string phoneme;

        ss >> start >> end >> phoneme;

        size_t a = phoneme.find('-');
        size_t b = phoneme.find('+');

        phoneme = phoneme.substr(a+1, b-a-1);
        
        std::cout <<" start:"<< start<<" end:"<<end<<" phoneme:" <<phoneme << std::endl;
    }
    std::cout << "Fin" << std::endl;

    return 0;
}