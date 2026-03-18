#include "model/Tool/Shape.h"

int Shape::len() const {return dims.size();}

size_t& Shape::operator[](size_t i) {
    return dims[i];
}

const size_t& Shape::operator[](size_t i) const {
    return dims[i];
}

std::string Shape::print() const{
    std::string txt="(";
    int t = len();
    if(t>1){    
        for(int i=0;i<t-1;i++){
            txt+=std::to_string(dims[i])+",";
        }
        txt+=std::to_string(dims[t-1])+")";
    }
    else if(t==1){
        txt+=std::to_string(dims[0])+")";
    }
    else{
        txt+=")";
    }
    return txt; 
}
