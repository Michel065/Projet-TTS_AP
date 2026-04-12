#include "model/Tool/Shape.h"

int Shape::len() const {return dims.size();}

size_t& Shape::operator[](size_t i) {
    if(i>=(size_t)len()){
        Throw_Error("Shape dim invalide i:",i," len :",len());
    }
    return dims[i];
}

const size_t& Shape::operator[](size_t i) const {
    if(i>=(size_t)len()){
        Throw_Error("Shape dim invalide i:",i," len :",len());
    }
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

int Shape::size() const{
    int taille=1;
    for(int i=0;i<len();i++){
        taille*=dims[i];
    }
    return taille;
}