#pragma once
#include <string>
#include<vector>

struct Shape {
    std::vector<int> dims;

    Shape(){

    }

    Shape(std::vector<int> data){
        dims=data;
    }

    int len(){
        return dims.size();
    }

    int& operator[](size_t i) {
        return dims[i];
    }

    const int& operator[](size_t i) const {
        return dims[i];
    }


    std::string print(){
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
};