#pragma once
#include <string>
#include <vector>

#include "outil/Print.h"

struct Shape {
    std::vector<size_t> dims;

    Shape(){}
    Shape(std::vector<size_t> data): dims(data){}
    int len() const;
    int size() const;
    size_t& operator[](size_t i);
    const size_t& operator[](size_t i) const;
    std::string print() const;
};