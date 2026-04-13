#pragma once
#include <string>
#include <vector>

#include "outil/Print.h"
#include "model/Json/Json_gestion.h"

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

inline void from_json(const json& j, Shape& shape) {
    shape = Shape(j.at("shape").get<std::vector<size_t>>());
}

inline void to_json(json& j, const Shape& shape) {
    j = {{"shape", shape.dims}};
}