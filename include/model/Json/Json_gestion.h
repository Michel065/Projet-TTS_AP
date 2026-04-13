#pragma once
#include "outil/Print.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

//https://stackoverflow.com/questions/9975672/c-automatic-factory-registration-of-derived-types

class Layer;

class LayerConstructorListe {
private:
    inline static std::unordered_map<std::string, std::function<Layer*()>> liste_des_constructeurs;

public:
    static void ajoute_layer_construct(const std::string& name, std::function<Layer*()> creator) {
        liste_des_constructeurs[name] = creator;
    }

    static Layer* create(const std::string& name) {
        for(auto& constructeur_i : liste_des_constructeurs){
            if(constructeur_i.first == name)
                return constructeur_i.second();
        }
        Throw_Error("Couche inconnue : ", name);
        return nullptr;
    }
};

template<typename T>
struct AutoRegisterLayer {
    AutoRegisterLayer() {
        T tmp;
        LayerConstructorListe::ajoute_layer_construct(tmp.get_name(), []() { return new T(); });
    }
};