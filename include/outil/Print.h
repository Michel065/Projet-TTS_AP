#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <string>

enum class Color {
    DEFAULT,
    GREEN,
    ORANGE,
    PINK,
    RED,
    PURPLE
};

// Couleurs ANSI
inline const char* get_color(Color c) {
    switch (c) {
        case Color::GREEN:  return "\033[32m";
        case Color::ORANGE: return "\033[38;5;208m";
        case Color::PINK:   return "\033[35m";
        case Color::RED:    return "\033[31m";
        case Color::PURPLE: return "\033[36m";
        default:            return "\033[0m";
    }
}

inline std::ostream& operator<<(std::ostream& os, Color color){// test pour faciliter le print
    return os << get_color(color);
}

template<typename... Args>
inline void Print_Color(Color color, const Args&... args) {
    std::cout << color;
    (std::cout << ... << args);
    std::cout << Color::DEFAULT << std::endl;
}

template<typename... Args>
inline void Print(const Args&... args) {
    Print_Color(Color::DEFAULT,args...);
}

template<typename... Args>
inline void Print_over(const Args&... args) {
    std::cout << Color::DEFAULT<<"\r";
    (std::cout << ... << args);
    std::cout << Color::DEFAULT << std::flush;
}

template<typename... Args>
inline void Throw_Error_Color(Color color, const Args&... args){
    std::ostringstream oss;
    oss<<color<<"Erreur: ";
    (oss << ... << args);
    oss<<Color::DEFAULT;
    const std::string msg = oss.str();
    throw std::runtime_error(msg);
}

template<typename... Args>
inline void Throw_Error(const Args&... args) {
    Throw_Error_Color(Color::RED,args...);
}