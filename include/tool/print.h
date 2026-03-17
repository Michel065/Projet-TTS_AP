#pragma once

#include <iostream>
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

template<typename... Args>
inline void Print_Color(Color color, const Args&... args) {
    std::cout << get_color(color);
    (std::cout << ... << args);
    std::cout << "\033[0m" << std::endl;
}

template<typename... Args>
inline void Print(const Args&... args) {
    Print_Color(Color::DEFAULT,args...);
}