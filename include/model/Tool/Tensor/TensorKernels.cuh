#pragma once
#include <cstddef>

void gpu_add(float* a, const float* b, size_t n);
void gpu_sub(float* a, const float* b, size_t n);
void gpu_mul(float* a, const float* b, size_t n);
void gpu_div(float* a, const float* b, size_t n);