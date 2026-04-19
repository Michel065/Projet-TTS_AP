#pragma once
#include "model/Tool/Tensor/TensorKernels.cuh"

// les ops
void gpu_add(float* a, const float* b, size_t n);
void gpu_sub(float* a, const float* b, size_t n);
void gpu_mul(float* a, const float* b, size_t n);
void gpu_div(float* a, const float* b, size_t n);

void gpu_add_scalar(float* a, const float scalar, size_t n);
void gpu_sub_scalar(float* a, const float scalar, size_t n);
void gpu_mul_scalar(float* a, const float scalar, size_t n);
void gpu_div_scalar(float* a, const float scalar, size_t n);


void gpu_exp(float* a, size_t n);
void gpu_pow(float* a, float val, size_t n);
void gpu_max(float* a, float val, size_t n);
void gpu_round(float* a, int decimals, size_t n);

void gpu_clip(float* a, float b_min, float b_max, size_t n);
void gpu_log(float* a, size_t n);
void gpu_sup(float* a, float scalar, size_t n);
void gpu_transpose(float* dest, const float* source, int rows, int cols, int batch = 1);

void gpu_matmul(float* dest, const float* source_a, const float* source_b, int rows, int trans, int cols);
void gpu_shuffle_axis0(float* dest, const float* src, const int* indices, int axis0_size, int stride);
void gpu_extraction_section_axe_0(float* dest,const float* src,int debut,int fin,int stride);

//methode specifique:
void gpu_sum_per_row(float* dest, const float* src, int rows, int cols);
void gpu_max_per_row(float* dest, const float* src, int rows, int cols);
void gpu_sum_axis0(float* dest, const float* src, int rows, int cols);





//methode boradcast dim 0
void gpu_add_broadcast_axis0(float* dest, const float* src, int nbr_broadcast, int stride);
void gpu_sub_broadcast_axis0(float* dest, const float* src, int nbr_broadcast, int stride);
void gpu_mul_broadcast_axis0(float* dest, const float* src, int nbr_broadcast, int stride);
void gpu_div_broadcast_axis0(float* dest, const float* src, int nbr_broadcast, int stride);




// matmul version broadcast
void gpu_broadcast_matmul(float* dest, const float* source_a, const float* source_b, int batch, int rows, int trans, int cols, bool batch_on_a=false);
void gpu_broadcast_all_matmul(float* dest,const float* source_a,const float* source_b,int batch,int rows,int trans,int cols);



// brod casr de subsur l'axe 1 pour le softmax
void gpu_sub_broadcast_axis1(float* dest, const float* src, int rows, int cols);
void gpu_div_broadcast_axis1(float* dest, const float* src, int rows, int cols);