#include "model/Tool/Tensor/TensorFuncKernels.cuh"
#include <string>



// fonction qui execute nos kernel raison de leurs presence ne compile pas hors cu
void gpu_add(float* a, const float* b, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);

    add_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, b, n);
    cuda_check_all("add_kernel");
}

void gpu_sub(float* a, const float* b, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);

    sub_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, b, n);
    cuda_check_all("sub_kernel");
}

void gpu_mul(float* a, const float* b, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);

    mul_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, b, n);
    cuda_check_all("mub_kernel");
}

void gpu_div(float* a, const float* b, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);
    div_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, b, n);
    cuda_check_all("div_kernel");
}

void gpu_add_scalar(float* a, const float scalar, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);

    add_kernel_scalar<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, scalar, n);
    cuda_check_all("add_kernel_scalar");
}

void gpu_sub_scalar(float* a, const float scalar, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);

    sub_kernel_scalar<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, scalar, n);
    cuda_check_all("sub_kernel_scalar");
}

void gpu_mul_scalar(float* a, const float scalar, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);

    mul_kernel_scalar<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, scalar, n);
    cuda_check_all("mub_kernel_scalar");
}

void gpu_div_scalar(float* a, const float scalar, size_t n){
    if(scalar == 0){
        Throw_Error("Division TensorDataGPU impossible scalar == 0.");
    }
    
    int blocks = CudaConfig::calculs_blocks_1D(n);
    div_kernel_scalar<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, scalar, n);
    cuda_check_all("div_kernel_scalar");
}








void gpu_exp(float* a, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);
    exp_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, n);
    cuda_check_all("exp_kernel");
}

void gpu_pow(float* a, float val, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);
    pow_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, val, n);
    cuda_check_all("pow_kernel");
}

void gpu_max(float* a, float val, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);
    max_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, val, n);
    cuda_check_all("max_kernel");
}

void gpu_round(float* a, int decimals, size_t n){
    float factor = 1.0f;
    for(int i = 0; i < decimals; i++){ // on le fait pas dedans c pas opti, et c trop cher en op
        factor *= 10.0f;
    }

    int blocks = CudaConfig::calculs_blocks_1D(n);
    round_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, factor, n);
    cuda_check_all("round_kernel");
}

void gpu_clip(float* a, float b_min, float b_max, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);
    clip_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, b_min, b_max, n);
    cuda_check_all("clip_kernel");
}

void gpu_log(float* a, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);
    log_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, n);
    cuda_check_all("log_kernel");
}

void gpu_sup(float* a, float scalar, size_t n){
    int blocks = CudaConfig::calculs_blocks_1D(n);
    sup_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(a, scalar, n);
    cuda_check_all("sup_kernel");
}

void gpu_transpose(float* dest, const float* source, int rows, int cols, int batch){
    dim3 blocks = CudaConfig::calculs_blocks_2D(cols, rows,batch);
    transpose_kernel<<<blocks, CudaConfig::threads_per_block_2D()>>>(dest, source, rows, cols, batch);
    cuda_check_all("transpose_kernel");
}

void gpu_matmul(float* dest, const float* source_a, const float* source_b, int rows, int trans, int cols){
    dim3 blocks = CudaConfig::calculs_blocks_2D(cols, rows);

    matmul_kernel_shared<<<blocks, CudaConfig::threads_per_block_2D()>>>(dest, source_a, source_b, rows, trans, cols);
    cuda_check_all("matmul_kernel");
}

void gpu_shuffle_axis0(float* dest, const float* src, const int* indices, int axis0_size, int stride){
    int total = axis0_size * stride;
    int blocks = CudaConfig::calculs_blocks_1D(total);
    shuffle_axis0_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(dest, src, indices, stride, total);
    cuda_check_all("shuffle_axis0_kernel");
}

void gpu_extraction_section_axe_0(float* dest,const float* src,int debut,int fin,int stride){
    int total = (fin - debut) * stride;
    int blocks = CudaConfig::calculs_blocks_1D(total);
    extraction_section_axe_0_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(dest, src, debut, stride,total);
    cuda_check_all("extraction_section_axe_0_kernel");
}

void gpu_sum_per_row(float* dest, const float* src, int rows, int cols){
    int blocks = CudaConfig::calculs_blocks_1D(rows);
    sum_per_row_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(dest, src, rows, cols);
    cuda_check_all("sum_per_row_kernel");
}

void gpu_max_per_row(float* dest, const float* src, int rows, int cols){
    int blocks = CudaConfig::calculs_blocks_1D(rows);
    max_per_row_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(dest, src, rows, cols);
    cuda_check_all("max_per_row_kernel");
}

void gpu_sum_axis0(float* dest, const float* src, int rows, int cols){
    int blocks = CudaConfig::calculs_blocks_1D(cols);
    sum_axis0_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(dest, src, rows, cols);
    cuda_check_all("sum_axis0_kernel");
}











//methode boradcast dim 0
void gpu_add_broadcast_axis0(float* dest, const float* src, int nbr_broadcast, int stride){
    int total = nbr_broadcast * stride;
    int blocks = CudaConfig::calculs_blocks_1D(total);
    add_broadcast_axis0_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(dest, src, total, stride);
    cuda_check_all("add_broadcast_axis0_kernel");
}

void gpu_sub_broadcast_axis0(float* dest, const float* src, int nbr_broadcast, int stride){
    int total = nbr_broadcast * stride;
    int blocks = CudaConfig::calculs_blocks_1D(total);
    sub_broadcast_axis0_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(dest, src, total, stride);
    cuda_check_all("sub_broadcast_axis0_kernel");
}

void gpu_mul_broadcast_axis0(float* dest, const float* src, int nbr_broadcast, int stride){
    int total = nbr_broadcast * stride;
    int blocks = CudaConfig::calculs_blocks_1D(total);
    mul_broadcast_axis0_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(dest, src, total, stride);
    cuda_check_all("mul_broadcast_axis0_kernel");
}

void gpu_div_broadcast_axis0(float* dest, const float* src, int nbr_broadcast, int stride){
    int total = nbr_broadcast * stride;
    int blocks = CudaConfig::calculs_blocks_1D(total);
    div_broadcast_axis0_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(dest, src, total, stride);
    cuda_check_all("div_broadcast_axis0_kernel");
}




// matmul version broadcast
void gpu_broadcast_matmul(float* dest, const float* source_a, const float* source_b, int batch, int rows, int trans, int cols, bool batch_on_a){
    dim3 blocks = CudaConfig::calculs_blocks_2D(cols, rows, batch);

    broadcast_matmul_kernel_shared<<<blocks, CudaConfig::threads_per_block_2D()>>>(dest, source_a, source_b, batch, rows, trans, cols, batch_on_a);
    cuda_check_all("matmul_kernel_shared_broadcast_B");
}

void gpu_broadcast_all_matmul(float* dest,const float* source_a,const float* source_b,int batch,int rows,int trans,int cols){
    dim3 blocks = CudaConfig::calculs_blocks_2D(cols, rows, batch);

    broadcast_all_matmul_kernel_shared<<<blocks,CudaConfig::threads_per_block_2D()>>>(dest, source_a, source_b, batch, rows, trans, cols);
    cuda_check_all("broadcast_all_matmul_kernel_shared");
}







// brod casr de subsur l'axe 1 pour le softmax
void gpu_sub_broadcast_axis1(float* dest, const float* src, int rows, int cols){
    int total = rows * cols;
    int blocks = CudaConfig::calculs_blocks_1D(total);
    sub_broadcast_axis1_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(dest, src, total, cols);
    cuda_check_all("sub_broadcast_axis1_kernel");
}

void gpu_div_broadcast_axis1(float* dest, const float* src, int rows, int cols){
    int total = rows * cols;
    int blocks = CudaConfig::calculs_blocks_1D(total);
    div_broadcast_axis1_kernel<<<blocks, CudaConfig::THREADS_PER_BLOCK_1D>>>(dest, src, total, cols);
    cuda_check_all("div_broadcast_axis1_kernel");
}