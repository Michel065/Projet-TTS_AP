#include "model/Tool/Tensor/TensorKernels.cuh"
#include "outil/Print.h"

// fonction kernel qui depande des threads blocks
__global__ void add_kernel(float* a, const float* b, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] += b[i];
    }
}

__global__ void sub_kernel(float* a, const float* b, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] -= b[i];
    }
}

__global__ void mul_kernel(float* a, const float* b, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] *= b[i];
    }
}

__global__ void div_kernel(float* a, const float* b, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] /= b[i];
    }
}




__global__ void add_kernel_scalar(float* a, const float scalar, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] += scalar;
    }
}

__global__ void sub_kernel_scalar(float* a, const float scalar, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] -= scalar;
    }
}

__global__ void mul_kernel_scalar(float* a, const float scalar, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] *= scalar;
    }
}

__global__ void div_kernel_scalar(float* a, const float scalar, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] /= scalar;
    }
}











__global__ void exp_kernel(float* a,size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] = expf(a[i]);
    }
}

__global__ void pow_kernel(float* a, float val, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] = powf(a[i], val);
    }
}

__global__ void max_kernel(float* a, float val, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] = fmaxf(a[i], val);
    }
}

__global__ void round_kernel(float* a, float factor, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] = roundf(a[i] * factor) / factor;
    }
}



__global__ void clip_kernel(float* a, float b_min, float b_max, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] = fminf(fmaxf(a[i], b_min), b_max); // fait avec des methodes sinon on perd en perf, si j'ai bien compris
    }
}

__global__ void log_kernel(float* a, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] = logf(a[i]);
    }
}

__global__ void sup_kernel(float* a, float scalar, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] = (a[i] > scalar) ? 1.0f : 0.0f;
    }
}

__global__ void transpose_kernel(float* dest, const float* source, int rows, int cols, int batch){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;

    if(b >= batch || i >= rows || j >= cols) return;

    const float* src = source + b * rows * cols; // modification pour supporte un batch pour la conv2d
    float* dst = dest + b * rows * cols;
    dst[j * rows + i] = src[i * cols + j];
}








// version de base a moi
__global__ void matmul_kernel(float* dest, const float* source_a, const float* source_b, int rows, int trans, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row >= rows || col >= cols){
        return;
    }

    float sum = 0.0f;
    for(int k = 0; k < trans; k++){
        sum += source_a[row * trans + k] * source_b[k * cols + col];
    }

    dest[row * cols + col] = sum;
}



//version 3 du site https://siboehm.com/articles/22/CUDA-MMM (pourquoi pas la meuilleur des versions proposé, la raison est simple j'ai pas compris)
__global__ void matmul_kernel_shared(float* dest,const float* source_a,const float* source_b,int rows,int trans,int cols){
    __shared__ float As[CudaMatmulConfig::BLOCKSIZE * CudaMatmulConfig::BLOCKSIZE];
    __shared__ float Bs[CudaMatmulConfig::BLOCKSIZE * CudaMatmulConfig::BLOCKSIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_col = blockIdx.x;
    int block_row = blockIdx.y;

    int row = block_row * CudaMatmulConfig::BLOCKSIZE + ty;
    int col = block_col * CudaMatmulConfig::BLOCKSIZE + tx;

    float sum = 0.0f;
    int nb_tiles = (trans + CudaMatmulConfig::BLOCKSIZE - 1) / CudaMatmulConfig::BLOCKSIZE;

    for(int tile = 0; tile < nb_tiles; tile++){
        int a_col = tile * CudaMatmulConfig::BLOCKSIZE + tx;
        int b_row = tile * CudaMatmulConfig::BLOCKSIZE + ty;

        if(row < rows && a_col < trans){
            As[ty * CudaMatmulConfig::BLOCKSIZE + tx] = source_a[row * trans + a_col];
        }else{
            As[ty * CudaMatmulConfig::BLOCKSIZE + tx] = 0.0f;
        }

        if(b_row < trans && col < cols){
            Bs[ty * CudaMatmulConfig::BLOCKSIZE + tx] = source_b[b_row * cols + col];
        }else{
            Bs[ty * CudaMatmulConfig::BLOCKSIZE + tx] = 0.0f;
        }

        __syncthreads();
        for(int k = 0; k < CudaMatmulConfig::BLOCKSIZE; k++){
            sum += As[ty * CudaMatmulConfig::BLOCKSIZE + k]
                 * Bs[k * CudaMatmulConfig::BLOCKSIZE + tx];
        }
        __syncthreads();
    }

    if(row < rows && col < cols){
        dest[row * cols + col] = sum;
    }
}




__global__ void shuffle_axis0_kernel(float* dest, const float* src, const int* indices, int stride, int total){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= total)
        return;
    int row = id / stride;
    int offset = id % stride;
    int src_row = indices[row];
    dest[row * stride + offset] = src[src_row * stride + offset];
}

__global__ void extraction_section_axe_0_kernel(float* dest,const float* src,int debut,int stride,int total){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= total)
        return;
    int row = id / stride;
    int offset = id % stride;
    int src_row = debut + row;
    dest[row * stride + offset] = src[src_row * stride + offset];
}

__global__ void sum_per_row_kernel(float* dest, const float* src, int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows)
        return;

    float sum = 0.0f;
    for(int j = 0; j < cols; j++){
        sum += src[row * cols + j];
    }
    dest[row] = sum;
}

__global__ void max_per_row_kernel(float* dest, const float* src, int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows)
        return;

    float max_val = src[row * cols];
    for(int j = 1; j < cols; j++){
        float v = src[row * cols + j];
        if(v > max_val)
            max_val = v;
    }
    dest[row] = max_val;
}

__global__ void sum_axis0_kernel(float* dest, const float* src, int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= cols)
        return;

    float sum = 0.0f;
    for(int i = 0; i < rows; i++){
        sum += src[i * cols + col];
    }
    dest[col] = sum;
}






















// le broadcast pour dim 0
__global__ void add_broadcast_axis0_kernel(float* dest, const float* src, int total, int stride){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= total)
        return;
    int offset = id % stride;
    dest[id] += src[offset];
}

__global__ void sub_broadcast_axis0_kernel(float* dest, const float* src, int total, int stride){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= total)
        return;
    int offset = id % stride;
    dest[id] -= src[offset];
}

__global__ void mul_broadcast_axis0_kernel(float* dest, const float* src, int total, int stride){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= total)
        return;
    int offset = id % stride;
    dest[id] *= src[offset];
}

__global__ void div_broadcast_axis0_kernel(float* dest, const float* src, int total, int stride){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= total)
        return;
    int offset = id % stride;
    dest[id] /= src[offset];
}












// matmul version broadcast
__global__ void broadcast_matmul_kernel_shared(float* dest,const float* source_a,const float* source_b,int batch,int rows,int trans,int cols,bool batch_on_a){
    __shared__ float As[CudaMatmulConfig::BLOCKSIZE * CudaMatmulConfig::BLOCKSIZE];
    __shared__ float Bs[CudaMatmulConfig::BLOCKSIZE * CudaMatmulConfig::BLOCKSIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_col = blockIdx.x;
    int block_row = blockIdx.y;
    int b = blockIdx.z;

    if(b >= batch) return;

    int row = block_row * CudaMatmulConfig::BLOCKSIZE + ty;
    int col = block_col * CudaMatmulConfig::BLOCKSIZE + tx;

    const float* A_base = source_a + (batch_on_a ? b * rows * trans : 0);
    const float* B_base = source_b + (batch_on_a ? 0 : b * trans * cols);
    float* C_base = dest + b * rows * cols;

    float sum = 0.0f;
    int nb_tiles = (trans + CudaMatmulConfig::BLOCKSIZE - 1) / CudaMatmulConfig::BLOCKSIZE;
    for(int tile = 0; tile < nb_tiles; tile++){
        int a_col = tile * CudaMatmulConfig::BLOCKSIZE + tx;
        int b_row = tile * CudaMatmulConfig::BLOCKSIZE + ty;

        if(row < rows && a_col < trans){
            As[ty * CudaMatmulConfig::BLOCKSIZE + tx] = A_base[row * trans + a_col];
        }else{
            As[ty * CudaMatmulConfig::BLOCKSIZE + tx] = 0.0f;
        }

        if(b_row < trans && col < cols){
            Bs[ty * CudaMatmulConfig::BLOCKSIZE + tx] = B_base[b_row * cols + col];
        }else{
            Bs[ty * CudaMatmulConfig::BLOCKSIZE + tx] = 0.0f;
        }

        __syncthreads();
        for(int k = 0; k < CudaMatmulConfig::BLOCKSIZE; k++){
            sum += As[ty * CudaMatmulConfig::BLOCKSIZE + k]
                 * Bs[k * CudaMatmulConfig::BLOCKSIZE + tx];
        }
        __syncthreads();
    }
    if(row < rows && col < cols){
        C_base[row * cols + col] = sum;
    }
}

//version global avec un batch sur a et b pour un res sur un dernier batch
__global__ void broadcast_all_matmul_kernel_shared(float* dest,const float* source_a,const float* source_b,int batch,int rows,int trans,int cols){
    __shared__ float As[CudaMatmulConfig::BLOCKSIZE * CudaMatmulConfig::BLOCKSIZE];
    __shared__ float Bs[CudaMatmulConfig::BLOCKSIZE * CudaMatmulConfig::BLOCKSIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_col = blockIdx.x;
    int block_row = blockIdx.y;
    int b = blockIdx.z;

    if(b >= batch) return;

    int row = block_row * CudaMatmulConfig::BLOCKSIZE + ty;
    int col = block_col * CudaMatmulConfig::BLOCKSIZE + tx;

    const float* A_base = source_a + b * rows * trans;
    const float* B_base = source_b + b * trans * cols;
    float* C_base = dest + b * rows * cols;

    float sum = 0.0f;
    int nb_tiles = (trans + CudaMatmulConfig::BLOCKSIZE - 1) / CudaMatmulConfig::BLOCKSIZE;

    for(int tile = 0; tile < nb_tiles; tile++){
        int a_col = tile * CudaMatmulConfig::BLOCKSIZE + tx;
        int b_row = tile * CudaMatmulConfig::BLOCKSIZE + ty;

        if(row < rows && a_col < trans){
            As[ty * CudaMatmulConfig::BLOCKSIZE + tx] = A_base[row * trans + a_col];
        }else{
            As[ty * CudaMatmulConfig::BLOCKSIZE + tx] = 0.0f;
        }

        if(b_row < trans && col < cols){
            Bs[ty * CudaMatmulConfig::BLOCKSIZE + tx] = B_base[b_row * cols + col];
        }else{
            Bs[ty * CudaMatmulConfig::BLOCKSIZE + tx] = 0.0f;
        }

        __syncthreads();
        for(int k = 0; k < CudaMatmulConfig::BLOCKSIZE; k++){
            sum += As[ty * CudaMatmulConfig::BLOCKSIZE + k]
                 * Bs[k * CudaMatmulConfig::BLOCKSIZE + tx];
        }
        __syncthreads();
    }
    if(row < rows && col < cols){
        C_base[row * cols + col] = sum;
    }
}







// brod casr de subsur l'axe 1 pour le softmax
__global__ void sub_broadcast_axis1_kernel(float* dest, const float* src, int total, int cols){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= total) return;
    int row = id / cols;
    dest[id] -= src[row];
}

__global__ void div_broadcast_axis1_kernel(float* dest, const float* src, int total, int cols){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= total) return;
    int row = id / cols;
    dest[id] /= src[row];
}