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

__global__ void transpose_kernel(float* dest, const float* source, int rows, int cols){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < rows && j < cols){
        dest[j * rows + i] = source[i * cols + j];
    }
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
__global__ void matmul_kernel_shared(float* dest, const float* source_a, const float* source_b, int rows, int trans, int cols){
    __shared__ float As[CudaMatmulConfig::BLOCKSIZE * CudaMatmulConfig::BLOCKSIZE];
    __shared__ float Bs[CudaMatmulConfig::BLOCKSIZE * CudaMatmulConfig::BLOCKSIZE];

    int threadCol = threadIdx.x;
    int threadRow = threadIdx.y;
    int cCol = blockIdx.x;
    int cRow = blockIdx.y;

    int row = cRow * CudaMatmulConfig::BLOCKSIZE + threadRow;
    int col = cCol * CudaMatmulConfig::BLOCKSIZE + threadCol;

    const float* A = source_a;
    const float* B = source_b;
    float* C = dest;

    A += cRow * CudaMatmulConfig::BLOCKSIZE * trans;
    B += cCol * CudaMatmulConfig::BLOCKSIZE;
    C += cRow * CudaMatmulConfig::BLOCKSIZE * cols + cCol * CudaMatmulConfig::BLOCKSIZE;

    float tmp = 0.0f;
    int nb_blocks = (trans + CudaMatmulConfig::BLOCKSIZE - 1) / CudaMatmulConfig::BLOCKSIZE;
    for(int bkIdx = 0; bkIdx < nb_blocks; bkIdx++){
        int a_col = bkIdx * CudaMatmulConfig::BLOCKSIZE + threadCol;
        int b_row = bkIdx * CudaMatmulConfig::BLOCKSIZE + threadRow;

        if(row < rows && a_col < trans){
            As[threadRow * CudaMatmulConfig::BLOCKSIZE + threadCol] = A[threadRow * trans + threadCol];
        }else{
            As[threadRow * CudaMatmulConfig::BLOCKSIZE + threadCol] = 0.0f;
        }

        if(b_row < trans && col < cols){
            Bs[threadRow * CudaMatmulConfig::BLOCKSIZE + threadCol] = B[threadRow * cols + threadCol];
        }else{
            Bs[threadRow * CudaMatmulConfig::BLOCKSIZE + threadCol] = 0.0f;
        }

        __syncthreads();
        for(int dotIdx = 0; dotIdx < CudaMatmulConfig::BLOCKSIZE; dotIdx++){
            tmp += As[threadRow * CudaMatmulConfig::BLOCKSIZE + dotIdx] * Bs[dotIdx * CudaMatmulConfig::BLOCKSIZE + threadCol];
        }
        __syncthreads();

        A += CudaMatmulConfig::BLOCKSIZE;
        B += CudaMatmulConfig::BLOCKSIZE * cols;
    }

    if(row < rows && col < cols){
        C[threadRow * cols + threadCol] = tmp;
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





