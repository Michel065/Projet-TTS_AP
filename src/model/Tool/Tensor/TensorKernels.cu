#include "model/Tool/Tensor/TensorKernels.cuh"
#include "outil/Print.h"
#include <cuda_runtime.h>
#include <string>

// init de ce qui est obligatoire
constexpr int THREADS_PAR_BLOCK = 256;
int calculs_blocks(size_t n){
    if(n <= 0){
        Throw_Error("val de size de nos data gpu == 0, Impossible");
    }
    return static_cast<int>((n + THREADS_PAR_BLOCK - 1) / THREADS_PAR_BLOCK);
}

//verif pour debug conseillé
void cuda_check(cudaError_t err, const char* source, const char* erreur){
    if(err != cudaSuccess){
        Throw_Error(source," ",erreur, " : ", cudaGetErrorString(err));
    }
}

void cuda_check_all(const char* source){
    cuda_check(cudaGetLastError(), source, "launch failed");
    cuda_check(cudaDeviceSynchronize(), source, "sync failed");
}




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

__global__ void add_kernel(float* a, const float scalar, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] += scalar;
    }
}

__global__ void sub_kernel(float* a, const float scalar, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] -= scalar;
    }
}

__global__ void mul_kernel(float* a, const float scalar, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] *= scalar;
    }
}

__global__ void div_kernel(float* a, const float scalar, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        a[i] /= scalar;
    }
}








// fonction qui execute nos kernel raison de leurs presence ne compile pas hors cu
void gpu_add(float* a, const float* b, size_t n){
    int blocks = calculs_blocks(n);

    add_kernel<<<blocks, THREADS_PAR_BLOCK>>>(a, b, n);
    cuda_check_all("add_kernel");
}

void gpu_sub(float* a, const float* b, size_t n){
    int blocks = calculs_blocks(n);

    sub_kernel<<<blocks, THREADS_PAR_BLOCK>>>(a, b, n);
    cuda_check_all("sub_kernel");
}

void gpu_mul(float* a, const float* b, size_t n){
    int blocks = calculs_blocks(n);

    mul_kernel<<<blocks, THREADS_PAR_BLOCK>>>(a, b, n);
    cuda_check_all("mub_kernel");
}

void gpu_div(float* a, const float* b, size_t n){
    int blocks = calculs_blocks(n);
    div_kernel<<<blocks, THREADS_PAR_BLOCK>>>(a, b, n);
    cuda_check_all("div_kernel");
}

void gpu_add(float* a, const float scalar, size_t n){
    int blocks = calculs_blocks(n);

    add_kernel<<<blocks, THREADS_PAR_BLOCK>>>(a, scalar, n);
    cuda_check_all("add_kernel_scalar");
}

void gpu_sub(float* a, const float scalar, size_t n){
    int blocks = calculs_blocks(n);

    sub_kernel<<<blocks, THREADS_PAR_BLOCK>>>(a, scalar, n);
    cuda_check_all("sub_kernel_scalar");
}

void gpu_mul(float* a, const float scalar, size_t n){
    int blocks = calculs_blocks(n);

    mul_kernel<<<blocks, THREADS_PAR_BLOCK>>>(a, scalar, n);
    cuda_check_all("mub_kernel_scalar");
}

void gpu_div(float* a, const float scalar, size_t n){
    if(scalar == 0){
        Throw_Error("Division TensorDataGPU impossible scalar == 0.");
    }
    
    int blocks = calculs_blocks(n);
    div_kernel<<<blocks, THREADS_PAR_BLOCK>>>(a, scalar, n);
    cuda_check_all("div_kernel_scalar");
}