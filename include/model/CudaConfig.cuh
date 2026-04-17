#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include "outil/Print.h"

inline void cuda_check(cudaError_t err, const char* msg){
    if(err != cudaSuccess){
        Throw_Error(std::string(msg)," : ",cudaGetErrorString(err));
    }
}

inline void cuda_check(cudaError_t err, const char* source, const char* erreur){
    if(err != cudaSuccess){
        Throw_Error(source," ",erreur, " : ", cudaGetErrorString(err));
    }
}

inline void cuda_check_all(const char* source){
    cuda_check(cudaGetLastError(), source, "launch failed");
    cuda_check(cudaDeviceSynchronize(), source, "sync failed");
}





namespace CudaConfig {
    constexpr int THREADS_PER_BLOCK_1D = 512;
    constexpr unsigned int THREADS_2D = 16;

    inline dim3 threads_per_block_2D(){
        return dim3(THREADS_2D, THREADS_2D);
    }

    inline int calculs_blocks_1D(size_t n){
        if(n == 0){
            Throw_Error("calculs_blocks_1D n == 0, impossible");
        }
        return static_cast<int>((n + THREADS_PER_BLOCK_1D - 1) / THREADS_PER_BLOCK_1D);
    }

    inline dim3 calculs_blocks_2D(size_t cols, size_t rows){
        if(cols == 0 || rows == 0){
            Throw_Error("calculs_blocks_2D cols ou rows == 0, impossible");
        }

        return dim3(static_cast<unsigned int>((cols + THREADS_2D - 1) / THREADS_2D),static_cast<unsigned int>((rows + THREADS_2D - 1) / THREADS_2D));
    }
}

namespace CudaMatmulConfig {
    constexpr int BLOCKSIZE = 16; //CudaConfig::THREADS_2D; // normalement
}
