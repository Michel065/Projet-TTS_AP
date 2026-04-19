#include "model/Layer_conv/LayerConv2D/LayerConv2DKernels.cuh"
#include "outil/Print.h"

//SOURCE : https://github.com/piojanu/CUDA-im2col-conv/blob/master/im2col.cu

__global__ void im2col_kernel(float* Dest, const float* Source, size_t Batch, size_t Height, size_t Width, size_t Channel, size_t Kernel, size_t pad, size_t rows, size_t cols){
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col_in_batch = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= rows || col_in_batch >= Batch * cols) return;

    size_t img_size = Height * Width; // valide car on fait un pad same!!!
    // calcul du batch
    size_t b = col_in_batch / img_size;
    size_t col = col_in_batch % img_size; // calcul col local
    // corespondance de la case source dans l'origine avant kernel
    size_t out_y = col / Width;
    size_t out_x = col % Width;

    size_t c  = row / (Kernel * Kernel); // on recup le channel
    size_t kk = row % (Kernel * Kernel); // on retrouve la valeur reel de la pos dans le kernel
    // on convertie en vrai coords du kernel
    size_t ky = kk / Kernel;
    size_t kx = kk % Kernel;
    // on applique pour trouvé le pixel reel source
    int src_y = int(out_y) + int(ky) - int(pad);
    int src_x = int(out_x) + int(kx) - int(pad);

    if(src_y >= 0 && src_y < (int)Height && src_x >= 0 && src_x < (int)Width){
        size_t src_idx = b * (Channel * Height * Width) + c * (Height * Width) + (size_t)src_y * Width + (size_t)src_x;
        size_t dest_idx = (b * rows + row) * cols + col;
        Dest[dest_idx] = Source[src_idx];
    }
}

__global__ void add_bias_conv_kernel(float* output, const float* bias, size_t batch, size_t nb_filters, size_t height, size_t width){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t total = batch * nb_filters * height * width;
    if(idx >= total) return;

    size_t hw = height * width;
    size_t f = (idx / hw) % nb_filters;
    output[idx] += bias[f];
}









// modifier pr rapport a ca version vue que je vais l'utilier pour le grad je peux pas perdre des valeurs
__global__ void col2im_kernel(float* Dest, const float* Source, size_t Batch, size_t Height, size_t Width,  size_t Channel, size_t Kernel, size_t pad, size_t rows, size_t cols){
    size_t col_in_batch = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row >= rows || col_in_batch >= Batch * cols) return;
    
    size_t img_size = Height * Width;
    size_t b = col_in_batch / img_size;
    size_t col = col_in_batch % img_size;

    size_t out_y = col / Width;
    size_t out_x = col % Width;

    size_t c  = row / (Kernel * Kernel);
    size_t kk = row % (Kernel * Kernel);
    size_t ky = kk / Kernel;
    size_t kx = kk % Kernel;

    int src_y = int(out_y) + int(ky) - int(pad);
    int src_x = int(out_x) + int(kx) - int(pad);

    if(src_y >= 0 && src_y < (int)Height && src_x >= 0 && src_x < (int)Width){
        size_t src_idx = b * (Channel * Height * Width) + c * (Height * Width) + (size_t)src_y * Width + (size_t)src_x;
        size_t dest_idx = (b * rows + row) * cols + col;
        atomicAdd(&Dest[src_idx], Source[dest_idx]); // on inverse juste
    }
}

__global__ void sum_bias_conv_kernel(float* grad_b, const float* grad_input, size_t batch, size_t nb_filters, size_t height, size_t width){
    size_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if(f >= nb_filters) return;
    size_t hw = height * width;
    float sum = 0.0f;
    for(size_t b = 0; b < batch; b++){
        size_t base = (b * nb_filters + f) * hw;
        for(size_t i = 0; i < hw; i++){
            sum += grad_input[base + i];
        }
    }
    grad_b[f] = sum;
}


__global__ void sum_batch_kernel(float* dest, const float* source, size_t batch, size_t rows, size_t cols){
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows || col >= cols) return;
    float sum = 0.0f;
    for(size_t b = 0; b < batch; b++){
        size_t idx = (b * rows + row) * cols + col;
        sum += source[idx];
    }
    dest[row * cols + col] = sum;
}