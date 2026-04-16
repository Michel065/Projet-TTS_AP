#include <xtensor/xarray.hpp>
#include <chrono>

#include "test.h"

void get_data_lineaire(Tensor& X, Tensor& y, Tensor& x_test){
    X = Tensor(DeviceType::CPU,{
        {-1.0f, -1.0f},
        {-1.2f, -0.8f},
        {-0.8f, -1.1f},
        {-1.1f, -1.3f},

        { 1.0f,  1.0f},
        { 1.2f,  0.9f},
        { 0.8f,  1.1f},
        { 1.1f,  1.3f}
    });

    y = Tensor(DeviceType::CPU,{
        {1.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 0.0f},

        {0.0f, 1.0f},
        {0.0f, 1.0f},
        {0.0f, 1.0f},
        {0.0f, 1.0f}
    });

    x_test = Tensor(DeviceType::CPU,{
        { 0.9f,  1.0f},
        {-0.9f, -1.0f}
    });
}

void get_data_non_lineaire(Tensor& X, Tensor& y, Tensor& x_test,size_t n,DeviceType device){
    X = Tensor(device,Shape({n, 2}));
    y = Tensor(device,Shape({n, 1}));

    float r1 = 1.0f;
    float r2 = 2.0f;
    size_t n_in = n / 2;
    size_t idx = 0;

    auto rand_float = [](float a, float b){
        return a + ((float)rand() / (float)RAND_MAX) * (b - a);
    };

    while(idx < n_in){
        float x = rand_float(-3.0f, 3.0f);
        float yy = rand_float(-3.0f, 3.0f);
        float dist = std::sqrt(x * x + yy * yy);

        if(dist >= r1 && dist <= r2){
            X.set({idx, 0}, x);
            X.set({idx, 1}, yy);
            y.set({idx, 0}, 1.0f);
            idx++;
        }
    }

    while(idx < n){
        float x = rand_float(-3.0f, 3.0f);
        float yy = rand_float(-3.0f, 3.0f);
        float dist = std::sqrt(x * x + yy * yy);

        if(dist < r1 || dist > r2){
            X.set({idx, 0}, x);
            X.set({idx, 1}, yy);
            y.set({idx, 0}, 0.0f);
            idx++;
        }
    }

    x_test = Tensor(device,{
        {1.5f, 0.0f},
        {1.2f, 0.8f},
        {0.3f, 0.3f},
        {3.0f, 0.0f},
        {2.5f, 1.5f}
    });
}




// recup d'internet et modifié
static uint32_t read_uint32_be(std::ifstream& file){
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t(bytes[0]) << 24) |
           (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8)  |
           uint32_t(bytes[3]);
}

void get_data_CNN(Tensor& X, Tensor& y, Tensor& x_test, Tensor& y_test,DeviceType device){
    std::string path_images = "./data/mnist/t10k-images-idx3-ubyte";
    std::string path_labels = "./data/mnist/t10k-labels-idx1-ubyte";
    std::ifstream file_images(path_images, std::ios::binary);
    std::ifstream file_labels(path_labels, std::ios::binary);
    uint32_t magic_images = read_uint32_be(file_images);
    uint32_t nb_images = read_uint32_be(file_images);
    uint32_t magic_labels = read_uint32_be(file_labels);
    if(magic_images != 2051){
        Throw_Error("Magic number images invalide.");
        return;
    }
    if(magic_labels != 2049){
        Throw_Error("Magic number labels invalide.");
        return;
    }
    size_t nb_test = 10;
    size_t nb_train = nb_images - nb_test;
    xt::xarray<float> arr_X = xt::zeros<float>(std::vector<size_t>{nb_train, 1, 28, 28});
    xt::xarray<float> arr_y = xt::zeros<float>(std::vector<size_t>{nb_train, 10});
    xt::xarray<float> arr_x_test = xt::zeros<float>(std::vector<size_t>{nb_test, 1, 28, 28});
    xt::xarray<float> arr_y_test = xt::zeros<float>(std::vector<size_t>{nb_test, 10});
    for(size_t i = 0; i < nb_images; i++){
        unsigned char label_char;
        file_labels.read(reinterpret_cast<char*>(&label_char), 1);
        size_t label = static_cast<size_t>(label_char);

        for(size_t r = 0; r < 28; r++){
            for(size_t c = 0; c < 28; c++){
                unsigned char pixel_char;
                file_images.read(reinterpret_cast<char*>(&pixel_char), 1);
                float pixel = float(pixel_char) / 255.0f;

                if(i < nb_test){
                    arr_x_test(i, 0, r, c) = pixel;
                }else{
                    arr_X(i - nb_test, 0, r, c) = pixel;
                }
            }
        }

        if(i < nb_test){
            arr_y_test(i, label) = 1.0f;
        }else{
            arr_y(i - nb_test, label) = 1.0f;
        }
    }

    X = Tensor(device, arr_X);
    y = Tensor(device, arr_y);
    x_test = Tensor(device, arr_x_test);
    y_test = Tensor(device, arr_y_test);
    Print("nb_images au total : ",nb_images);
}



void test_perf_cpu_gpu_simple(size_t N){
    Shape shape({N, N});

    // CPU
    Tensor a_cpu(DeviceType::CPU, shape, true);
    Tensor b_cpu(DeviceType::CPU, shape, true);

    // GPU
    Tensor a_gpu(DeviceType::GPU, shape, true);
    Tensor b_gpu(DeviceType::GPU, shape, true);

    //CPU 
    auto start_cpu = std::chrono::high_resolution_clock::now();
    a_cpu += b_cpu;
    a_cpu *= b_cpu;
    a_cpu -= b_cpu;

    auto end_cpu = std::chrono::high_resolution_clock::now();
    double time_cpu = std::chrono::duration<double>(end_cpu - start_cpu).count();

    //GPU 
    auto start_gpu = std::chrono::high_resolution_clock::now();
    a_gpu += b_gpu;
    a_gpu *= b_gpu;
    a_gpu -= b_gpu;

    auto end_gpu = std::chrono::high_resolution_clock::now();
    double time_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();

    Print("CPU time = ", time_cpu);
    Print("GPU time = ", time_gpu);
}
