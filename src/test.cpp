#include "test.h"

void get_data_lineaire(Tensor& X, Tensor& y, Tensor& x_test){
    X = Tensor(std::vector<std::vector<float>>{
        {-1.0f, -1.0f},
        {-1.2f, -0.8f},
        {-0.8f, -1.1f},
        {-1.1f, -1.3f},

        { 1.0f,  1.0f},
        { 1.2f,  0.9f},
        { 0.8f,  1.1f},
        { 1.1f,  1.3f}
    });

    y = Tensor(std::vector<std::vector<float>>{
        {1.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 0.0f},

        {0.0f, 1.0f},
        {0.0f, 1.0f},
        {0.0f, 1.0f},
        {0.0f, 1.0f}
    });

    x_test = Tensor(std::vector<std::vector<float>>{
        { 0.9f,  1.0f},
        {-0.9f, -1.0f}
    });
}

void get_data_non_lineaire(Tensor& X, Tensor& y, Tensor& x_test,size_t n){
    X = Tensor(Shape({n, 2}));
    y = Tensor(Shape({n, 1}));

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
            X.set({(int)idx, 0}, x);
            X.set({(int)idx, 1}, yy);
            y.set({(int)idx, 0}, 1.0f);
            idx++;
        }
    }

    while(idx < n){
        float x = rand_float(-3.0f, 3.0f);
        float yy = rand_float(-3.0f, 3.0f);
        float dist = std::sqrt(x * x + yy * yy);

        if(dist < r1 || dist > r2){
            X.set({(int)idx, 0}, x);
            X.set({(int)idx, 1}, yy);
            y.set({(int)idx, 0}, 0.0f);
            idx++;
        }
    }

    x_test = Tensor(std::vector<std::vector<float>>{
        {1.5f, 0.0f},
        {1.2f, 0.8f},
        {0.3f, 0.3f},
        {3.0f, 0.0f},
        {2.5f, 1.5f}
    });
}
