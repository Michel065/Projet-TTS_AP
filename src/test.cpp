#include "test.h"

void test_RNN(){
    size_t n = 5000;

    Tensor X(Shape({n, 2}));
    Tensor y(Shape({n, 1}));

    float r1 = 1.0f;
    float r2 = 2.0f;

    size_t n_in = n / 2;
    size_t idx = 0;

    auto rand_float = [](float a, float b) {
        return a + ((float)rand() / (float)RAND_MAX) * (b - a);
    };

    while(idx < n_in){
        float x = rand_float(-3.0f, 3.0f);
        float yy = rand_float(-3.0f, 3.0f);

        float dist = std::sqrt(x*x + yy*yy);

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

        float dist = std::sqrt(x*x + yy*yy);

        if(dist < r1 || dist > r2){
            X.set({(int)idx, 0}, x);
            X.set({(int)idx, 1}, yy);
            y.set({(int)idx, 0}, 0.0f);
            idx++;
        }
    }

    Tensor x_test = xt::xarray<float>{
        {1.5f, 0.0f},   // dedans
        {1.2f, 0.8f},   // dedans
        {0.3f, 0.3f},   // trop proche
        {3.0f, 0.0f},   // trop loin
        {2.5f, 1.5f}    // trop loin
    };

    Print("construction model.");
    Model model("ring_band", Shape({2}), 0.01);

    model.add(new LayerDense(16));
    model.add(new LayerRelu());
    model.add(new LayerDense(12));
    model.add(new LayerRelu());
    model.add(new LayerDense(8));
    model.add(new LayerRelu());
    model.add(new LayerDense(1));
    model.add(new LayerSigmoid());

    Print("entrainement.");
    model.fit(X, y, 1000);

    Print("");
    Print("Test:");
    Tensor pred = model.forward(x_test).round(3);
    Print("Prediction :", pred);
}
