#include "test_data.h"

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
// modif methode, ca marché pas donc recup fichier forat cvv et conv xr array puis conv tensor.
void load_mnist_csv(Tensor& X, Tensor& y, const std::string& path,DeviceType device){
    std::ifstream file(path);
    if(!file.is_open()){
        throw std::runtime_error("Impossible d'ouvrir le fichier : " + path);
    }
    std::string line;
    if(!std::getline(file, line)){
        throw std::runtime_error("Fichier vide : " + path);
    }
    std::vector<float> images_data;
    std::vector<float> labels_data;
    size_t nb_samples = 0;
    while(std::getline(file, line)){
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        if(!std::getline(ss, cell, ',')){
            throw std::runtime_error("Ligne invalide dans : " + path);
        }
        int label = std::stoi(cell);
        if(label < 0 || label > 9){
            throw std::runtime_error("Label invalide dans : " + path);
        }
        for(int k = 0; k < 10; k++){
            labels_data.push_back(k == label ? 1.0f : 0.0f); // conv en onehot
        }
        size_t pixel_count = 0;
        while(std::getline(ss, cell, ',')){
            float v = std::stof(cell);
            images_data.push_back(v);
            pixel_count++;
        }
        if(pixel_count != 28 * 28){
            Throw_Error("Nombre de pixels invalide,",pixel_count);
        }
        nb_samples++;
    }
    X = Tensor(device,xt::adapt(images_data, {nb_samples, size_t(1), size_t(28), size_t(28)}));
    y = Tensor(device,xt::adapt(labels_data, {nb_samples, size_t(10)}));
}

void get_data_CNN(Tensor& X_train,Tensor& y_train,Tensor& X_test,Tensor& y_test,DeviceType device){
    load_mnist_csv(X_train, y_train, "./data/mnist/mnist_train.csv", device);
    load_mnist_csv(X_test, y_test, "./data/mnist/mnist_test.csv", device);
}



// pour le test de l'auto encoder
void load_Cifar_10_csv_X(Tensor& X, const std::string& path, DeviceType device){
    std::ifstream file(path);
    if(!file.is_open()){
        throw std::runtime_error("Impossible d'ouvrir le fichier : " + path);
    }

    std::string line;
    if(!std::getline(file, line)){
        throw std::runtime_error("Fichier vide : " + path);
    }

    std::vector<float> images_data;
    size_t nb_samples = 0;

    while(std::getline(file, line)){
        if(line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;

        while(std::getline(ss, cell, ',')){
            cells.push_back(cell);
        }

        if(cells.size() != 3072 && cells.size() != 3073){
            Throw_Error("Ligne CIFAR invalide, colonnes=", cells.size(), " path=", path);
        }

        for(size_t i = 0; i < 32 * 32 * 3; i++){
            images_data.push_back(std::stof(cells[i]));
        }

        nb_samples++;
    }
    
    Print("Conversion cpu gpu data");
    X = Tensor(device,xt::adapt(images_data, {nb_samples, size_t(3), size_t(32), size_t(32)}));
}

void get_data_Cifar_10(Tensor& X_train,Tensor& X_test,DeviceType device){
    load_Cifar_10_csv_X(X_train, "./data/cifar-10/CIFAR-10-train.csv", device);
    load_Cifar_10_csv_X(X_test, "./data/cifar-10/CIFAR-10-test.csv", device);
}
