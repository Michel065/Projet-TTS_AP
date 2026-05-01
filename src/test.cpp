

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


void print_exemeple_image(Tensor& images,size_t index){
    //on suppose toujours le meme format nbr images, channels , h , w
    size_t C = images.get_shape()[1];
    size_t H = images.get_shape()[2];
    size_t W = images.get_shape()[3];

    for(size_t c = 0; c < C; c++){
        Print("Channel ", c);
        for(size_t h = 0; h < H; h++){
            std::string ligne = "";
            for(size_t w = 0; w < W; w++){
                float val = images.get({index, c, h, w});
                if(val >1){val/=255.0;}

                // on fait une sorte de degradé pour l'instant
                if(val > 0.75f) ligne += "#";
                else if(val > 0.5f) ligne += "O";
                else if(val > 0.25f) ligne += ".";
                else ligne += " ";
            }
            Print(ligne);
        }
        Print("\n");
    }
}

void evaluate_cnn(Model& model, Tensor X, Tensor y){
    Tensor y_pred = model.predict(X);
    const xt::xarray<float> y_pred_data = y_pred.get_data()->to_json();
    const xt::xarray<float> y_data = y.get_data()->to_json();

    size_t batch = y_pred_data.shape()[0];
    size_t nb_classes = y_pred_data.shape()[1];
    size_t correct = 0;

    std::vector<size_t> count_true(nb_classes, 0);
    std::vector<size_t> count_pred(nb_classes, 0);
    std::vector<size_t> count_correct_per_class(nb_classes, 0);
    std::vector<std::vector<size_t>> confusion(nb_classes, std::vector<size_t>(nb_classes, 0));

    for(size_t i = 0; i < batch; i++){
        size_t pred_class = 0;
        size_t true_class = 0;
        float pred_max = y_pred_data(i, 0);
        float true_max = y_data(i, 0);

        for(size_t j = 1; j < nb_classes; j++){
            if(y_pred_data(i, j) > pred_max){
                pred_max = y_pred_data(i, j);
                pred_class = j;
            }
            if(y_data(i, j) > true_max){
                true_max = y_data(i, j);
                true_class = j;
            }
        }

        count_true[true_class]++;
        count_pred[pred_class]++;
        confusion[true_class][pred_class]++;

        if(pred_class == true_class){
            correct++;
            count_correct_per_class[true_class]++;
        }
    }

    float accuracy = static_cast<float>(correct) / static_cast<float>(batch);
    Print("Accuracy : ", accuracy * 100.0f, "%");
    Print("");

    Print("Stats par classe :");
    for(size_t c = 0; c < nb_classes; c++){
        float class_acc = 0.0f;
        if(count_true[c] > 0)
            class_acc = static_cast<float>(count_correct_per_class[c]) / static_cast<float>(count_true[c]);

        Print(
            "Classe ", c,
            " | reel: ", count_true[c],
            " | predit: ", count_pred[c],
            " | correct: ", count_correct_per_class[c],
            " | recall: ", class_acc * 100.0f, "%"
        );
    }

    Print("");
    Print("Matrice de confusion (reel => predit) :");
    for(size_t i = 0; i < nb_classes; i++){
        std::string ligne = "Classe " + std::to_string(i) + " : ";
        for(size_t j = 0; j < nb_classes; j++){
            ligne += std::to_string(confusion[i][j]);
            if(j + 1 < nb_classes)
                ligne += " ";
        }
        Print(ligne);
    }
}
