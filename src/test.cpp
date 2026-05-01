#include "test.h"

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

void test_non_lineaire(DeviceType device){
    Tensor X, y, x_test;
    get_data_non_lineaire(X, y, x_test,2000,device);
    size_t nbr_neur_in = (X.get_shape()[1]);
    int nbr_neur_out = y.get_shape()[1];

    Print("construction model.");
    Model model({.input_shape = Shape({nbr_neur_in}), .eta = 2 ,.device=device});
    model.add(new LayerNormalisation({-3,-3},{3,3}));// c le min et le max a la main
    model.add(new LayerDense(20));
    model.add(new LayerRelu());
    model.add(new LayerDense(20));
    model.add(new LayerRelu());
    model.add(new LayerDense(5));
    model.add(new LayerRelu());
    model.add(new LayerDense(nbr_neur_out));
    model.add(new LayerSigmoid());
    model.set_loss_function(new LossBinaryCrossEntropy());

    model.add_callback(new CallbackEarlyStopLoss({.patience = 15}));

    //model.set_affichge_level(2);

    Print("entrainement.");
    model.fit(X,y,1500,256);
    
    Print("Test:");
    Tensor y_test = model.predict(x_test).round(2)*100;
    Print("Prediction :",y_test);

    //model.print();
    model.create_graph_loss_entrainement();
    //model.save("./models/model.json");
    
}

void test_load(){
    Tensor X, y, x_test;
    get_data_non_lineaire(X, y, x_test,1000);

    Model model("./models/model.json");
    model.set_loss_function(new LossBinaryCrossEntropy());
    

    Print("Test:");
    Tensor y_test = model.predict(x_test).round(2)*100;
    
    Print("Prediction :",y_test);
}

void test_CNN(DeviceType device){
    Tensor X, y, x_test, y_test;
    Print("Chargement des datas:"); // je met un print car pas otpi tres long.
    get_data_CNN(X, y, x_test,y_test,device);
    Print("Chargement des datas Fini. X(",X.get_shape()[0],") y(",x_test.get_shape()[0],")");

    Print("construction model.");
    Model model({.input_shape = Shape({1,28,28}), .eta = 1, .device=device});
    model.add(new LayerNormalisationImage());

    model.add(new LayerConv2D(2,3));
    model.add(new LayerRelu());
    model.add(new LayerMaxPool2D());
 
    model.add(new LayerConv2D(16,3));
    model.add(new LayerRelu());
    model.add(new LayerMaxPool2D());

    model.add(new LayerFlatten());

    model.add(new LayerDense(10));
    model.add(new LayerRelu());
    model.add(new LayerDense(10));
    model.add(new LayerSoftMax());
    model.set_loss_function(new LossCrossEntropy());
    model.add_callback(new CallbackEarlyStopLoss({.patience = 5}));

    model.set_affichge_level(1);

    Print("entrainement.");
    model.fit(X,y,75,128);

    Print("Test:");
    evaluate_cnn(model,x_test,y_test);
    model.create_graph_loss_entrainement();
    //model.save("./models/model_cnn.json",false);
}

void test_CNN_load(DeviceType device){
    Tensor X, y, x_test, y_test;
    Print("Chargement des datas:");
    get_data_CNN(X, y, x_test,y_test,device);
    Print("Chargement des datas Fini. X(",X.get_shape()[0],") y(",x_test.get_shape()[0],")");
    
    int nbr_image_train=200;
    x_test=x_test.extraction_section_axe_0(0,nbr_image_train);
    y_test=y_test.extraction_section_axe_0(0,nbr_image_train);
    Print("reduce X(",X.get_shape()[0],") y(",x_test.get_shape()[0],")");
    
    Print("construction model.");
    Model model("./models/model_cnn.json");
    // ajout du loss
    model.set_loss_function(new LossCrossEntropy());
    
    Print("Test:");
    evaluate_cnn(model,x_test,y_test);
}

void test_UpSampling(DeviceType device){
    Tensor X, y, x_test, y_test;
    Print("Chargement des datas:");
    get_data_CNN(X, y, x_test,y_test,device);
    Print("Chargement des datas Fini. X(",X.get_shape()[0],") y(",x_test.get_shape()[0],")");

    Print("construction model.");
    Model model({.input_shape = Shape({1,28,28}), .eta = 1, .device=device});
    model.add(new LayerNormalisationImage());

    model.add(new LayerConv2D(2,3));
    model.add(new LayerRelu());
    model.add(new LayerMaxPool2D());
 
    model.add(new LayerConv2D(16,3));
    model.add(new LayerRelu());
    model.add(new LayerMaxPool2D());

    model.add(new LayerFlatten());

    model.add(new LayerDense(10));
    model.add(new LayerRelu());
    model.add(new LayerDense(10));
    model.add(new LayerSoftMax());
    model.set_loss_function(new LossCrossEntropy());
    model.add_callback(new CallbackEarlyStopLoss({.patience = 5}));

    model.set_affichge_level(1);

    Print("entrainement.");
    model.fit(X,y,75,128);

    Print("Test:");
    evaluate_cnn(model,x_test,y_test);
    model.create_graph_loss_entrainement();
}