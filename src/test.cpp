#include "test.h"

void print_exemeple_image(Tensor& images,size_t index,bool norm){
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
                if(norm && val > 1){val/=255.0;}

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


void print_exemple_images_bi(Tensor& images1, Tensor& images2, size_t index, bool norm){
    size_t C = images1.get_shape()[1];
    size_t H = images1.get_shape()[2];
    size_t W = images1.get_shape()[3];

    for(size_t c = 0; c < C; c++){
        Print("Channel ", c);
        for(size_t h = 0; h < H; h++){
            std::string ligne1 = "";
            std::string ligne2 = "";
            for(size_t w = 0; w < W; w++){
                float val1 = images1.get({index, c, h, w});
                if(norm && val1 > 1) val1 /= 255.0f;

                if(val1 > 0.75f) ligne1 += "#";
                else if(val1 > 0.5f) ligne1 += "O";
                else if(val1 > 0.25f) ligne1 += ".";
                else ligne1 += " ";

                float val2 = images2.get({index, c, h, w});
                if(norm && val2 > 1) val2 /= 255.0f;

                if(val2 > 0.75f) ligne2 += "#";
                else if(val2 > 0.5f) ligne2 += "O";
                else if(val2 > 0.25f) ligne2 += ".";
                else ligne2 += " ";
            }
            Print(ligne1 + "     |     " + ligne2);
        }
        Print("\n");
    }
}

void test_AutoEncoder_Cifar(DeviceType device){
    Tensor X, x_test;

    Print("Chargement des datas:");
    get_data_Cifar_10(X, x_test, device);

    X /= 255.0f;

    int nbr_val = 1;
    x_test = x_test.extraction_section_axe_0(0, nbr_val);
    x_test /= 255.0f;

    Print("Chargement des datas Fini. X(", X.get_shape()[0], ") test(", x_test.get_shape()[0], ")");

    Print("construction encoder.");
    Model* encoder = new Model({.input_shape = Shape({3,32,32}), .eta = 1, .device=device});

    encoder->add(new LayerConv2D(32,3));
    encoder->add(new LayerRelu());
    encoder->add(new LayerMaxPool2D());

    encoder->add(new LayerConv2D(16,3));
    encoder->add(new LayerRelu());
    encoder->add(new LayerMaxPool2D());

    encoder->add(new LayerFlatten());

    encoder->add(new LayerDense(256));
    encoder->add(new LayerRelu());

    Print("construction decoder.");
    Model* decoder = new Model({.input_shape = Shape({256}), .eta = 0.05, .device=device});

    decoder->add(new LayerDense(16 * 8 * 8));
    decoder->add(new LayerRelu());

    decoder->add(new LayerUnflatten(Shape({16,8,8})));

    decoder->add(new LayerUpSampling2D());
    decoder->add(new LayerConv2D(16,3));
    decoder->add(new LayerRelu());

    decoder->add(new LayerUpSampling2D());
    decoder->add(new LayerConv2D(32,3));
    decoder->add(new LayerRelu());

    decoder->add(new LayerConv2D(3,3));
    decoder->add(new LayerSigmoid());

    encoder->print();
    decoder->print();

    Auto_Encoder ae(encoder, decoder);
    ae.set_loss_function(new LossMSE());

    Print("entrainement.");
    ae.fit(X, 1, 128);

    Tensor Y_test_pred = ae.predict(x_test);
    debug_check_tensor_non_vide(Y_test_pred);
    for(size_t i = 0; i < (size_t)nbr_val; i++){
        debug_check_tensor_non_vide_batch(Y_test_pred, i, "Y_pred");
        debug_check_tensor_non_vide_batch(x_test, i, "Y_reel");
        print_exemple_images_bi(x_test, Y_test_pred, i, false);
    }
}


void test_AutoEncoder_Cifar_v2(DeviceType device){
    Tensor X, x_test;
    Print("Chargement des datas:");
    get_data_Cifar_10(X, x_test, device);
    int nbr_val = 1;
    x_test = x_test.extraction_section_axe_0(0, nbr_val);
    Print("Chargement des datas Fini. X(", X.get_shape()[0], ") test(", x_test.get_shape()[0], ")");

    float eta = 0.001;
    Print("construction encoder.");
    Model* encoder = new Model({.model_name = "Encoder", .input_shape = Shape({3,32,32}), .eta = eta, .device=device});
    encoder->add(new LayerConv2D(16,3));
    encoder->add(new LayerRelu());
    encoder->add(new LayerMaxPool2D());

    encoder->add(new LayerConv2D(16,3));
    encoder->add(new LayerRelu());
    encoder->add(new LayerMaxPool2D());

    encoder->add(new LayerFlatten());
    encoder->add(new LayerIdentity()); // latent = 1024

    Print("construction decoder.");
    Model* decoder = new Model({.model_name = "Decoder", .input_shape = Shape({1024}), .eta = eta, .device=device}); //16*8*8 = 2024

    decoder->add(new LayerUnflatten(Shape({16,8,8})));

    decoder->add(new LayerUpSampling2D());
    decoder->add(new LayerConv2D(16,3));
    decoder->add(new LayerRelu());

    decoder->add(new LayerUpSampling2D());
    decoder->add(new LayerConv2D(16,3));
    decoder->add(new LayerRelu());

    decoder->add(new LayerConv2D(3,3));
    decoder->add(new LayerSigmoid());


    // petit print
    encoder->print();
    decoder->print();

    Auto_Encoder ae(encoder, decoder);
    ae.set_loss_function(new LossMSE());
    ae.add_callback(new CallbackEarlyStopLoss({.patience = 2}));

    Print("entrainement.");
    ae.fit(X, 5, 128);

    Tensor Y_test_pred = ae.predict(x_test);
    for(size_t i = 0; i < (size_t)nbr_val; i++){
        print_exemple_images_bi(x_test, Y_test_pred, i, false);
    }
}