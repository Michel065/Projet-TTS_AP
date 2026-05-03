#include "model/Auto_Encoder.h"
#include "outil/Print.h"

Auto_Encoder::Auto_Encoder(Model* encoder, Model* decoder){
    _encoder = encoder;
    _decoder = decoder;
    _loss_function = new LossMSE();
}

Auto_Encoder::~Auto_Encoder(){
    delete _encoder;
    delete _decoder;
    delete _loss_function;

    _encoder = nullptr;
    _decoder = nullptr;
    _loss_function = nullptr;
}

void Auto_Encoder::set_loss_function(Loss* loss){
    _loss_function = loss;
}

Tensor Auto_Encoder::encode(Tensor& X){
    return _encoder->forward(X);
}

Tensor Auto_Encoder::decode(Tensor& Z){
    return _decoder->forward(Z);
}

Tensor Auto_Encoder::forward(Tensor& X){
    Tensor Z = encode(X);
    return decode(Z);
}

Tensor Auto_Encoder::predict(Tensor& X){
    return forward(X);
}

float  Auto_Encoder::round_esti(float val, int deci){
	float dix = std::pow(10.0f, deci);
	return std::round(val * dix) / dix;
}


const std::vector<int>  Auto_Encoder::genere_indices_shuffle(int n){
    std::vector<int> indices(n);
    for(int i = 0; i < n; i++)
        indices[i] = i;

    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));
	return indices;
}

float Auto_Encoder::update_time_estimation(float temps_it, int total_it, int actuel_it) {
    avg_time = (avg_time * count + temps_it) / (count + 1);
    count++;
    return avg_time * (total_it - actuel_it);
}

void Auto_Encoder::fit(Tensor X, int epochs, int batch_size, bool shuffle){
    if(_loss_function == nullptr){
        Throw_Error("Fonction loss manquante. Auto_Encoder::fit");
    }
    early_stop = false;
    auto debut_train = std::chrono::high_resolution_clock::now();
    float temps_restant = 0.0f;
    for(int epoch = 0; epoch < epochs; epoch++){
        if(shuffle){
            const std::vector<int> indices = genere_indices_shuffle(X.get_shape()[0]);
            X.shuffle(indices);
        }
        std::vector<Tensor> X_split = X.separation_batch(batch_size);
        int nbr_split = X_split.size();
        float loss_moy = 0.0f;
        for(int id_it = 0; id_it < nbr_split; id_it++){
            auto debut_it = std::chrono::high_resolution_clock::now();
            Tensor input = X_split[id_it];

            Tensor latent = _encoder->forward(input);
            Tensor Y_pred = _decoder->forward(latent);

            float loss = _loss_function->calcul_loss(Y_pred, input);
            loss_moy += loss;

            Tensor grad = _loss_function->calcul_grad(Y_pred, input);

            Tensor grad_latent = _decoder->backward(grad);
            _encoder->backward(grad_latent);

            float temps_it = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - debut_it).count();
            temps_restant = update_time_estimation(temps_it,epochs * nbr_split,epoch * nbr_split + id_it + 1);
            Print_over("Epochs : ", epoch + 1, "/", epochs," iteration : ", id_it + 1, "/", nbr_split," loss train : ", round_esti(loss_moy / (id_it + 1))," temps restant (fin du train): ", std::round(temps_restant), "s");
        }
        _train_loss_history.push_back(loss_moy / nbr_split);
        run_callback();

        if(early_stop){
            Print("\nArret anticipe du training.");
            break;
        }
    }
    early_stop = false;
    auto fin_train = std::chrono::high_resolution_clock::now();
    float temps_total = std::chrono::duration<float>(fin_train - debut_train).count();
    Print("\nTemps total d'entrainement :", temps_total, "s.");
}

std::vector<float>& Auto_Encoder::get_history(){
    return _train_loss_history;
}

void Auto_Encoder::add_callback(Callback* callback){
    callback->set_Model_auto_encoder(this);
    _callbacks.push_back(callback);
}

void Auto_Encoder::run_callback(){
    for(auto callback : _callbacks){
        callback->on_epoch_end();
    }
}

void Auto_Encoder::stop_training(){
    early_stop = true;
}


Auto_Encoder::Auto_Encoder(std::string path){
    _encoder = nullptr;
    _decoder = nullptr;
    _loss_function = new LossMSE();

    load_path(path);
}

Model* Auto_Encoder::get_encoder() const{
    return _encoder;
}

Model* Auto_Encoder::get_decoder() const{
    return _decoder;
}

void Auto_Encoder::save(std::string path,bool aff){
    json j = this;

    std::ofstream file(path);
    if(!file.is_open())
        Throw_Error("Impossible d'ouvrir le fichier en écriture : ", path);

    file << j.dump();
    file.close();
	if(aff)
        Print("Auto_Encoder sauvegarde path:", path);
}

void Auto_Encoder::load_path(std::string path){
    std::ifstream file(path);
    if(!file.is_open())
        Throw_Error("Impossible d'ouvrir le fichier en lecture : ", path);

    json j;
    file >> j;
    file.close();

    from_json(j, this);

    Print("Auto_Encoder charge path:", path);
}

void Auto_Encoder::set_encoder_decoder(Model* encoder, Model* decoder){
    delete _encoder;
    delete _decoder;
    _encoder = encoder;
    _decoder = decoder;
}