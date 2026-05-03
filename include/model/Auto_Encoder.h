#pragma once

#include "model/Model.h"
#include "model/Loss/Loss.h"
#include "model/Loss/LossMSE.h"

class Auto_Encoder {
private:
    Model* _encoder;
    Model* _decoder;
    Loss* _loss_function;

    std::vector<float> _train_loss_history;
    std::vector<Callback*> _callbacks;
    
    float avg_time = 0;
    int count = 0;
    float update_time_estimation(float temps_it, int total_it, int current_it);

public:
    Auto_Encoder(Model* encoder, Model* decoder);
    
    Auto_Encoder(const Auto_Encoder&) = delete;
    Auto_Encoder& operator=(const Auto_Encoder&) = delete;
    ~Auto_Encoder();
    
    Tensor encode(Tensor& X);
    Tensor decode(Tensor& Z);
    Tensor forward(Tensor& X);
    Tensor predict(Tensor& X);

    void set_loss_function(Loss* loss);
    void fit(Tensor X, int epochs, int batch_size, bool shuffle = true);

    std::vector<float>& get_history();
    float round_esti(float val, int deci = 5);
    const std::vector<int>  genere_indices_shuffle(int n);

    void add_callback(Callback* callback);
    void run_callback();
    void stop_training();
	bool early_stop=false;



    Auto_Encoder(std::string path);

    void save(std::string path,bool aff = true);
    void load_path(std::string path);
    void set_encoder_decoder(Model* encoder, Model* decoder);
    
    Model* get_encoder() const;
    Model* get_decoder() const;
};

inline void to_json(json& j, const Auto_Encoder* ae){
    j = {
        {"encoder", ae->get_encoder()},
        {"decoder", ae->get_decoder()}
    };
}

inline void from_json(const json& j, Auto_Encoder* ae){
    Model* encoder = new Model();
    Model* decoder = new Model();

    from_json(j.at("encoder"), encoder);
    from_json(j.at("decoder"), decoder);

    ae->set_encoder_decoder(encoder, decoder);
}