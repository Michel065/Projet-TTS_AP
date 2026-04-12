#include "model/Layer_conv/LayerConv2D.h"
#include "model/Model.h"

LayerConv2D::LayerConv2D(size_t nb_filters, size_t kernel) : Layer("Conv2D"){
    _nb_filters = nb_filters;
    _kernel = kernel;
    _device = DeviceType::CPU;
    if( (int)(_kernel / 2) == 0){   
        Throw_Error("Kernel doit etre impaire");
        return;
    } 
}

LayerConv2D::LayerConv2D() : Layer("Conv2D"){
    _device = DeviceType::CPU;
}


void LayerConv2D::build(){
    if( _shape_input.len()!=3){   
        Throw_Error("Dimensions non valides."+_shape_input.print());
        return;
    } 
    _nbr_channel = _shape_input[0];
    set_output_shape(Shape({_nb_filters,_shape_input[1],_shape_input[2]}));

    Shape shape_poid({_nb_filters,_nbr_channel,_kernel,_kernel});
    Shape shape_b({_nb_filters});

    _W = Tensor(_device,shape_poid,true);
    _b = Tensor(_device,shape_b,false);

    //pour le print
    _nb_params = _nb_filters * _nbr_channel * _kernel * _kernel;//les filtres
    _nb_params += _nb_filters;//le bias


    print_couche_msg("Build termine.",Color::GREEN);
}

void LayerConv2D::get_from_model(){
    if(_model != nullptr)
        return;
    _eta = _model->get_eta();
    _device = _model->get_device();
}

Tensor LayerConv2D::forward(Tensor& input){
    _last_input = input;

    size_t batch = input.shape[0];
    size_t Hauteur = input.shape[2];
    size_t Largeur = input.shape[3];

    size_t pad = _kernel / 2;
    Tensor output(input.get_device(),Shape({batch, _nb_filters, Hauteur, Largeur}), false);
    for(size_t b = 0; b < batch; b++){
        //pour chaque batch
        for(size_t f = 0; f < _nb_filters; f++){
            //pour chaque filtre
            for(size_t y = 0; y < Hauteur; y++){
                //pour chaque pixel hauteur
                for(size_t x = 0; x < Largeur; x++){
                    //pour chaque pixel largeur

                    float sum = 0.0f;
                    for(size_t c = 0; c < _nbr_channel; c++){
                        //pour chaque channel donc probablement 1 mais on sait jamais pour les tests
                        for(size_t ky = 0; ky < _kernel; ky++){
                            //ensuite on parcours la taille du kernel
                            for(size_t kx = 0; kx < _kernel; kx++){
                                //on centre avec pad
                                int in_y = int(y) + int(ky) - int(pad);
                                int in_x = int(x) + int(kx) - int(pad);

                                //ca remplace la gestion du padding sur les bordures
                                if(in_y >= 0 && in_y < (int)Hauteur && in_x >= 0 && in_x < (int)Largeur){ // si on depasse pour eviter d'avoir a faire le padding juste on idgore l'operation
                                    sum += input.get({b, c, (size_t)in_y, (size_t)in_x}) * _W.get({f, c, ky, kx});
                                }
                            }
                        }
                    }

                    sum += _b.get({f});
                    output.set({b, f, y, x},sum);
                }
            }
        }
    }

    return output;
}


Tensor LayerConv2D::backward(const Tensor& grad){
    size_t batch = _last_input.shape[0];
    size_t Hauteur = _last_input.shape[2];
    size_t Largeur = _last_input.shape[3];

    size_t pad = _kernel / 2;

    Tensor grad_W(grad.get_device(), _W.shape, false);
    Tensor grad_b(grad.get_device(),_b.shape, false);
    Tensor grad_prec(grad.get_device(),_last_input.shape, false);

    //grad_b
    for(size_t b = 0; b < batch; b++){
        for(size_t f = 0; f < _nb_filters; f++){
            for(size_t y = 0; y < Hauteur; y++){
                for(size_t x = 0; x < Largeur; x++){
                    grad_b.set({f}, grad_b.get({f})+grad.get({b,f,y,x}));
                }
            }
        }
    }

    //grad_W
    for(size_t f = 0; f < _nb_filters; f++){
        for(size_t c = 0; c < _nbr_channel; c++){
            for(size_t ky = 0; ky < _kernel; ky++){
                for(size_t kx = 0; kx < _kernel; kx++){

                    float sum = 0.0f;
                    for(size_t b = 0; b < batch; b++){
                        for(size_t y = 0; y < Hauteur; y++){
                            for(size_t x = 0; x < Largeur; x++){

                                int in_y = int(y) + int(ky) - int(pad);
                                int in_x = int(x) + int(kx) - int(pad);

                                if(in_y >= 0 && in_y < (int)Hauteur && in_x >= 0 && in_x < (int)Largeur){
                                    sum += _last_input.get({b,c,(size_t)in_y,(size_t)in_x}) * grad.get({b,f,y,x});
                                }
                            }
                        }
                    }

                    grad_W.set({f,c,ky,kx},sum / batch);
                }
            }
        }
    }

    //grad_prec
    for(size_t b = 0; b < batch; b++){
        for(size_t c = 0; c < _nbr_channel; c++){
            for(size_t y = 0; y < Hauteur; y++){
                for(size_t x = 0; x < Largeur; x++){

                    float sum = 0.0f;

                    for(size_t f = 0; f < _nb_filters; f++){
                        for(size_t ky = 0; ky < _kernel; ky++){
                            for(size_t kx = 0; kx < _kernel; kx++){

                                int out_y = int(y) - int(ky) + int(pad);
                                int out_x = int(x) - int(kx) + int(pad);

                                if(out_y >= 0 && out_y < (int)Hauteur && out_x >= 0 && out_x < (int)Largeur){
                                    sum += _W.get({f,c,ky,kx}) * grad.get({b,f,(size_t)out_y,(size_t)out_x});
                                }
                            }
                        }
                    }

                    grad_prec.set({b,c,y,x},sum);
                }
            }
        }
    }
 
    _W -= grad_W * _eta;
    _b -= grad_b * _eta;

    return grad_prec;
}

void LayerConv2D::to_json(json& j) const{
    j["_W"]=_W; 
    j["_b"]=_b;
}

void LayerConv2D::load_json(const json& j) {
    j.at("_W").get_to(_W);
    j.at("_b").get_to(_b);
}