#include "model/Layer_conv/LayerConv2D/LayerConv2DCPU.h"


void LayerConv2DCPU::forward(Tensor& output, Tensor& input, const Tensor& _W, const Tensor& _b,size_t nb_filters, size_t kernel, size_t pad, Shape shape_input){
    size_t batch = input.get_shape()[0];
    for(size_t b = 0; b < batch; b++){
        //pour chaque batch
        for(size_t f = 0; f < nb_filters; f++){
            //pour chaque filtre
            for(size_t y = 0; y < shape_input[1]; y++){
                //pour chaque pixel shape_input[1]
                for(size_t x = 0; x < shape_input[2]; x++){
                    //pour chaque pixel shape_input[2]
                    float sum = 0.0f;
                    for(size_t c = 0; c < shape_input[0]; c++){
                        //pour chaque channel donc probablement 1 mais on sait jamais pour les tests
                        for(size_t ky = 0; ky < kernel; ky++){
                            //ensuite on parcours la taille du kernel
                            for(size_t kx = 0; kx < kernel; kx++){
                                //on centre avec pad
                                int in_y = int(y) + int(ky) - int(pad);
                                int in_x = int(x) + int(kx) - int(pad);

                                //ca remplace la gestion du padding sur les bordures
                                if(in_y >= 0 && in_y < (int)shape_input[1] && in_x >= 0 && in_x < (int)shape_input[2]){ // si on depasse pour eviter d'avoir a faire le padding juste on igore l'operation
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
}


void LayerConv2DCPU::backward(Tensor& grad_W, Tensor& grad_b,Tensor& grad_output,const Tensor& grad_input, Tensor& last_input, Tensor& _W, size_t nb_filters, size_t kernel, size_t pad,Shape shape_input){
    size_t batch = last_input.get_shape()[0];

    //grad_b
    for(size_t b = 0; b < batch; b++){
        for(size_t f = 0; f < nb_filters; f++){
            for(size_t y = 0; y < shape_input[1]; y++){
                for(size_t x = 0; x < shape_input[2]; x++){
                    grad_b.set({f}, grad_b.get({f})+grad_input.get({b,f,y,x}));
                }
            }
        }
    }

    //grad_W
    for(size_t f = 0; f < nb_filters; f++){
        for(size_t c = 0; c < shape_input[0]; c++){
            for(size_t ky = 0; ky < kernel; ky++){
                for(size_t kx = 0; kx < kernel; kx++){

                    float sum = 0.0f;
                    for(size_t b = 0; b < batch; b++){
                        for(size_t y = 0; y < shape_input[1]; y++){
                            for(size_t x = 0; x < shape_input[2]; x++){

                                int in_y = int(y) + int(ky) - int(pad);
                                int in_x = int(x) + int(kx) - int(pad);

                                if(in_y >= 0 && in_y < (int)shape_input[1] && in_x >= 0 && in_x < (int)shape_input[2]){
                                    sum += last_input.get({b,c,(size_t)in_y,(size_t)in_x}) * grad_input.get({b,f,y,x});
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
        for(size_t c = 0; c < shape_input[0]; c++){
            for(size_t y = 0; y < shape_input[1]; y++){
                for(size_t x = 0; x < shape_input[2]; x++){

                    float sum = 0.0f;

                    for(size_t f = 0; f < nb_filters; f++){
                        for(size_t ky = 0; ky < kernel; ky++){
                            for(size_t kx = 0; kx < kernel; kx++){

                                int out_y = int(y) - int(ky) + int(pad);
                                int out_x = int(x) - int(kx) + int(pad);

                                if(out_y >= 0 && out_y < (int)shape_input[1] && out_x >= 0 && out_x < (int)shape_input[2]){
                                    sum += _W.get({f,c,ky,kx}) * grad_input.get({b,f,(size_t)out_y,(size_t)out_x});
                                }
                            }
                        }
                    }

                    grad_output.set({b,c,y,x},sum);
                }
            }
        }
    }
}
