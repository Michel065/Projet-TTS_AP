#include "model/Layer_conv/LayerMaxPool2D.h"
#include "model/Model.h"

LayerMaxPool2D::LayerMaxPool2D() : Layer("MaxPool2D") {
}

void LayerMaxPool2D::build(){
    if(_shape_input.len() != 3){
        Throw_Error("Dimensions non valides."+_shape_input.print());
        return;
    }

    _channels = _shape_input[0];

    set_output_shape(Shape({_channels, _shape_input[1]/2, _shape_input[2]/2}));

    print_couche_msg("Build termine.", Color::GREEN);
}

Tensor LayerMaxPool2D::forward(Tensor& input){
    Shape shape_i= input.get_shape();

    _last_batch = shape_i[0];
    size_t out_H = _shape_output[1];
    size_t out_W = _shape_output[2];

    Tensor output(input.get_device(),Shape({_last_batch, _channels, out_H, out_W}), false);
    _mask = Tensor(input.get_device(),shape_i, false);// pour replacer la val lors du back

    for(size_t b = 0; b < _last_batch; b++){
        for(size_t c = 0; c < _channels; c++){
            for(size_t y = 0; y < out_H; y++){
                for(size_t x = 0; x < out_W; x++){

                    size_t in_y = y * 2; // c'est fixe pas modifiable (!= torch)
                    size_t in_x = x * 2;
                    
                    // init
                    float max_val = input.get({b, c, in_y, in_x});
                    size_t m_y = in_y, m_x = in_x;
                    for(size_t dec_y = 0; dec_y < 2; dec_y++){
                        for(size_t dec_x = 0; dec_x < 2; dec_x++){
                            float val = input.get({b, c, in_y + dec_y, in_x + dec_x});
                            if(val > max_val){
                                max_val = val;
                                m_y = in_y + dec_y;
                                m_x = in_x + dec_x;
                            }
                        }
                    }
                    output.set({b,c,y,x},max_val);
                    _mask.set({b,c,m_y,m_x},1.0f);
                }
            }
        }
    }
    return output;
}

Tensor LayerMaxPool2D::backward(const Tensor& grad){
    size_t out_H = _shape_output[1];
    size_t out_W = _shape_output[2];
    Tensor grad_out(grad.get_device(),Shape({_last_batch,_channels,_shape_input[1],_shape_input[2]}), false);

    for(size_t b = 0; b < _last_batch; b++){
        for(size_t c = 0; c < _channels; c++){
            for(size_t y = 0; y < out_H; y++){
                for(size_t x = 0; x < out_W; x++){
                    size_t in_y = y * 2;
                    size_t in_x = x * 2;
                    for(size_t dec_y = 0; dec_y < 2; dec_y++){
                        for(size_t dec_x = 0; dec_x < 2; dec_x++){
                            size_t yy = in_y + dec_y;
                            size_t xx = in_x + dec_x;
                            if(_mask.get({b,c,yy,xx}) == 1.0f){
                                grad_out.set({b,c,yy,xx},grad.get({b,c,y,x}));
                            }
                        }
                    }
                }
            }
        }
    }
    return grad_out;
}

void LayerMaxPool2D::to_json(json& j) const{}

void LayerMaxPool2D::load_json(const json& j){}