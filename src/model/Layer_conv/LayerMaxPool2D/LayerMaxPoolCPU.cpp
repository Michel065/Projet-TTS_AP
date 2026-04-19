#include "model/Layer_conv/LayerMaxPool2D/LayerMaxPoolCPU.h"

void LayerMaxPoolCPU::forward(Tensor& output, Tensor& _mask, Tensor& input, size_t _taille_batch, Shape shape_output){
    size_t channels = shape_output[0];
    size_t out_H = shape_output[1];
    size_t out_W = shape_output[2];

    for(size_t b = 0; b < _taille_batch; b++){
        for(size_t c = 0; c < channels; c++){
            for(size_t y = 0; y < out_H; y++){
                for(size_t x = 0; x < out_W; x++){
                    size_t in_y = y * 2;
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
}


void LayerMaxPoolCPU::backward(Tensor& grad_out, Tensor& _mask, Tensor& grad_in, size_t _taille_batch, Shape shape_output){
    size_t channels = shape_output[0];
    size_t out_H = shape_output[1];
    size_t out_W = shape_output[2];
    for(size_t b = 0; b < _taille_batch; b++){
        for(size_t c = 0; c < channels; c++){
            for(size_t y = 0; y < out_H; y++){
                for(size_t x = 0; x < out_W; x++){
                    size_t in_y = y * 2;
                    size_t in_x = x * 2;
                    for(size_t dec_y = 0; dec_y < 2; dec_y++){
                        for(size_t dec_x = 0; dec_x < 2; dec_x++){
                            size_t yy = in_y + dec_y;
                            size_t xx = in_x + dec_x;
                            if(_mask.get({b,c,yy,xx}) == 1.0f){
                                grad_out.set({b,c,yy,xx},grad_in.get({b,c,y,x}));
                            }
                        }
                    }
                }
            }
        }
    }
}
