#include "model/Layer_conv/LayerConv2D/LayerConv2D.h"
#include "model/Model.h"

//https://eunomia.dev/others/cuda-tutorial/06-cnn-convolution/

LayerConv2D::LayerConv2D(size_t nb_filters, size_t kernel) : Layer("Conv2D"){
    if( (int)(kernel / 2) == 0){   
        Throw_Error("Kernel doit etre impaire");
        return;
    }
    _nb_filters = nb_filters;
    _kernel = kernel;
    _padding = kernel / 2;
    _device = DeviceType::CPU;
}

LayerConv2D::LayerConv2D() : Layer("Conv2D"){
    _device = DeviceType::CPU;
}


void LayerConv2D::build(){
    if( _shape_input.len()!=3){   
        Throw_Error("Dimensions non valides."+_shape_input.print());
        return;
    } 
    set_output_shape(Shape({_nb_filters,_shape_input[1],_shape_input[2]}));

    Shape shape_poid({_nb_filters,_shape_input[0],_kernel, _kernel});
    Shape shape_b({_nb_filters});

    _W = Tensor(_device,shape_poid,true);
    _b = Tensor(_device,shape_b,false);

    //pour le print
    _nb_params = shape_poid.size();//les filtres
    _nb_params += shape_b.size();//le bias


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
    
    Tensor output(input.get_device(),Shape({input.get_shape()[0], _nb_filters, _shape_input[1], _shape_input[2]}), false);
    if(input.is_cpu()){
        LayerConv2DCPU::forward(output,input,_W,_b,_nb_filters,_kernel,_padding,_shape_input);
    }else if(input.is_gpu()){

    }

    return output;
}


Tensor LayerConv2D::backward(const Tensor& grad){
    Tensor grad_W(grad.get_device(), _W.get_shape(), false);
    Tensor grad_b(grad.get_device(),_b.get_shape(), false);
    Tensor grad_prec(grad.get_device(),_last_input.get_shape(), false);

    if(grad.is_cpu()){
        LayerConv2DCPU::backward(grad_W,grad_b,grad_prec,grad,_last_input,_W,_nb_filters,_kernel,_padding,_shape_input);
    }else if(grad.is_gpu()){
        
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