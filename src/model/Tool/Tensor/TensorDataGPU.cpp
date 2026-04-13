#include "model/Tool/Tensor/TensorDataGPU.h"
#include "model/Tool/Tensor/Tensor.h"

const TensorDataGPU* TensorDataGPU::check_gpu(const Tensor& a) const{
    if(!a.is_gpu())
        Throw_Error("TensorDataGPU: Tensor non cpu");
    auto da = dynamic_cast<TensorDataGPU*>(a.get_data());
    if(!da)
        Throw_Error("TensorDataGPU: cast impossible");
    return da;
}