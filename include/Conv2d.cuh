//
// Created by lucas on 14/04/19.
//

#ifndef NEURAL_NET_IN_CPP_CONV2D_CUDA_H
#define NEURAL_NET_IN_CPP_CONV2D_CUDA_H

#include "Module.h"

class Conv2d : public Module {
private:
    Tensor<double> input_;
    Tensor<double> product_;
    int stride, padding;
    double *d_in;
    double *d_out;
public:
    Tensor<double> kernels;
    Tensor<double> bias;

    Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, int seed = 0);

    int getOutputSize();

    // void setInputPointer(double* d_in) override;

    Tensor<double> &initOutputTensor();
    
    double * forward(Tensor<double> &input, double *d_in) override;

    Tensor<double> backprop(Tensor<double> chain_gradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;

    // __global__ void Conv2d_gpu();
};


#endif //NEURAL_NET_IN_CPP_CONV2D_H
