//
// Created by lucas on 14/04/19.
//

#ifndef NEURAL_NET_IN_CPP_CONV2D_CUDA_H
#define NEURAL_NET_IN_CPP_CONV2D_CUDA_H

#include "Module.h"

class Conv2d : public Module {
private:
    int input_dims[4]{};
    int input_size;
    int output_dims[4]{};
    int output_num_dims = 4;
    int output_size;

    Tensor<double> input_;
    Tensor<double> product_;
    int stride, padding;
    double *d_in;
    double *d_out;
    double *d_kernel;
    double *d_bias;
public:
    Tensor<double> kernels;
    Tensor<double> bias;

    Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, int seed = 0);

    void setInputProps(int num_dims, int const *dims, int size)override;
    int getOutputNumDims() { return output_num_dims; }override;
    int* getOutputDims() { return output_num_dims; }override;
    int getOutputSize() { return output_dims; }override;
    void setD_in(double* d_ptr) { d_in = d_ptr; }override;
    void setD_out(double* d_ptr) { d_out = d_ptr; }override;

    Tensor<double> &initOutputTensor() override;
    
    double * forward(Tensor<double> &input, double *d_in) override;

    Tensor<double> backprop(Tensor<double> chain_gradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
};


#endif //NEURAL_NET_IN_CPP_CONV2D_H
