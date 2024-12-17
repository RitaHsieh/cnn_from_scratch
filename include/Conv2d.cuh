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

    void setInputProps(int num_dims, int const *dims, int size) override;
    int getOutputNumDims() override { return output_num_dims; };
    int* getOutputDims() override { return output_dims; };
    int getOutputSize() override { return output_size; };
    void setD_in(double* d_ptr) override { d_in = d_ptr; };
    void setD_out(double* d_ptr) override { d_out = d_ptr; };

    Tensor<double> &initOutputTensor() override;
    
    Tensor<double> &forward(Tensor<double> &input) override;

    void forward() override;

    Tensor<double> backprop(Tensor<double> chain_gradient, double learning_rate) override;
    
    double * backprop(double* d_ptr, double learning_rate, bool test) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
};


#endif //NEURAL_NET_IN_CPP_CONV2D_H
