//
// Created by lucas on 11/04/19.
//

#ifndef NEURAL_NET_IN_CPP_RELU_H
#define NEURAL_NET_IN_CPP_RELU_H


#include "Tensor.h"
#include "Module.h"

class ReLU : public Module{
private:
    int input_dims[2]{};
    // int input_num_dims;
    int input_size;
    int output_dims[2]{};
    int output_num_dims = 2;
    int output_size;

    // double* d_weight;
    double* d_in;
    double* d_out;

    Tensor<double> input_;
    Tensor<double> product_;
public:
    ReLU();

    void setInputProps(int num_dims, int const *dims, int size)override;
    int getOutputNumDims() { return output_num_dims; }override;
    int* getOutputDims() { return output_dims; }override;
    int getOutputSize() { return output_size; }override;
    void setD_in(double* d_ptr) { d_in = d_ptr; }override;
    void setD_out(double* d_ptr) { d_out = d_ptr; }override;
    
    void forward();
    void backprop();

    Tensor<double> &forward(Tensor<double> &input) override;

    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
};


#endif //NEURAL_NET_IN_CPP_RELU_H
