//
// Created by tabelini on 18/04/19.
//

#ifndef NEURAL_NET_IN_CPP_MAXPOOL_H
#define NEURAL_NET_IN_CPP_MAXPOOL_H


#include "Module.h"

class MaxPool : public Module {
private:
    int input_dims[4]{};
    int input_size;
    int output_dims[4]{};
    int output_num_dims = 4;
    int output_size;

    Tensor<double> output_;
    Tensor<double> input_;
    Tensor<int> indexes;
    int stride_, size_;

    double *d_in;
    double *d_out;
    int *d_indexes;

public:

    void setInputProps(int num_dims, int const *dims, int size)override;
    int getOutputNumDims() { return output_num_dims; }override;
    int* getOutputDims() { return output_num_dims; }override;
    int getOutputSize() { return output_dims; }override;
    void setD_in(double* d_ptr) { d_in = d_ptr; }override;
    void setD_out(double* d_ptr) { d_out = d_ptr; }override;

    explicit MaxPool(int size, int stride);

    Tensor<double> &forward(Tensor<double> &input) override;

    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
};


#endif //NEURAL_NET_IN_CPP_MAXPOOL_H
