//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_FULLYCONNECTED_H
#define NEURAL_NET_IN_CPP_FULLYCONNECTED_H

#include "Module.h"
#include "Tensor.h"

/*
 * Fully Connected layer
 * Output: Mx + b
 */
class FullyConnected : public Module {
private:
    Tensor<double> weights;
    Tensor<double> bias;
    Tensor<double> input_;
    Tensor<double> product_;

    int input_dims[4];
    int input_num_dims;
    int input_size;
    int output_dims[4];
    int output_num_dims;
    int output_size;

    double* d_weights;
    double* d_bias;
    double* d_in;
    double* d_out;

public:
    FullyConnected(int input_size, int output_size, int seed = 0);

    void setInputProps(int num_dims, int const *dims, int size);
    int getOutputNumDims() { return output_num_dims; };
    int* getOutputDims() { return output_dims; };
    int getOutputSize() { return output_size; };
    void setD_in(double* d_ptr) { d_in = d_ptr; };
    void setD_out(double* d_ptr) { d_out = d_ptr; };
    
    void forward();
    double* backprop(double* d_ptr, double learning_rate);

    Tensor<double> &forward(Tensor<double> &input) override;

    Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) override;

    void load(FILE *file_model) override;

    void save(FILE *file_model) override;
};


#endif //NEURAL_NET_IN_CPP_FULLYCONNECTED_H
