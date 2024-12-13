//
// Created by lucas on 11/04/19.
//

#include "../include/ReLU.h"

ReLU::ReLU() = default;

void ReLU::setInputDims(int num_dims, int const *dims, int size) {
    assert(num_dims > 0 && num_dims <= 4);
    // set input_dims
    input_dims[0] = dims[0];
    input_dims[1] = 1;
    for (int i = 1; i < num_dims; ++i) {
        input_dims[1] *= dims[i];
    }
    // set input_size
    input_size = size;

    // calculate output_dims
    copy(input_dims, input_dims+output_num_dims, output_dims);
    // calculate ouptut_size
    ouptut_size = input_size;
}

Tensor<double> &ReLU::forward(Tensor<double> &input) {
    input_ = input;
    product_ = input.relu();

    return product_;
}

Tensor<double> ReLU::backprop(Tensor<double> chainGradient, double learning_rate) {
    return chainGradient * input_.reluPrime();
}

void ReLU::load(FILE *file_model) {

}

void ReLU::save(FILE *file_model) {

}
