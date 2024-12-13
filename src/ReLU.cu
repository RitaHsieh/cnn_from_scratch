//
// Created by lucas on 11/04/19.
//

#include "../include/ReLU.cuh"

ReLU::ReLU() = default;

void ReLU::setInputProps(int num_dims, int const *dims, int size) {
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

void ReLU::forward() {
    forward_cuda<<<this->input_dims[0], 32>>>(this->d_in, this->d_out, this->input_size);
}

__global__ void forward_cuda(double* d_in, double* d_out, int input_size) {
    
    // blockDim = 32
    // gridDim = batch_size

    int tx = threadIdx.x;   // no
    // int diff_tx = blockDim.x; // # 32
    int bx = blockIdx.x;    // no. of batch

    for(int i = tx; i<input_size; i+=32) {
        d_out[bx * input_size + tx] = (d_in[bx * input_size + tx]>0) ? d_in[bx * input_size + tx]>0 : 0;
    }
}

Tensor<double> &ReLU::forward(Tensor<double> &input) {
    input_ = input;
    product_ = input.relu();

    return product_;
}

void ReLU::backprop() {
    backprop_cuda<<<this->input_dims[0], 32>>>(this->d_in, this->d_out, this->input_size);
}

__global__ void backprop_cuda(double* d_in, double* d_out, int input_size) {

    // blockDim = 32
    // gridDim = batch_size

    int tx = threadIdx.x;   // no
    // int diff_tx = blockDim.x; // # 32
    int bx = blockIdx.x;    // no. of batch

    for(int i = tx; i<input_size; i+=32) {
        d_in[bx * input_size + tx] = (d_out[bx * input_size + tx]>0) ? d_out[bx * input_size + tx]>0 : 0;
    }
}

Tensor<double> ReLU::backprop(Tensor<double> chainGradient, double learning_rate) {
    return chainGradient * input_.reluPrime();
}

void ReLU::load(FILE *file_model) {

}

void ReLU::save(FILE *file_model) {

}
