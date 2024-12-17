//
// Created by lucas on 11/04/19.
//

#include "../include/ReLU.cuh"

__global__ void forward_cuda(double* d_in, double* d_out, int input_dims_1) {
    
    // blockDim = 128
    // gridDim = batch_size // * ceil(input_dims[1]/128) 

    int tx = threadIdx.x;   
    // int diff_tx = blockDim.x; // # 128
    int bx = blockIdx.x;    // no. of batch

    for(int i = tx; i<input_dims_1; i+=128) {
        d_out[bx * input_dims_1 + i] = (d_in[bx * input_dims_1 + i]>0) ? d_in[bx * input_dims_1 + i] : 0;
        // d_out[bx * input_dims_1 + tx] = 1;
    }
}

__global__ void backprop_cuda(double* d_in, double* d_out, int input_dims_1) {

    // blockDim = 32
    // gridDim = batch_size

    int tx = threadIdx.x;   // no
    // int diff_tx = blockDim.x; // # 32
    int bx = blockIdx.x;    // no. of batch

    for(int i = tx; i<input_dims_1; i+=32) {
        d_in[bx * input_dims_1 + i] = (d_out[bx * input_dims_1 + i]>0) ? d_out[bx * input_dims_1 + i] : 0;
    }
}


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
    assert(size==input_dims[0]*input_dims[1]);
    input_size = size;

    // calculate output_dims
    output_dims[0] = input_dims[0];
    output_dims[1] = input_dims[1];
    //copy(input_dims, input_dims+output_num_dims, output_dims);
    // calculate ouptut_size
    output_size = input_size;
}

void ReLU::forward() {

    // dim3 grid(this->input_dims[0] * ceil(input_dims[1]/128));
    int grid = this->input_dims[0];
    forward_cuda<<<grid, 128>>>(this->d_in, this->d_out, this->input_dims[1]);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "RELU::forward::CUDA error: " << cudaGetErrorString(err) << std::endl;
    // }

    // test
    // Tensor<double> output_gpu(output_num_dims, output_dims);
    // cudaMemcpy(output_gpu.getData(), this->d_out, output_size, cudaMemcpyDeviceToHost);
    // for(int i = 0; i<10; i++) {
    //     std::cout << "test in relu:" << output_gpu.getData()[i] << std::endl;
    // }
    
}

Tensor<double> &ReLU::forward(Tensor<double> &input) {
    input_ = input;
    product_ = input.relu();

    return product_;
}

double* ReLU::backprop(double* d_ptr, double learning_rate, bool test) {
    this->d_out = d_ptr;
    backprop_cuda<<<this->input_dims[0], 32>>>(this->d_in, this->d_out, this->input_dims[1]);
    return this->d_in;
}

Tensor<double> ReLU::backprop(Tensor<double> chainGradient, double learning_rate) {
    std::cout << this->input_.num_dims << std::endl;
    return chainGradient * input_.reluPrime();
}

void ReLU::load(FILE *file_model) {

}

void ReLU::save(FILE *file_model) {

}
