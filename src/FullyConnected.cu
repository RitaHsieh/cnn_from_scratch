//
// Created by lucas on 10/04/19.
//

#include <random>
#include "../include/FullyConnected.cuh"
#include "../include/Tensor.h"

FullyConnected::FullyConnected(int input_size, int output_size, int seed) {
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
    int weights_dims[] = {input_size, output_size};
    weights = Tensor<double>(2, weights_dims);
    weights.randn(generator, distribution, sqrt(2.0 / input_size));

    cudaMalloc((void **)&d_weights, weights.getSize()*sizeof(double));
    cudaMemcpy(&d_weights, weights.getData(), weights.getSize()*sizeof(double), cudaMemcpyHostToDevice);

    int bias_dims[] = {output_size};
    bias = Tensor<double>(1, bias_dims);
    bias.randn(generator, distribution, 0);

    cudaMalloc((void **)&d_bias, bias.getSize()*sizeof(double));
    cudaMemcpy(&d_bias, bias.getData(), bias.getSize()*sizeof(double), cudaMemcpyHostToDevice);
}

void FullyConnected::setInputProps(int num_dims, int const *dims, int size) {
    // set input_dims
    input_dims[0] = dims[0];
    // flatten
    int flatten_size = 1;
    for (int i = 1; i < num_dims; ++i) {
        flatten_size *= dims[i];
    }
    input_dims[1] = flatten_size;

    // set input_size
    input_size = size;

    // calculate output_dims
    output_dims[0] = dims[0];
    output_dims[1] = weights.dims[1];

    // calculate ouptut_size
    output_size = output_dims[0] * output_dims[1];
}

void FullyConnected::forward() {
    dim3 block(32, this->output_dims[1]);
    forward_cuda<<<ceil(this->input_dims[0], 32), >>>(          \
        this->d_in, this->d_out, this->d_weights, this->d_bias,  \
        this->input_dims[1], this->input_dims[0]);
}

__global__ void forward_cuda(
    double* d_in, double* d_out, double* d_weights, double* d_bias, 
    int input_dim, int d
    ) {

    // blockDim = (32, output_dim)
    // gridDim = batch_size/32

    // d_in: {batch_size, input_dim}
    // d_weights: {input_dim, output_dim}
    // d_bias: {output_dim}
    // d_out: {batch_size, output_dim}

    // no. of batch = bx * 32 + tx;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;

    double value = 0;
    // based on whethere an strict on batch_size
    // if(bx*32+tx > input_dim) {
    //     break;
    // }
    int block_offset = (bx*32+tx)*d;
    for(int k = 0; k < d; k++) {
        value += d_in[block_offset + k] * d_weights[k*d + ty];
    }
    d_out[(bx*32+tx)*blockDim.y + ty] = value + d_bias[ty];
}


Tensor<double> &FullyConnected::forward(Tensor<double> &input) {
    input_num_dims = input.num_dims;
    std::copy(input.dims, input.dims + input.num_dims, input_dims);
    if (input.num_dims != 2) {
        // flatten tensor
        int flatten_size = 1;
        for (int i = 1; i < input.num_dims; ++i) {
            flatten_size *= input.dims[i];
        }
        int dims[] = {input.dims[0], flatten_size};
        input.view(2, dims);
    }
    input_ = input;
    product_ = input.matmul(weights) + bias;

    return product_;
}

double* FullyConnected::backprop(double* d_ptr, double learning_rate) {
    this->d_out = d_ptr;
    double *d_in_new, *d_weights_new;
    cudaMalloc((void**)&d_in_new, this->input_size);
    cudaMalloc((void**)&d_weights_new, this->weights->getSize());

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // calculate sram size
    //int sram_size_weightsGradient = this->weights->getSize() * sizeof(double);
    //int sram_size_biasGradient =  this->bias->getSize() * sizeof(double);
    //int sram_size_input = this->input_size * sizeof(double);

    backprop_cuda_weights_and_bias<<<grid, block, 0, streams[0]>>>(
        this->d_in, this->d_out, this->d_weights, d_weights_new, this->d_bias,
        learning_rate, this->input_dims[1], this->input_dims[0], this->output_dims[1]
    );
    backprop_cuda_input<<<grid, block, 0, streams[1]>>>(
        d_in_new, this->d_out, this->d_weights, 
        learning_rate, this->input_dims[0], this->input_dims[1], this->output_dims[1]
    );

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    cudaFree(this->d_in);
    cudaFree(this->d_out);

    this->d_in = d_in_new;
    this->d_weights = d_weights_new;

    return this->d_in;
}

__global__ void backprop_cuda_weights_and_bias(
    double* d_in, double* d_out, double* d_weights, double* d_weights_new, double* d_bias,
    double learning_rate, int input_dims_1, int d, int output_dims_1) {
    // i: input_dims[0]=output_dims[0](batch_size), k: input_dims[1], j:output_dims[1] 
    // blockDim = (32, 32)
    // gridDim = (ceil(k/32), ceil(j/32))

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //extern __shared__ double sram[];
    //double* weightGradient = &sram;

    int k = bx * 32 + tx;  
    int j = by * 32 + ty;
    if(k>=input_dims_1 || j>=output_dims_1) {
        return;
    }

    double* weightGradient = 0;
    double* biasGradient = 0;
    for(int i = 0; i<d; i++) {
        if(k==0) {
            biasGradient += d_out[i*output_dims_1*j];
        }
        weightGradient += d_in[i*input_dims_1 + k] * d_out[i*output_dims_1*j];
    }
    d_weights_new[k*output_dims_1 + j] -= learning_rate * weightGradient;
    if(k==0) {
        d_bias[j] -= learning_rate * biasGradient;
    }
}

__global__ void backprop_cuda_input(
    double* d_in_new, double* d_out, double* d_weights,
    double learning_rate, int b, int input_dims_1, int output_dims_1) {
    // i: input_dims[0]=output_dims[0](batch_size), k: input_dims[1], j:output_dims[1] 
    // blockDim = (32, 32)
    // gridDim = (ceil(i/32), ceil(j/32))

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int i = bx * 32 + tx; 
    int k = by * 32 + ty;

    if(i>=b || k>=input_dims_1) {
        return;
    }

    double* inputGradient = 0;
    for(int j = 0; j<output_dims_1; i++) {
        inputGradient += d_out[i*output_dims_1 + j] * d_weights[k*output_dims_1 + j];
    }
    d_in_new[i*input_dims_1 + k] -=  learning_rate * inputGradient;
}

Tensor<double> FullyConnected::backprop(Tensor<double> chainGradient, double learning_rate) {
    Tensor<double> weightGradient = input_.matrixTranspose().matmul(chainGradient);
    Tensor<double> biasGradient = chainGradient.columnWiseSum();
    chainGradient = chainGradient.matmul(weights.matrixTranspose());
    chainGradient.view(input_num_dims, input_dims);
    weights -= weightGradient * learning_rate;
    bias -= biasGradient * learning_rate;
    return chainGradient;
}

void FullyConnected::load(FILE *file_model) {
    double value;
    for (int i = 0; i < weights.dims[0]; ++i) {
        for (int j = 0; j < weights.dims[1]; ++j) {
            int read = fscanf(file_model, "%lf", &value); // NOLINT(cert-err34-c)
            if (read != 1) throw std::runtime_error("Invalid model file");
            weights.set(i, j, value);
        }
    }

    for (int i = 0; i < bias.dims[0]; ++i) {
        int read = fscanf(file_model, "%lf", &value); // NOLINT(cert-err34-c)
        if (read != 1) throw std::runtime_error("Invalid model file");
        bias.set(i, value);
    }
}

void FullyConnected::save(FILE *file_model) {
    for (int i = 0; i < weights.dims[0]; ++i) {
        for (int j = 0; j < weights.dims[1]; ++j) {
            fprintf(file_model, "%.18lf ", weights.get(i, j));
        }
    }

    for (int i = 0; i < bias.dims[0]; ++i) {
        fprintf(file_model, "%.18lf ", bias.get(i));
    }
}

// FullyConnected::~FullyConnected() {
// }