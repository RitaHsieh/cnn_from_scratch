//
// Created by tabelini on 18/04/19.
//

#include "../include/MaxPool.h"

MaxPool::MaxPool(int size, int stride) {
    size_ = size;
    stride_ = stride;
}

MaxPool::void setInputProps(int num_dims, int const *dims, int size) {
    // set input_dims, output_dims, input_size, output_size
    for(int i=0; i<num_dims; i++) {
        input_dims[0] = dims[0];
    }

    input_size = size;

    output_dims[0] = input_dims[0];
    output_dims[1] = input_dims[1];
    output_dims[2] = ((input_dims[2] - (size_ - 1) - 1) / stride_) + 1;
    output_dims[3] = ((input_dims[3] - (size_ - 1) - 1) / stride_) + 1;
    
    output_size = 1;
    for(int i=0; i< num_dims; i++) {
        output_size *= output_dims[i];
    }

    cudaMalloc((void **) &d_indexes, output_size * sizeof(int));
}

__global__
void MaxPool_forward(
        double* d_out,
        double* d_in, 
        int* d_indexes,
        int N, int C,          // Batch size, number of filters, channels
        int Ho, int Wo,               // Output height and width
        int Hi, int Wi,               // Input height and width
        int size, int stride
    ) 
{
    int i = blockIdx.x; // batch index
    int j = blockIdx.y; // output volume index
    int k = threadIdx.x; // vertical index in the output volume
    int l = threadIdx.y; // horizontal index in the output volume

    if (k >= Ho || l >= Wo) return;

    double max = -999999999; // -infinity
    int index = 0;
    for (int m = 0; m < size; ++m) {
        for (int n = 0; n < size; ++n) {
            int input_y = k * stride + m;
            int input_x = l * stride + n;
            double value = d_in[((i*C +j)*Hi + input_y)*Wi + input_x];
            if (value > max) {
                index = m * size + n;
                max = value;
            }
        }
    }

    d_out[((i * C + j) * Ho + k) * Wo + l] = max;
    d_indexes[((i * C + j) * Ho + k) * Wo + l] = index;

}

void MaxPool::forward() {
    dim3 numBlocks(output_dims[0], output_dims[1]);
    dim3 threadsPerBlock(output_dims[2], output_dims[3]);
    //     int size, int stride
    MaxPool_forward<<<numBlocks, threadsPerBlock>>>( \
                d_out, d_in, d_indexes, \
                input_dims[0], input_dims[1], \
                output_dims[2], output_dims[3], input_dims[2], intput_dims[3], \
                size, stride \
            );
}

__global__
void MaxPool_backward(
        double* d_out,
        double* d_in, 
        int* d_indexes,
        int N, int C,          // Batch size, number of filters, channels
        int Ho, int Wo,               // Output height and width
        int Hi, int Wi,               // Input height and width
        int size, int stride )
{
    int i = blockIdx.x; // batch index
    int j = blockIdx.y; // output volume index
    int k = threadIdx.x; // vertical index in the input volume
    int l = threadIdx.y; // horizontal index in the input volume

    double input_gradient = 0;
    for(int hi = (k+1)-size; hi < k+size; hi += stride) {
        if(hi > 0 && hi < Hi) {
            for (int wi = (l+1) - size; wi < l + size; l += stride) {
                if (wi > 0 && wi < Wi) {
                    int ho = hi / stride;
                    int wo = wi / stride;
                    
                    int idx = d_indexes[((i * C + j) * Ho + ho )* Wo + wo];
                    int idx_x = idx / size;
                    int idx_y = idx % size;

                    if(hi + idx_x == k and wi + idx_y == l) {
                        input_gradient = d_out[((i * C + j) * Ho + ho )* Wo + wo];
                    }
                }
            }
        }
    }
    d_in[((i * C + j) * Hi + k)*Wi + l] = input_gradient;

}

double * MaxPool::backprop(double* d_chain_gradient, double learning_rate) {
    d_out = d_chain_gradient;
    dim3 numBlocks(input_dims[0], input_dims[1]);
    dim3 threadsPerBlock(input_dims[2], input_dims[3]);
    //     int size, int stride
    MaxPool_backward<<<numBlocks, threadsPerBlock>>>( \
                d_out, d_in, d_indexes, \
                input_dims[0], input_dims[1], \
                output_dims[2], output_dims[3], input_dims[2], intput_dims[3], \
                size, stride \
            );
    return d_in;
}

void MaxPool::load(FILE *file_model) {

}

void MaxPool::save(FILE *file_model) {

}
