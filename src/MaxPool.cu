//
// Created by tabelini on 18/04/19.
//

#include "../include/MaxPool.cuh"

MaxPool::MaxPool(int size, int stride) {
    size_ = size;
    stride_ = stride;
}

void MaxPool::setInputProps(int num_dims, int const *dims, int size) {
    // set input_dims, output_dims, input_size, output_size
    for(int i=0; i<num_dims; i++) {
        input_dims[i] = dims[i];
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

    cudaMalloc((void **)&d_indexes, output_size * sizeof(int));
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
                output_dims[2], output_dims[3], input_dims[2], input_dims[3], \
                size_, stride_ \
    );

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "MaxPool::forward::CUDA error: " << cudaGetErrorString(err) << std::endl;
    // }

    //test indexes
    // Tensor<int> indexes_gpu = this->indexes;
    // indexes_gpu.zero();
    // cudaMemcpy(indexes_gpu.getData(), d_indexes, indexes_gpu.getSize() * sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "test indexes: " << (indexes_gpu==this->indexes) << std::endl;

    
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
    int ho, wo, idx, idx_x, idx_y;
    int hStart = (k-(size-1)%stride) == 0 ?  k-(size-1) : ((k-(size-1)/stride) + 1) * stride;
    int hEnd = (int)(k/stride)*stride;
    int wStart = (l-(size-1)%stride) == 0 ?  l-(size-1) : ((l-(size-1)/stride) + 1) * stride;
    int wEnd = (int)(l/stride)*stride;
    for(int hi = hStart; hi < hEnd; hi += stride) {
        if(hi > 0 && hi < Hi) {
            for (int wi = wStart; wi < wEnd; wi += stride) {
                if (wi > 0 && wi < Wi) {
                    ho = hi / stride;
                    wo = wi / stride;
    
                    idx = d_indexes[((i * C + j) * Ho + ho )* Wo + wo];
                    idx_x = idx / size;
                    idx_y = idx % size;

                    if((hi + idx_x) == k && (wi + idx_y) == l) {
                        input_gradient = d_out[((i * C + j) * Ho + ho )* Wo + wo];
                    }
                }
            }
        }
    }
    d_in[((i * C + j) * Hi + k)*Wi + l] = input_gradient;

}

__global__  void backprop_cuda(
    double* d_out, double* d_in, int* d_indexes, \
    int input_dims0, int input_dims1, int input_dims2, int input_dims3,
    int output_dims2, int output_dims3, \
    int size, int stride
) {
    // gridDim: [input_dims0, input_dims1]
    // blockDim: [output_dims2, output_dims3]

    // d_in: [input_dims0, input_dims1, input_dims2, input_dims3]
    // d_indexes & d_out: [output_dims0, output_dims1, output_dims2, output_dims3]

    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;
    int l = threadIdx.y;

    int index = d_indexes[((i*input_dims1+j)*output_dims2+k)*output_dims3+l];  // [i][j][k][l];
    int m = index / size;
    int n = index % size;
    int input_y = k * stride + m;
    int input_x = l * stride + n;
    d_in[((i*input_dims1+j)*input_dims2+input_y)*input_dims3+input_x] = d_out[((i*input_dims1+j)*output_dims2+k)*output_dims3+l];
    // d_in[i][j][input_y][input_x] = d_out[bx][by][tx][ty];
}

double * MaxPool::backprop(double* d_chain_gradient, double learning_rate, bool test) {
    this->d_out = d_chain_gradient;
    if(test) {
        cudaMemcpy(d_indexes, this->indexes.getData(), this->indexes.getSize() * sizeof(int), cudaMemcpyHostToDevice);
    }

    // dim3 numBlocks(input_dims[0], input_dims[1]);
    // dim3 threadsPerBlock(input_dims[2], input_dims[3]);
    //     int size, int stride
    // MaxPool_backward<<<numBlocks, threadsPerBlock>>>( \
    //             d_out, d_in, d_indexes, \
    //             input_dims[0], input_dims[1], \
    //             output_dims[2], output_dims[3], input_dims[2], input_dims[3], \
    //             size_, stride_ \
    //         );

    dim3 grid(input_dims[0], input_dims[1]);
    dim3 block(output_dims[2], output_dims[3]);
    // cudaMemset(this->d_indexes, 0, this->indexes.getSize() * sizeof(int));
    backprop_cuda<<<grid, block>>>(
        this->d_out, this->d_in, this->d_indexes, \
        this->input_dims[0], this->input_dims[1], this->input_dims[2], this->input_dims[3],
        this->output_dims[2], this->output_dims[3], \
        this->size_, this->stride_
    );
    
    if(test) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "MaxPool::backprop::CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
    }
    return this->d_in;
}



Tensor<double> &MaxPool::forward(Tensor<double> &input) {
    int w = ((input.dims[3] - (size_ - 1) - 1) / stride_) + 1;
    int h = ((input.dims[2] - (size_ - 1) - 1) / stride_) + 1;
    int dims[] = {input.dims[0], input.dims[1], h, w};
    output_ = Tensor<double>(4, dims);
    indexes = Tensor<int>(4, dims);
    for (int i = 0; i < input.dims[0]; ++i) { // for each batch image
        for (int j = 0; j < input.dims[1]; ++j) { // for each image channel
            for (int k = 0; k < dims[2]; ++k) { // for each output y
                for (int l = 0; l < dims[3]; ++l) { // for each output x
                    double max = -999999999; // -infinity
                    int index = 0;
                    for (int m = 0; m < size_; ++m) {
                        for (int n = 0; n < size_; ++n) {
                            int input_y = k * stride_ + m;
                            int input_x = l * stride_ + n;
                            double value = input.get(i, j, input_y, input_x);
                            if (value > max) {
                                index = m * size_ + n;
                                max = value;
                            }
                        }
                    }
                    output_.set(i, j, k, l, max);
                    indexes.set(i, j, k, l, index);
                }
            }
        }
    }
    input_ = input;

    return output_;
}

Tensor<double> MaxPool::backprop(Tensor<double> chainGradient, double learning_rate) {
    Tensor<double> input_gradient(input_.num_dims, input_.dims);
    input_gradient.zero();

    for (int i = 0; i < input_.dims[0]; ++i) { // for each batch image
        for (int j = 0; j < input_.dims[1]; ++j) { // for each image channel
            for (int k = 0; k < output_.dims[2]; ++k) { // for each output y
                for (int l = 0; l < output_.dims[3]; ++l) { // for each output x
                    double chain_grad = chainGradient.get(i, j, k, l);
                    int index = indexes.get(i, j, k, l);
                    int m = index / size_;
                    int n = index % size_;
                    int input_y = k * stride_ + m;
                    int input_x = l * stride_ + n;
                    input_gradient.set(i, j, input_y, input_x, chain_grad);
                }
            }
        }
    }

    return input_gradient;
}

void MaxPool::load(FILE *file_model) {

}

void MaxPool::save(FILE *file_model) {

}
