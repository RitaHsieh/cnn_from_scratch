//
// Created by lucas on 10/04/19.
//

#include "../include/NetworkModel.cuh"
#include "../include/LRScheduler.h"
#include "../include/Tensor.h"
#include "../include/Module.h"
#include "../include/Conv2d.cuh"

using namespace std;

void CHECK(cudaError_t err, int line=0) {
    if(err != cudaSuccess) {
        cout << line << ":cuda error: " << err << endl;
    }
}

NetworkModel::NetworkModel(std::vector<Module *> &modules, OutputLayer *output_layer, LRScheduler* lr_scheduler) {
    modules_ = modules;
    lr_scheduler_ = lr_scheduler;
    output_layer_ = output_layer;
}

bool NetworkModel::init(int batch_size, int image_width, int image_height) {
    // two things to be done:
    // 1. allocate device memory for input, output and the data between the layers
    // 2. set dimension for each layer
    int num_dims = 4;
    int* dims = new int[4]{batch_size, 1, image_width, image_height};
    int size = batch_size * 1 * image_width * image_height;
    double* d_ptr;
    cudaMalloc((void **)&d_ptr, size * sizeof(double));
    this->d_in = d_ptr;
    for(auto &layer: modules_) {
        layer->setInputProps(num_dims, dims, size);
        layer->setD_in(d_ptr);
        num_dims = layer->getOutputNumDims();
        dims = layer->getOutputDims();
        size = layer->getOutputSize();
        cudaMalloc((void **)&d_ptr, size * sizeof(double));
        layer->setD_out(d_ptr);
    }
    this->output_num_dims = num_dims;
    this->output_dims = dims;
    this->output_size = size;
    this->d_out = d_ptr;

    return true;
}


bool NetworkModel::initForTest(int batch_size, int image_width, int image_height, int layer_idx, int seed) {
    int num_dims = 4;
    int* dims = new int[4]{batch_size, 1, image_width, image_height};
    int size = batch_size * 1 * image_width * image_height;
    double* d_ptr;
    CHECK(cudaMalloc((void **)&d_ptr, size * sizeof(double)), 58);
    
    this->d_in = d_ptr;

    int i = 0;
    for(auto &layer: modules_) {
        cout << "init layer: " << i << " with input size: " << size << endl;
        layer->setInputProps(num_dims, dims, size);
        layer->setD_in(d_ptr);
        
        if(i++ == layer_idx) {
            
            // create a fake input
            std::default_random_engine generator(seed);
            std::normal_distribution<double> distribution(0.0, 1.0);
            Tensor<double> input(num_dims, dims);
            input.randn(generator, distribution, sqrt(2.0 / size));
            CHECK(cudaMemcpy(d_ptr, input.getData(), size* sizeof(double), cudaMemcpyHostToDevice), 75);
            // test cpu version
            Tensor<double> output_cpu = layer->forward(input);
            // test gpu version
            //      alloc for output
            num_dims = layer->getOutputNumDims();
            dims = layer->getOutputDims();
            size = layer->getOutputSize();
            CHECK(cudaMalloc((void **)&d_ptr, size * sizeof(double)), 83);
            layer->setD_out(d_ptr);
            //      run CUDA ver.
            Tensor<double> output_gpu(num_dims, dims);
            layer->forward();
            CHECK(cudaMemcpy(output_gpu.getData(), d_ptr, size* sizeof(double), cudaMemcpyDeviceToHost), 88);
            cout << "test:" << output_gpu.getData()[2] << endl;
            // compare results from cpu and gpu versions
            return (output_cpu==output_gpu);
        }
        
        num_dims = layer->getOutputNumDims();
        dims = layer->getOutputDims();
        size = layer->getOutputSize();
        CHECK(cudaMalloc((void **)&d_ptr, size* sizeof(double)), 97);
        layer->setD_out(d_ptr);
    }
    this->output_num_dims = num_dims;
    this->output_dims = dims;
    this->output_size = size;
    this->d_out = d_ptr;

    return true;
}

bool NetworkModel::initForTest_backprop(int batch_size, int image_width, int image_height, int layer_idx, int seed) {
    double learning_rate = 0.2;
    
    int num_dims = 4;
    int* dims = new int[4]{batch_size, 1, image_width, image_height};
    int size = batch_size * 1 * image_width * image_height;
    double* d_ptr;
    CHECK(cudaMalloc((void **)&d_ptr, size * sizeof(double)), 115);
    this->d_in = d_ptr;

    int i = 0;
    for(auto &layer: modules_) {
        cout << "init layer: " << i << " with input size: " << size << endl;
        layer->setInputProps(num_dims, dims, size);
        layer->setD_in(d_ptr);

        if(i++ == layer_idx) {
            // create a fake input
            std::default_random_engine generator(seed);
            std::normal_distribution<double> distribution(0.0, 1.0);
            Tensor<double> input(num_dims, dims);
            input.randn(generator, distribution, sqrt(2.0 / size));
            cout << "fake input[0]:" << input.getData()[0] << endl;
            Tensor<double> output_cpu = layer->forward(input);
            CHECK(cudaMemcpy(d_ptr, input.getData(), size* sizeof(double), cudaMemcpyHostToDevice), 131);
            
            // alloc for output
            int out_num_dims = layer->getOutputNumDims();
            cout << out_num_dims << endl;
            int* out_dims = layer->getOutputDims();
            int out_size = layer->getOutputSize();
            CHECK(cudaMalloc((void **)&d_ptr, out_size * sizeof(double)), 138);
            layer->setD_out(d_ptr);

            // create a fake inputGradient
            std::default_random_engine generator1(++seed);
            // std::normal_distribution<double> distribution(0.0, 1.0);
            Tensor<double> inputGradient(out_num_dims, out_dims);
            inputGradient.randn(generator1, distribution, sqrt(2.0 / out_size));
            CHECK(cudaMemcpy(d_ptr, inputGradient.getData(), out_size * sizeof(double), cudaMemcpyHostToDevice), 147);
            cout << "fake inputGradient[0]:" << inputGradient.getData()[0] << endl;
            // test cpu version
            
            Tensor<double> outputGradient_cpu = layer->backprop((Tensor<double>)inputGradient, learning_rate);
            cout << "Finish CPU version: result[0]:" << outputGradient_cpu.getData()[1] << endl;
            // test gpu version
            //      run CUDA ver.
            Tensor<double> outputGradient_gpu(num_dims, dims);
            d_ptr = layer->backprop((double*)d_ptr, learning_rate);
            CHECK(cudaMemcpy(outputGradient_gpu.getData(), d_ptr, size * sizeof(double), cudaMemcpyDeviceToHost), 157);
            cout << "Finish GPU version: result[0]:" << outputGradient_gpu.getData()[1] << endl;
            
            // test bias & weight
            
            // update
            num_dims = output_num_dims;
            dims = output_dims;
            size = output_size;
            
            // compare results from cpu and gpu versions
            return (outputGradient_cpu==outputGradient_gpu);
        }
        
        num_dims = layer->getOutputNumDims();
        dims = layer->getOutputDims();
        size = layer->getOutputSize();
        CHECK(cudaMalloc((void **)&d_ptr, size * sizeof(double)), 169);
        layer->setD_out(d_ptr);
    }
    this->output_num_dims = num_dims;
    this->output_dims = dims;
    this->output_size = size;
    this->d_out = d_ptr;

    return true;
}

double NetworkModel::trainStep(Tensor<double> &x, vector<int>& y) {
    // Forward
    Tensor<double> output = forwardCUDA(x);
    //cout << "after forwardCUDA" << endl;
    //Backprop
    pair<double, Tensor<double>> loss_and_cost_gradient = output_layer_->backprop(y);
    Tensor<double> chain_gradient = loss_and_cost_gradient.second;
    cudaMemcpy(this->d_out, chain_gradient.getData(), this->output_size * sizeof(double), cudaMemcpyHostToDevice);
    double* d_update_ptr = this->d_out;
    for (int i = (int) modules_.size() - 1; i >= 0; --i) {
        cout << "it:" << iteration <<", backprop in no. " << i << " layer" << endl;
        d_update_ptr = modules_[i]->backprop(d_update_ptr, lr_scheduler_->learning_rate);
    }
    //cout << "after backpropCUDA" << endl;
    ++iteration;
    lr_scheduler_->onIterationEnd(iteration);
    // Return loss
    return loss_and_cost_gradient.first;
}

Tensor<double> NetworkModel::forwardCUDA(Tensor<double> &x) {
    cudaMemcpy(this->d_in, x.getData(), x.getSize()*sizeof(double), cudaMemcpyHostToDevice);
    for (auto &module : modules_) {
        module->forward();
    }
    Tensor<double> y = Tensor<double>(this->output_num_dims, output_dims);
    cudaMemcpy(y.getData(), this->d_out, this->output_size * sizeof(double), cudaMemcpyDeviceToHost);
    return output_layer_->predict(y);
}

// This will be the forwardCUDA for general version

// Tensor<double> NetworkModel::forwardCUDA(Tensor<double> &x) {
//     int size = x.dims[0] * x.dims[1] * x.dims[2] * x.dims[3] * sizeof(double); // input is double
//     double *d_x;

//     cudaMalloc((void **) &d_x, size);
//     cudaMemcpy(d_x, x.getData(), size, cudaMemcpyHostToDevice);
    
//     int *input_dim = x.dims;
//     for (auto &module : modules_) {
//         d_x = module->forward(input_dim, d_x);
//         input_dim = module->getOutputDim(); // d1, d2, d3, d4
//     }
//     int output_size = modules_[-1]->getOutputSize();
//     x = modules_[-1]->initOutputTensor();
//     cudaMemcpy(x.getData(), d_x, output_size, cudaMemcpyDeviceToHost);
    
//     cudaFree(d_x);
//     return output_layer_->predict(x);
// }

// Take conv2d as example: see Conv2d.cpp
Tensor<double> NetworkModel::forward(Tensor<double> &x) {
    for (auto &module : modules_) {
        x = module->forward(x);
    }
    return output_layer_->predict(x);
}

std::vector<int> NetworkModel::predict(Tensor<double> &x) {
    Tensor<double> output = forward(x);
    std::vector<int> predictions;
    for (int i = 0; i < output.dims[0]; ++i) {
        int argmax = -1;
        double max = -1;
        for (int j = 0; j < output.dims[1]; ++j) {
            if (output.get(i, j) > max) {
                max = output.get(i, j);
                argmax = j;
            }
        }
        predictions.push_back(argmax);
    }

    return predictions;
}

void NetworkModel::load(std::string path) {
    FILE *model_file = fopen(path.c_str(), "r");
    if (!model_file) {
        throw std::runtime_error("Error reading model file.");
    }
    for (auto &module : modules_) {
        module->load(model_file);
    }
}

void NetworkModel::save(std::string path) {
    FILE *model_file = fopen(path.c_str(), "w");
    if (!model_file) {
        throw std::runtime_error("Error reading model file.");
    }
    for (auto &module : modules_) {
        module->save(model_file);
    }
}

NetworkModel::~NetworkModel() {
    for (auto &module : modules_) {
        delete module;
    }
    delete output_layer_;
    delete lr_scheduler_;
}

void NetworkModel::eval() {
    for (auto &module : modules_) {
        module->eval();
    }
}
