//
// Created by lucas on 10/04/19.
//

#include "../include/NetworkModel.h"
#include "../include/LRScheduler.h"
#include "../include/Tensor.h"
#include "../include/Module.h"
#include "../include/Conv2d.cuh"

using namespace std;

NetworkModel::NetworkModel(std::vector<Module *> &modules, OutputLayer *output_layer, LRScheduler* lr_scheduler) {
    modules_ = modules;
    lr_scheduler_ = lr_scheduler;
    output_layer_ = output_layer;
}

// TODO: allocate global memory
double NetworkModel::trainStep(Tensor<double> &x, vector<int>& y) {
    // Forward
    Tensor<double> output = forwardCUDA(x);

    //Backprop
    // TODO: call backpropCUDA(y) instead
    pair<double, Tensor<double>> loss_and_cost_gradient = output_layer_->backprop(y);
    Tensor<double> chain_gradient = loss_and_cost_gradient.second;
    for (int i = (int) modules_.size() - 1; i >= 0; --i) {
        chain_gradient = modules_[i]->backprop(chain_gradient, lr_scheduler_->learning_rate);
    }
    ++iteration;
    lr_scheduler_->onIterationEnd(iteration);
    // Return loss
    return loss_and_cost_gradient.first;
}

// TODO: create CUDA version of forward that call for a loop of forwardCUDA(x)
Tensor<double> NetworkModel::forwardCUDA(Tensor<double> &x) {
    int size = x.dims[0] * x.dims[1] * x.dims[2] * x.dims[3] * sizeof(double); // input is double
    double *d_x, *d_out;

    cudaMalloc((void **) &d_x, size);
    cudaMemcpy(d_x, x.getData(), size, cudaMemcpyHostToDevice);
    // 全部都用好之後可以全部都用指標當作I/O
    // 但是現在暫時不行
    // modules_[0]->setInputPointer(d_x);
    Conv2d* conv_layer = dynamic_cast<Conv2d*>(modules_[0]);
    d_out = conv_layer->forward(x, d_x);

    int output_size = conv_layer->getOutputSize();
    x = conv_layer->initOutputTensor();
    cudaMemcpy(x.getData(), d_out, output_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_out); // TODO: general
    for (int i=1; i < (int) modules_.size(); ++i) {
        x = modules_[i]->forward(x);
    }
    cudaFree(d_x); // TODO: general
    return output_layer_->predict(x);

}

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