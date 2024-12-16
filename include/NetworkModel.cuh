//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_NETWORKMODEL_H
#define NEURAL_NET_IN_CPP_NETWORKMODEL_H

#include <vector>
#include "Tensor.h"
#include "Module.h"
#include "OutputLayer.h"
#include "../include/LRScheduler.h"

/*
 * Train and test a neural network defined by Modules
 */
class NetworkModel {
private:
    std::vector<Module *> modules_;
    OutputLayer *output_layer_;
    LRScheduler* lr_scheduler_;
    int iteration = 0;

    double* d_in;
    double* d_out;

    int output_num_dims;
    int* output_dims;
    int output_size;
public:
    NetworkModel(std::vector<Module *> &modules, OutputLayer *output_layer, LRScheduler* lr_scheduler);

    bool init(int batch_size, int image_width, int image_height);

    bool initForTest(int batch_size, int image_width, int image_height, int layer_idx, int seed);

    bool initForTest_backprop(int batch_size, int image_width, int image_height, int layer_idx, int seed);
    
    double trainStep(Tensor<double> &x, std::vector<int> &y);

    Tensor<double> forward(Tensor<double> &x);

    Tensor<double> forwardCUDA(Tensor<double> &x);

    std::vector<int> predict(Tensor<double> &x);

    void load(std::string path);

    void save(std::string path);

    virtual ~NetworkModel();

    void eval();
};


#endif //NEURAL_NET_IN_CPP_NETWORKMODEL_H
