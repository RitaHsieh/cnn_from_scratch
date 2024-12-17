//
// Created by lucas on 10/04/19.
//

#ifndef NEURAL_NET_IN_CPP_MODULE_H
#define NEURAL_NET_IN_CPP_MODULE_H

#include "Tensor.h"

/*
 * Interface to be used as a building block for models
 */
class Module {
public:
    Tensor<double> input_;
protected:
    bool isEval = false;
public:
    virtual void setInputProps(int num_dims, int const *dims, int size) {
        throw std::runtime_error("This method is not supported for this type.");
    }

    virtual int getOutputNumDims() {
        throw std::runtime_error("This method is not supported for this type.");
    }

    virtual int* getOutputDims() {
        throw std::runtime_error("This method is not supported for this type.");
    }

    virtual int getOutputSize() {
        throw std::runtime_error("This method is not supported for this type.");
    }

    virtual void setD_in(double* d_ptr) {
        throw std::runtime_error("This method is not supported for this type.");
    }

    virtual void setD_out(double* d_ptr) {
        throw std::runtime_error("This method is not supported for this type.");
    }

    virtual void allocOutputDeviceMemory() {
        throw std::runtime_error("This method is not supported for this type.");
    }

    virtual Tensor<double> &initOutputTensor() {
        throw std::runtime_error("This method is not supported for this type.");
    }

    virtual Tensor<double> &forward(Tensor<double> &input) {
        throw std::runtime_error("This method is not supported for this type.");
    }

    virtual void forward() {
        throw std::runtime_error("This method is not supported for this type.");
    }
    
    virtual double * backprop(double * d_ptr, double learning_rate, bool test) {
        throw std::runtime_error("This method is not supported for this type.");
    }   

    virtual Tensor<double> backprop(Tensor<double> chainGradient, double learning_rate) {
        throw std::runtime_error("This method is not supported for this type.");
    }   
    
    virtual void load(FILE *file_model) = 0;

    virtual void save(FILE *file_model) = 0;

    void train();

    void eval();

    virtual ~Module() = default;
};

inline void Module::eval() {
    this->isEval = true;
}

inline void Module::train() {
    this->isEval = false;
}


#endif //NEURAL_NET_IN_CPP_MODULE_H
