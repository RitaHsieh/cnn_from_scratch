#include <iostream>
#include <sys/time.h>
#include "../include/NetworkModel.cuh"
#include "../include/Module.h"
#include "../include/FullyConnected.cuh"
#include "../include/Sigmoid.h"
#include "../include/Dropout.h"
#include "../include/SoftmaxClassifier.h"
#include "../include/MNISTDataLoader.h"
#include "../include/ReLU.cuh"
#include "../include/Tensor.h"
#include "../include/Conv2d.cuh" // change from h to cuh
#include "../include/MaxPool.cuh"
#include "../include/LinearLRScheduler.h"

using namespace std;

/*
 * Train a neural network on the MNIST data set and evaluate its performance
 */
const int BATCH_SIZE = 32;
const int IMAGE_HEIGHT = 28;
const int IMAGE_WIDTH = 28;

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int main(int argc, char **argv) {
    double start, end;
    start = getTimeStamp();

    if (argc < 3) {
        throw runtime_error("Please provide the data directory path as an argument");
    }

    int seed = 0;

    vector<Module *> modules = {new Conv2d(1, 8, 3, 1, 0, seed), new MaxPool(2, 2), new ReLU(), new FullyConnected(1352, 30, seed), new ReLU(),
                                new FullyConnected(30, 10, seed)};
    
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        std::cerr << "CUDA error1: " << cudaGetErrorString(err1) << std::endl;
    }
    else {
        std::cout << "module init successfully" << std::endl;
    }
    
    int num_modules = modules.size();

    auto lr_sched = new LinearLRScheduler(0.2, -0.000005);
    NetworkModel model = NetworkModel(modules, new SoftmaxClassifier(), lr_sched);
    
    // Test 
    int backprop = atoi(argv[1]);   // forward: 0, backprop: 1 
    assert(backprop==0 || backprop==1);

    int layer_idx = atoi(argv[2]);
    assert(layer_idx < num_modules);

    // model.init(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT);
    bool result = false;
    if(backprop==0) {
        result = model.initForTest(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, layer_idx, seed);
    }
    else{
        result = model.initForTest_backprop(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, layer_idx, seed);
    }
    
    cout << "result: " << result << endl;

    end = getTimeStamp();
    printf("Total time: %.3f sec\n", end - start);

    // cout << start - end << endl;
    return 0;
}