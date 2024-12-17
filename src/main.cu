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

    if (argc < 2) {
        throw runtime_error("Please provide the data directory path as an argument");
    }
    printf("Data directory: %s\n", argv[1]);
    string data_path = argv[1];

    printf("Loading training set... ");
    // fflush(stdout);
    // cout << "Load image from: " << data_path + "/train-images-idx3-ubyte" << endl;
    MNISTDataLoader train_loader(data_path + "/train-images-idx3-ubyte", data_path + "/train-labels-idx1-ubyte", BATCH_SIZE);
    printf("Loaded.\n");

    int seed = 0;
    vector<Module *> modules = {new Conv2d(1, 8, 3, 1, 0, seed), new MaxPool(2, 2), new ReLU(), new FullyConnected(1352, 30, seed), new ReLU(),
                                new FullyConnected(30, 10, seed)};
    auto lr_sched = new LinearLRScheduler(0.2, -0.000005);
    NetworkModel model = NetworkModel(modules, new SoftmaxClassifier(), lr_sched);
    // model.load("network.txt");
    model.init(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT);

    int epochs = 1;
    printf("Training for %d epoch(s).\n", epochs);
    // Train network
    int num_train_batches = train_loader.getNumBatches();
    for (int k = 0; k < epochs; ++k) {
        // printf("Epoch %d\n", k + 1);
        for (int i = 0; i < num_train_batches; ++i) {
            pair<Tensor<double>, vector<int> > xy = train_loader.nextBatch();
            // cout << "before trainStep" << endl;
            double loss = model.trainStep(xy.first, xy.second);
            // cout << "after trainStep" << endl;
            if ((i + 1) % 10 == 0) {
                // printf("\rIteration %d/%d - Batch Loss: %.4lf", i + 1, num_train_batches, loss);
                // fflush(stdout);
            }
        }
        // printf("\n");
    }
    // Save weights
    model.save("network.txt");

    printf("Loading testing set... ");
    // fflush(stdout);
    MNISTDataLoader test_loader(data_path + "/t10k-images-idx3-ubyte", data_path + "/t10k-labels-idx1-ubyte", BATCH_SIZE);
    printf("Loaded.\n");

    model.eval();

    // Test and measure accuracy
    int hits = 0;
    int total = 0;
    printf("Testing...\n");
    int num_test_batches = test_loader.getNumBatches();
    for (int i = 0; i < num_test_batches; ++i) {
        if ((i + 1) % 10 == 0 || i == (num_test_batches - 1)) {
            printf("\rIteration %d/%d", i + 1, num_test_batches);
            // fflush(stdout);
        }
        pair<Tensor<double>, vector<int> > xy = test_loader.nextBatch();
        vector<int> predictions = model.predict(xy.first);
        for (int j = 0; j < predictions.size(); ++j) {
            if (predictions[j] == xy.second[j]) {
                hits++;
            }
        }
        total += xy.second.size();
    }
    printf("\n");

    end = getTimeStamp();
    printf("Total time: %.3f sec\n", end - start);

    printf("Accuracy: %.2f%% (%d/%d)\n", ((double) hits * 100) / total, hits, total);

    // cout << start - end << endl;
    return 0;
}