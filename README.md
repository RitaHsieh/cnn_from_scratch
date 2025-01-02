# Neural Network in Pure C++

Simple modular implementation of a neural network in C++ using only the STL. 

### Installation
Get the MNIST data set:

```sh
bash get_mnist.sh
```
Generate your Makefile:
```sh
cmake -B build -S .
```
Make the code:
```sh
cmake --build build 
```
or
```sh
make
```
Make clean the code:
```sh
cmake --build build --target clean
```
Run:
```sh
srun -N1 -n1 --gres=gpu:1 ./neural_net_in_cpp
./neural_net_in_cpp data
```
Run validation for each layer
```sh
./build/neural_net_in_cpp_test [forward/backward] [layer idx]
```
result: 1 means correct!
`layer idx` should be in [0:5]
- layer0: cnn: kernel, bias, result
- layer1: maxpool: result
- layer2: ReLU: result
- layer3: FC: 
- layer4: ReLU
- layer5: FC
- (layer6, not accelerated): softmax
```c++
vector<Module *> modules = {
    new Conv2d(1, 8, 3, 1, 0, seed), 
    new MaxPool(2, 2), 
    new ReLU(), 
    new FullyConnected(1352, 30, seed), 
    new ReLU(), 
    new FullyConnected(30, 10, seed)
};
```

The training should take about a minute and achieve ~97% accuracy.

### Todos
 - [x] Fully connected;
 - [x] Sigmoid;
 - [x] Dropout;
 - [x] ReLU;
 - [ ] Tanh;
 - [ ] Leaky ReLU;
 - [ ] Batch normalization;
 - [x] Convolutional layers;
 - [x] Max pooling;
 - [ ] Other optimizers (Adam, RMSProp, etc);
 - [x] Learning rate scheduler;
 - [ ] Plots;
 - [ ] Filter visualization
 - [ ] CUDA?

License
----

MIT
