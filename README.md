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
Run:
```sh
srun -N1 -n1 --gres=gpu:1 ./neural_net_in_cpp
./neural_net_in_cpp data
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
