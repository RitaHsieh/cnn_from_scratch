#include "../include/Conv2d.cuh"

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, int seed) {
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    int kernel_dims[] = {out_channels, in_channels, kernel_size, kernel_size};
    kernels = Tensor<double>(4, kernel_dims);
    kernels.randn(generator, distribution, sqrt(2.0 / (kernel_size * kernel_size * out_channels)));

    int bias_dims[] = {out_channels};
    bias = Tensor<double>(1, bias_dims);
    bias.randn(generator, distribution, 0);

    // allocate memory for kernel
    size_t k_mem_size = kernels.dims[0] * kernels.dims[1] * kernels.dims[2] * kernels.dims[3] * sizeof(double);
    cudaMalloc((void **) &d_kernel, k_mem_size);
    cudaMemcpy(d_kernel, kernels.getData(), k_mem_size, cudaMemcpyHostToDevice);

    // allocate memory for bias
    size_t bias_size = bias.dims[0] * sizeof(double);
    cudaMalloc((void **) &d_bias, bias_size);
    cudaMemcpy(d_bias, bias.getData(), bias_size, cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Conv2d::CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    this->stride = stride;
    this->padding = padding;
}

void Conv2d::setInputProps(int num_dims, int const *dims, int size) {
    // set input_dims, output_dims, input_size, output_size
    for(int i=0; i<num_dims; i++) {
        input_dims[i] = dims[i];
    }

    input_size = size;

    output_dims[0] = input_dims[0];
    output_dims[1] = kernels.dims[0];
    output_dims[2] = ((input_dims[2] + 2 * padding - (kernels.dims[2] - 1) - 1) / stride) + 1;
    output_dims[3] = ((input_dims[3] + 2 * padding - (kernels.dims[3] - 1) - 1) / stride) + 1;
    
    output_size = 1;
    for(int i=0; i< num_dims; i++) {
        output_size *= output_dims[i];
    }
}

__global__
void Conv2d_gpu(
    int stride, int padding, \
    double* d_kernel, \
    double* d_in, double* d_out, \
    double* d_bias,\
    int N, int F, int C,\
    int Hk, int Wk, int Hi, int Wi, int Ho, int Wo ) 
{
    // Block indices
    int i = blockIdx.x; // batch index
    int j = blockIdx.y; // output volume index
    int k = threadIdx.x; // vertical index in the output volume
    int l = threadIdx.y; // horizontal index in the output volume

    int im_si = stride * k - padding; // height
    int im_sj = stride * l - padding; // width
    double total = 0.0;
    double a, b;
    for (int m = 0; m < C; ++m) { 
        for (int n = 0; n < Hk; ++n) {
            for (int o = 0; o < Wk; ++o) {
                int x = im_si + n, y = im_sj + o;
                if (x < 0 || x >= Hi || y < 0 || y >= Wi)
                    continue; 
                a = d_in[((i * C + m)*Hi + x) * Wi + y];
                b = d_kernel[((j * C + m)*Hk + n) * Wk+ o];
                total += a * b;
            }
        }
    }
    total = total + d_bias[j];
    d_out[((i * F+ j) * Ho + k) * Wo + l] = total;
}

__global__
void backprop_cuda_input(
    int stride, int padding, \
    double* d_out, double* d_kernel, double* d_input_temp, double* d_in, \
    int N, int F, int C, \
    int Hk, int Wk, int Hi, int Wi, int Ho, int Wo, \
    double learning_rate ) 
{
    // Block indices
    int i = blockIdx.x; // batch index
    int j = blockIdx.y; // output volume index
    int k = threadIdx.x; // vertical index in the output volume
    int l = threadIdx.y; // horizontal index in the output volume

    int im_si = stride * k - padding; // height -25~3
    int im_sj = stride * l - padding; // width
    double total = 0.0;
    double a, b;
    
    for (int m = 0; m < F; ++m) { 
        for (int n = 0; n < Ho; ++n) {
            for (int o = 0; o < Wo; ++o) {
                int x = im_si + n; // -25~0
                int y = im_sj + o;
                if (x < 0 || x >= Hk || y < 0 || y >= Wk)
                    continue; 
                // a = d_kernel[m][0][Hk-1-x][Wk-1-y];
                a = d_kernel[((m * C)*Hk + Hk-1-x) * Wk + Wk-1-y];
                // a = d_kernel[((m * C)*Hk + x) * Wk + y];

                // b = d_out[i][m][n][o];
                b = d_out[((i * F + m)*Ho + n) * Wo+ o];
                
                total += a * b;
            }
        }
    }
    // d_input_temp[((i * C+ j) * Hi + k) * Wi + l] = d_in[((i * C+ j) * Hi + k) * Wi + l] - learning_rate*total;
    d_input_temp[((i * C+ j) * Hi + Hi-1-k) * Wi + Wi-1-l] = total;
    // d_input_temp[i][j][Hi-1-k][Wi-1-l] = -learning_rate*total;
}

__global__
void backprop_cuda_kernel_bias(
    int stride, int padding, \
    double* d_out, double* d_kernel, double* d_in, \
    double* d_kernel_temp, double* d_bias, \
    int N, int F, int C, \
    int Hk, int Wk, int Hi, int Wi, int Ho, int Wo, \
    double learning_rate ) 
{
    // Block indices
    int i = blockIdx.x; // batch index
    int j = blockIdx.y; // output volume index
    int k = threadIdx.x; // vertical index in the output volume
    int l = threadIdx.y; // horizontal index in the output volume

    int im_si = stride * k - padding; // height -25~3
    int im_sj = stride * l - padding; // width
    double kernelGradient = 0.0;
    double biasGradient = 0.0;
    double a, b;
    
    for (int m = 0; m < N; ++m) { 
        for (int n = 0; n < Ho; ++n) {
            for (int o = 0; o < Wo; ++o) {
                int x = im_si + n; // -25~0
                int y = im_sj + o;
                if (x < 0 || x >= Hi || y < 0 || y >= Wi)
                    continue; 
                
                // a = d_in[m][0][x][y];
                a = d_in[((m * C)*Hi + x) * Wi + y];
                // a = d_kernel[m][0][Hk-1-x][Wk-1-y];
                // a = d_kernel[((m * C)*Hk + Hk-1-x) * Wk + Wk-1-y];

                // b = d_out[m][i][n][o];
                b = d_out[((m * F+i)*Ho + n) * Wo + o];
                // b = d_out[i][m][n][o];
                // b = d_out[((i * F + m)*Ho + n) * Wo+ o];
                
                kernelGradient += a * b;
                biasGradient += b;
            }
        }
    }

    // d_kernel_temp[i][j][k][l] = d_kernel[i][j][k][l] - learning_rate*total;
    d_kernel_temp[(((i * C)+j)*Hk + k) * Wk + l] = d_kernel[(((i * C)+j)*Hk + k) * Wk + l] - learning_rate*kernelGradient;
    if(k==0 && l==0) {
        d_bias[i] -= learning_rate * biasGradient;
    }
    // d_input_temp[((i * C+ j) * Hi + Hi-1-k) * Wi + Wi-1-l] = total;
    // d_input_temp[i][j][Hi-1-k][Wi-1-l] = -learning_rate*total;
}

void Conv2d::forward() {
    dim3 numBlocks(output_dims[0], output_dims[1]);
    dim3 threadsPerBlock(output_dims[2], output_dims[3]);
    Conv2d_gpu<<<numBlocks, threadsPerBlock>>>( \
        stride, padding, d_kernel, d_in, d_out, d_bias, \
        input_dims[0], kernels.dims[0], input_dims[1], \
        kernels.dims[2], kernels.dims[3], \
        input_dims[2], input_dims[3], \
        output_dims[2], output_dims[3]
    );
}

Tensor<double> &Conv2d::initOutputTensor() {
    int output_w = ((input_.dims[3] + 2 * padding - (kernels.dims[3] - 1) - 1) / stride) + 1;
    int output_h = ((input_.dims[2] + 2 * padding - (kernels.dims[2] - 1) - 1) / stride) + 1;
    int result_dims[] = {input_.dims[0], kernels.dims[0], output_h, output_w};
    Tensor<double> product(4, result_dims);
    product_ = product;
    return product_;
}

__global__ 
void Conv2d_input_gradient_gpu(
    double* d_out,   // Gradient of the output (N, F, Ho, Wo)  
    double* d_kernel,      // Kernel gradient to be computed (F, C, Hk, Wk)
    double* d_input_temp,  // Input tensor (N, C, Hi, Wi)
    int N, int F, int C,          // Batch size, number of filters, channels
    int Ho, int Wo,               // Output height and width
    int Hi, int Wi,               // Input height and width
    int Hk, int Wk,               // Kernel height and width
    int padding, int stride, 
    int learning_rate)      // Padding and stride
{
    int b = blockIdx.x;   // Batch index
    int d = blockIdx.y;   // Channel index of the input
    int i = threadIdx.x; // Input height index
    int j = threadIdx.y; // Input width index

    if (i >= Hi || j >= Wi) return; // Boundary check

    // Compute input gradient
    double input_grad_sum = 0.0;

    for (int f = 0; f < F; ++f) {
        for (int p = 0; p < Hk; ++p) {
            for (int q = 0; q < Wk; ++q) {
                int i_out = (i - p + padding) / stride;
                int j_out = (j - q + padding) / stride;

                if (i_out >= 0 && i_out < Ho && j_out >= 0 && j_out < Wo) {
                    double chain_grad = d_out[((b * F + f) * Ho + i_out) * Wo + j_out];
                    double kernel_value = d_kernel[((f * C + d) * Hk + p) * Wk + q];
                    input_grad_sum += chain_grad * kernel_value;
                }
            }
        }
    }

    d_input_temp[((b * C + d) * Hi + i) * Wi + j] = input_grad_sum;
}

extern __shared__ double shared_mem[];
__global__ void Conv2d_kernel_gradient_gpu(
    double* d_out,      // Gradient of the output (N, F, Ho, Wo)
    double* d_in,       // Input tensor (N, C, Hi, Wi)
    double* d_kernel,   // Kernel gradient to be computed (F, C, Hk, Wk)
    double* d_bias,
    double* d_kernel_temp,
    int N, int F, int C,          // Batch size, number of filters, channels
    int Ho, int Wo,               // Output height and width
    int Hi, int Wi,               // Input height and width
    int Hk, int Wk,               // Kernel height and width
    int padding, int stride, 
    int learning_rate,
    int kernel_size )      // Padding and stride
{
    int f = threadIdx.x / C;   // Filter index
    int c = threadIdx.x % C;   // Channel index
    int p = threadIdx.y;  // Kernel height index
    int q = threadIdx.z;  // Kernel width index

    if (p >= Hk || q >= Wk) return; // Boundary check

    // Shared memory for accumulating kernel gradient within a block
    shared_mem[((f * C + c) * Hk + p) * Wk + q] = 0.0;
    if( c == 0 && p == 0 && q == 0 ){
        shared_mem[kernel_size + f] = 0.0;
    }
    __syncthreads();

    // Compute kernel gradient
    for (int n = 0; n < N; ++n) {
        for (int ho = 0; ho < Ho; ++ho) {
            for (int wo = 0; wo < Wo; ++wo) {
                int i = ho * stride - padding + p;
                int j = wo * stride - padding + q;

                if (i >= 0 && i < Hi && j >= 0 && j < Wi) {
                    double input_value = d_in[((n * C + c) * Hi + i) * Wi + j];
                    double chain_grad_value = d_out[((n * F + f) * Ho + ho) * Wo + wo];
                    shared_mem[((f * C + c) * Hk + p) * Wk + q] += chain_grad_value * input_value;

                    if (c == 0 && p == 0 && q == 0) {
                        shared_mem[kernel_size + f] += chain_grad_value;
                    }
                }
            }
        }
    }

    // Synchronize threads within the block
    d_kernel_temp[((f * C + c) * Hk + p) * Wk + q] = d_kernel[((f * C + c) * Hk + p) * Wk + q] \
                             - learning_rate * shared_mem[((f * C + c) * Hk + p) * Wk + q];
    if(c == 0 && p == 0 && q == 0 ){
        d_bias[f] = shared_mem[kernel_size + f];
    }
}


double * Conv2d::backprop(double* d_chain_gradient, double learning_rate, bool test) {
    
    d_out = d_chain_gradient;
    double * d_input_temp, * d_kernel_temp;

    size_t kernel_size = kernels.dims[0] * kernels.dims[1] * kernels.dims[2] * kernels.dims[3] * sizeof(double);
    cudaMalloc((void **) &d_kernel_temp, kernel_size);
    cudaMalloc((void **) &d_input_temp, input_size * sizeof(double));
    
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
 

    dim3 numBlocks(input_dims[0], input_dims[1]);
    dim3 threadsPerBlock(input_dims[2], input_dims[3]);

    backprop_cuda_input<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(
        1, output_dims[2]-1, 
        this->d_out, this->d_kernel, d_input_temp, this->d_in,
        input_dims[0], kernels.dims[0], kernels.dims[1],
        kernels.dims[2], kernels.dims[3], \
        input_dims[2], input_dims[3], \
        output_dims[2], output_dims[3], \
        learning_rate
    );
    
    dim3 grid(kernels.dims[0], kernels.dims[1]);
    dim3 block(kernels.dims[2], kernels.dims[3]);

    backprop_cuda_kernel_bias<<<grid, block, 0, stream[1]>>>(
        1, 0, 
        this->d_out, this->d_kernel, this->d_in, \
        d_kernel_temp, this->d_bias, \
        input_dims[0], kernels.dims[0], kernels.dims[1],
        kernels.dims[2], kernels.dims[3], \
        input_dims[2], input_dims[3], \
        output_dims[2], output_dims[3], \
        learning_rate
    );

    if(test) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Conv2d::backprop::CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        else {
            std::cout << "Conv2d::backprop::CUDA success!" << std::endl;
        }
    }

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    cudaFree(d_in);
    cudaFree(d_kernel);

    d_in = d_input_temp;
    d_kernel = d_kernel_temp;

    if(test) {
        Tensor<double> kernels_gpu = this->kernels;
        kernels_gpu.zero();
        cudaMemcpy(kernels_gpu.getData(), d_kernel, kernels_gpu.getSize() * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "test kernels:" << (kernels_gpu==this->kernels) << std::endl;
    
        Tensor<double> bias_gpu = this->bias;
        bias_gpu.zero();
        cudaMemcpy(bias_gpu.getData(), d_bias, bias_gpu.getSize() * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "test bias:" << (bias_gpu==this->bias) << std::endl; 
    }

    return d_in;
}

Tensor<double> &Conv2d::forward(Tensor<double> &input) {
    input_ = input;
    
    product_ = input.convolve2d(kernels, stride, padding, bias);

    return product_;
}

// Tensor<double> &Conv2d::forward(Tensor<double> &input) {
//     input_ = input;
//     product_ = input.convolve2d(kernels, stride, padding, bias);

//     return product_;
// }

// TODO: add Conv2d::backpropCUDA() here, and kernel call for operation
Tensor<double> Conv2d::backprop(Tensor<double> chain_gradient, double learning_rate) {
    Tensor<double> kernels_gradient(kernels.num_dims, kernels.dims);
    Tensor<double> input_gradient(input_.num_dims, input_.dims);
    Tensor<double> bias_gradient(1, bias.dims);
    kernels_gradient.zero();
    input_gradient.zero();
    bias_gradient.zero();

    // backprop convolution -- not using Tensor.convolve2d for efficiency
    for (int i = 0; i < input_.dims[0]; ++i) { // for each batch img
        for (int f = 0; f < kernels.dims[0]; f++) { // for each filter
            
            int x = -padding;
            for (int cx = 0; cx < chain_gradient.dims[2]; x += stride, cx++) { // for each x in the chain gradient
                int y = -padding;
                for (int cy = 0; cy < chain_gradient.dims[3]; y += stride, cy++) { // for each y in the chain gradient
                    
                    double chain_grad = chain_gradient.get(i, f, cx, cy);
                    for (int fx = 0; fx < kernels.dims[2]; fx++) { // for each x in the filter
                        int ix = x + fx; // input x
                        if (ix >= 0 && ix < input_.dims[2]) {
                            for (int fy = 0; fy < kernels.dims[3]; fy++) { // for each y in the filter
                                int iy = y + fy; // input y
                                
                                if (iy >= 0 && iy < input_.dims[3]) {
                                    
                                    for (int fc = 0; fc < kernels.dims[1]; fc++) { // for each channel in the filter
                                        kernels_gradient.add(f, fc, fx, fy, input_.get(i, fc, ix, iy) * chain_grad);
                                        input_gradient.add(i, fc, ix, iy, kernels.get(f, fc, fx, fy) * chain_grad);

                                    }
                                }
                            }
                        }
                    }
                    bias_gradient.add(f, chain_grad);
                }
            }
        }
    }
    kernels -= kernels_gradient * learning_rate;
    bias -= bias_gradient * learning_rate;

    return input_gradient;
}

void Conv2d::load(FILE *file_model) {
    double value;
    for (int i = 0; i < kernels.dims[0]; ++i) {
        for (int j = 0; j < kernels.dims[1]; ++j) {
            for (int k = 0; k < kernels.dims[2]; ++k) {
                for (int l = 0; l < kernels.dims[3]; ++l) {
                    int read = fscanf(file_model, "%lf", &value); // NOLINT(cert-err34-c)
                    if (read != 1) throw std::runtime_error("Invalid model file");
                    kernels.set(i, j, k, l, value);
                }
            }
        }
    }
    for (int m = 0; m < bias.dims[0]; ++m) {
        int read = fscanf(file_model, "%lf", &value); // NOLINT(cert-err34-c)
        if (read != 1) throw std::runtime_error("Invalid model file");
        bias.set(m, value);
    }
}

void Conv2d::save(FILE *file_model) {
    for (int i = 0; i < kernels.dims[0]; ++i) {
        for (int j = 0; j < kernels.dims[1]; ++j) {
            for (int k = 0; k < kernels.dims[2]; ++k) {
                for (int l = 0; l < kernels.dims[3]; ++l) {
                    fprintf(file_model, "%.18lf ", kernels.get(i, j, k, l));
                }
            }
        }
    }
    for (int m = 0; m < bias.dims[0]; ++m) {
        fprintf(file_model, "%.18lf ", bias.get(m));
    }
}
