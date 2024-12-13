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

    this->stride = stride;
    this->padding = padding;
}

__global__
void Conv2d_gpu(int stride, int padding, double* kernel, double* d_in, double* d_out, double* bias,\
                int k_outchannel, int k_inchannel, int k_height, int k_width, \
                int batch_size, int depth, int height, int width, \
                int out_height, int out_width) {
    // Block indices
    int i = blockIdx.x; // batch index
    int j = blockIdx.y; // output volume index
    int k = threadIdx.x; // vertical index in the output volume
    int l = threadIdx.y; // horizontal index in the output volume

    if (k >= out_height || l >= out_width) return;

    int im_si = stride * k - padding; // height
    int im_sj = stride * l - padding; // width
    double total = 0;
    double a, b;
    for (int m = 0; m < k_inchannel; ++m) { // pra cada canal do filtro
        for (int n = 0; n < k_height; ++n) {
            for (int o = 0; o < k_width; ++o) {
                int x = im_si + n, y = im_sj + o;
                if (x < 0 || x >= height || y < 0 || y >= width)
                    continue; // se for regiao do padding, pula (soma 0)
                // double a = get(i, m, x, y);
                a = d_in[((i*batch_size + m)*depth + x) * height + y];
                // double b = kernels.get(j, m, n, o);
                b = kernel[((j*k_outchannel + m)*k_inchannel + n)*k_height + o];
                total += a * b;
            }
        }
    }
    d_out[((i * batch_size+ j)*k_outchannel + k)*out_height + l] = total;
}

// void setInputPointer(double *pt) {
//     d_in = pt;
// }
// TODO: add &Conv2d::forwardCUDA() here, and kernel call for convolve2dCUDA
// see: Tensor.cpp
// 暫時把input留著，等backprop做完再拿掉
double* Conv2d::forward(Tensor<double> &input, double * x) {
    input_ = input;
    d_in = x;

    // allocate memory for output
    int output_w = ((input.dims[3] + 2 * padding - (kernels.dims[3] - 1) - 1) / stride) + 1;
    int output_h = ((input.dims[2] + 2 * padding - (kernels.dims[2] - 1) - 1) / stride) + 1;
    int size = input.dims[0] * kernels.dims[0] * output_h * output_w;
    cudaMalloc((void **) &d_out, size);

    // allocate memory for kernel
    double* d_kernel;
    int kernel_size = kernels.dims[0] * kernels.dims[1] * kernels.dims[2] * kernels.dims[3];
    cudaMalloc((void **) &d_kernel, kernel_size);
    cudaMemcpy(d_kernel, kernels.getData(), kernel_size, cudaMemcpyHostToDevice);

    // allocate memory for bias
    double* d_bias;
    int bias_size = bias.dims[0];
    cudaMalloc((void **) &d_bias, bias_size);
    cudaMemcpy(d_bias, bias.getData(), bias_size, cudaMemcpyHostToDevice);

    dim3 numBlocks(input.dims[0], kernels.dims[0]);
    dim3 threadsPerBlock(output_h, output_w);
    Conv2d_gpu<<<numBlocks, threadsPerBlock>>>( \
                stride, padding, d_kernel, d_in, d_out, d_bias, 
                kernels.dims[0], kernels.dims[1], kernels.dims[2], kernels.dims[3], \
                input.dims[0], input.dims[1], input.dims[2], input.dims[3], \
                output_h, output_w
            );

    return d_out;
}


int Conv2d::getOutputSize() {
    int output_w = ((input_.dims[3] + 2 * padding - (kernels.dims[3] - 1) - 1) / stride) + 1;
    int output_h = ((input_.dims[2] + 2 * padding - (kernels.dims[2] - 1) - 1) / stride) + 1;
    int size = input_.dims[0] * kernels.dims[0] * output_h * output_w;
    return size;
}

Tensor<double> &Conv2d::initOutputTensor() {
    int output_w = ((input_.dims[3] + 2 * padding - (kernels.dims[3] - 1) - 1) / stride) + 1;
    int output_h = ((input_.dims[2] + 2 * padding - (kernels.dims[2] - 1) - 1) / stride) + 1;
    int result_dims[] = {input_.dims[0], kernels.dims[0], output_h, output_w};
    Tensor<double> product(4, result_dims);
    product_ = product;
    return product_;
}
// Tensor<double> &Conv2d::forward(Tensor<double> &input) {
//     input_ = input;
//     product_ = input.convolve2d(kernels, stride, padding, bias);

//     return product_;
// }

// TODO: add Conv2d::backpropCUDA() here, and kernel call for operation
__global__
void Conv2dBackProp_gpu() {

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
