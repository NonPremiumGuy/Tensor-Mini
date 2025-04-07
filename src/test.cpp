#include "tensor.hpp"
#include "common.hpp"
#include "one_hot_encode.hpp"
#include "layer.hpp"
#include "conv_layer.hpp"
#include "pool_layer.hpp"
#include "dense.hpp"
#include "network.hpp"
//#include "T.hpp"

#include <iostream>
#include <memory>

int main() {
    //std::vector<int> a_dimension = { 2, 2 };
    //std::unique_ptr<double[]> a_data = std::make_unique<double[]>(4);
    //double v = 0;
    //for (int i = 0; i < 4; i++) {
    //    a_data[i] = v;
    //    v++;
    //}
    //tensor<double> a(a_dimension, std::move(a_data));
    //std::cout << "element: " << a.size() << '\n';

    //tensor<double> cpy_cns = a + a;


    //std::vector<std::shared_ptr<layer<double>>> layers;

    //layers.push_back(std::make_shared<conv2d<double>>(1, 32, std::vector<int>{28, 28}, std::vector<int>{3, 3}, activation_type::ReLU));
    //layers.push_back(std::make_shared<pool<double>>(std::vector<int>{2, 2}, pooling_type::AVG));
    //layers.push_back(std::make_shared<conv2d<double>>(activation_type::ReLU));
    //layers.push_back(std::make_shared<pool<double>>(std::vector<int>{ 2, 2 }, pooling_type::MAX));
    //layers.push_back(std::make_shared<dense<double>>(10, activation_type::SoftMax));

    //network<double> cnn(layers);
    //cnn.train(int(10));

    //std::string test_data = "/data/test.csv";
    //cnn.output(test_data);


// Create a 4x4 input tensor and fill it with values 1 to 16
    //tensor<double> in(std::vector<int>{1, 1, 4, 4 });
    //in[0] = 1;
    //in[1] = 2;
    //in[2] = 2;
    //in[3] = 1;
    //in[4] = 4;
    //in[5] = 1;
    //in[6] = 1;
    //in[7] = 2;
    //in[8] = 1;
    //in[9] = 2;
    //in[10] = 2;
    //in[11] = 1;
    //in[12] = 2;
    //in[13] = 1;
    //in[14] = 3;
    //in[15] = 3;
    ////for (int i = 0; i < 16; ++i) {
    ////    in[i] = static_cast<double>(i + 1);
    ////}

    //for (int i = 0; i < 4; ++i) {
    //    for (int j = 0; j < 4; ++j) {
    //        std::cout << in[i * 4 + j] << "\t";
    //    }
    //    std::cout << std::endl;
    //}
    //std::vector<int> stride = in.stride();
    //std::cout << "stride: ";
    //for (int i = 0; i < stride.size(); i++) {
    //    std::cout << stride[i] << ' ';
    //}
    //std::vector<int> kernel = { 2, 2 };
    //std::vector<int> stride_shape = { 1, 1 };
    //std::vector<int> axes_to_pool = { 2, 3 };


    //pool<double> pooling(in, kernel, stride_shape, axes_to_pool, pooling_type::AVG, padding_type::VALID);
    //tensor<double> out = pooling.forward(in);

    //std::vector<int> out_shape = pooling.get_output_shape();
    //std::cout << "\nOutput Tensor (" << out_shape[0] << "x" << out_shape[1] << "x" << out_shape[2] << "x" << out_shape[3] << "):\n";
    //for (int i = 0; i < out.size(); ++i) {
    //    std::cout << out[i] << "\t";
    //    if ((i + 1) % out_shape[1] == 0) std::cout << "\n";
    //}

    //tensor<double> g(out.dims());
    //
    //g[0] = 4;
    //g[1] = 8;
    //g[2] = 0;
    //g[3] = 8;
    //g[4] = 4;
    //g[5] = 0;
    //g[6] = 0;
    //g[7] = 0;
    //g[8] = 0;

    //pooling.backward(g);
    //conv2d<double> c(1, 32, std::vector<int>{4, 4}, std::vector<int>{2, 2}, std::vector<int>{2, 2}, padding_type::FULL, activation_type::ReLU);
    //tensor<double> out = in.slice(1, 0, 1);
    //std::cout << out.size() << '\n';
    //for (int d : out.dims()) std::cout << d << ' ';
    //std::cout << '\n';
    //for (int i = 0; i < out.size(); ++i) {
    //    std::cout << out[i] << "\t";
    //}
    //std::vector<int> shape = { 4, 4 };
    //tensor<float> t(shape);

    //// Fill it
    //for (int i = 0; i < t.size(); i++) t[i] = i + 1;

    //auto s = t.slice(0, 1, 3);  // Get rows 1 and 2
    //std::cout << "Sliced shape: ";
    //for (int d : s.dims()) std::cout << d << ' ';
    //std::cout << '\n';

    //for (int i = 0; i < s.size(); i++) {
    //    std::cout << s[i] << ' ';
    //}
    //Tensor<double, > a;
    using T = float;

    // Create input tensor: 1 channel, 3x3
    std::vector<int> input_shape = { 3, 3 }; // H, W
    std::vector<int> kernel_shape = { 2, 2 }; // K_H, K_W
    std::vector<int> stride_shape = { 1, 1 };

    conv2d<T> conv(
        1,                  // input channel
        1,                  // output channel
        input_shape,
        kernel_shape,
        stride_shape,
        padding_type::VALID,
        activation_type::Leaky_ReLU
    );

    tensor<T> input(std::vector<int>{ 1, 3, 3 }); // C, H, W

    std::cout << "Input shape: ";
    for (auto i : input.dims()) std::cout << i << " ";
    std::cout << "\n";

    for (int i = 0; i < input.size(); ++i) {
        input[i] = static_cast<T>(i + 1); 
    }

    tensor<T>& kernel = conv._kernel;
    std::cout << "Kernel shape: ";
    for (auto i : kernel.dims()) std::cout << i << " ";
    std::cout << "\n";
    std::vector<int> kernel_index = { 0, 0, 0, 0 };
    kernel[{0, 0, 0, 0}] = 1.0f;
    kernel[{0, 0, 0, 1}] = 0.0f;
    kernel[{0, 0, 1, 0}] = 0.0f;
    kernel[{0, 0, 1, 1}] = -1.0f;

    // Run forward
    tensor<T> output = conv.forward(input);
    std::cout << "right here";

    // Print output shape and values
    std::vector<int> out_shape = output.dims();
    std::cout << "Output shape: [";
    for (int s : out_shape) std::cout << s << " ";
    std::cout << "]\n";

    std::cout << "Output values:\n";
    for (int i = 0; i < output.size(); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << "\n";

    return 0;
}

