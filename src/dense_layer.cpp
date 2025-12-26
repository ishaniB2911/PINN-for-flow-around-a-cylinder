#include "dense_layer.h"
#include <random>
#include <cmath>

DenseLayer::DenseLayer(int inputSize, int outputSize, std::string act)
    : weights(outputSize, inputSize), activation(act) {
    bias.resize(outputSize);
    weights.randomInit(0.5);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);
    for (int i = 0; i < outputSize; ++i) {
        bias[i] = dis(gen);
    }
}

std::vector<double> DenseLayer::forward(const std::vector<double>& x) {
    std::vector<double> z = weights.multiply(x);
    for (size_t i = 0; i < z.size(); ++i) {
        z[i] += bias[i];
    }
    if (activation == "tanh") {
        for (size_t i = 0; i < z.size(); ++i) {
            z[i] = std::tanh(z[i]);
        }
    }
    // "linear" -> no-op
    return z;
}

