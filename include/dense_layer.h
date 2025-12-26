#pragma once
#include <string>
#include <vector>
#include "matrix.h"

class DenseLayer {
public:
    Matrix weights;
    std::vector<double> bias;
    std::string activation; // "tanh" or "linear"

    DenseLayer(int inputSize, int outputSize, std::string act = "tanh");
    std::vector<double> forward(const std::vector<double>& x);
};

