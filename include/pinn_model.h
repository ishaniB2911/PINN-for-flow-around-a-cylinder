#pragma once
#include <vector>
#include "dense_layer.h"

class PINNModel {
public:
    struct Derivatives {
        double u, v, p;
        double u_x, u_y, v_x, v_y, p_x, p_y;
        double u_xx, u_yy, v_xx, v_yy;
    };

    std::vector<DenseLayer> layers;
    double learningRate;

    PINNModel();

    // Forward pass: returns [u, v, p]
    std::vector<double> forward(double x, double y);

    // Derivatives via centered finite differences
    Derivatives computeDerivatives(double x, double y, double h = 1e-4);

    // Physics: Navier–Stokes residual loss
    double physicsLoss(double x, double y, double Re);

    // Boundary loss (Dirichlet)
    double boundaryLoss(double x, double y, double u_true, double v_true);

    // One training iteration over randomly sampled points
    double train(double Re, int nSteps = 5);

    // Naive finite-difference gradient update (last layer only)
    void updateWeights(double x, double y, double Re);
};

