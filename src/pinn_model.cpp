#include "pinn_model.h"
#include <random>
#include <cmath>
#include <algorithm>

PINNModel::PINNModel() : learningRate(0.001) {
    // Network architecture: 2 -> 32 -> 32 -> 32 -> 3
    layers.emplace_back(2, 32, "tanh");
    layers.emplace_back(32, 32, "tanh");
    layers.emplace_back(32, 32, "tanh");
    layers.emplace_back(32, 3, "linear"); // outputs: u, v, p
}

std::vector<double> PINNModel::forward(double x, double y) {
    std::vector<double> output = {x, y};
    for (auto& layer : layers) {
        output = layer.forward(output);
    }
    return output; // [u, v, p]
}

PINNModel::Derivatives PINNModel::computeDerivatives(double x, double y, double h) {
    Derivatives d{};
    auto result = forward(x, y);
    d.u = result[0];
    d.v = result[1];
    d.p = result[2];

    // First derivatives
    auto result_xp = forward(x + h, y);
    auto result_xm = forward(x - h, y);
    d.u_x = (result_xp[0] - result_xm[0]) / (2 * h);
    d.v_x = (result_xp[1] - result_xm[1]) / (2 * h);
    d.p_x = (result_xp[2] - result_xm[2]) / (2 * h);

    auto result_yp = forward(x, y + h);
    auto result_ym = forward(x, y - h);
    d.u_y = (result_yp[0] - result_ym[0]) / (2 * h);
    d.v_y = (result_yp[1] - result_ym[1]) / (2 * h);
    d.p_y = (result_yp[2] - result_ym[2]) / (2 * h);

    // Second derivatives
    d.u_xx = (result_xp[0] - 2 * d.u + result_xm[0]) / (h * h);
    d.u_yy = (result_yp[0] - 2 * d.u + result_ym[0]) / (h * h);
    d.v_xx = (result_xp[1] - 2 * d.v + result_xm[1]) / (h * h);
    d.v_yy = (result_yp[1] - 2 * d.v + result_ym[1]) / (h * h);

    return d;
}

double PINNModel::physicsLoss(double x, double y, double Re) {
    auto d = computeDerivatives(x, y);
    double nu = 1.0 / Re;

    // Continuity: du/dx + dv/dy = 0
    double continuity = d.u_x + d.v_y;

    // Momentum (x): u du/dx + v du/dy = -dp/dx + nu * (d2u/dx2 + d2u/dy2)
    double momentum_x = d.u * d.u_x + d.v * d.u_y + d.p_x - nu * (d.u_xx + d.u_yy);

    // Momentum (y): u dv/dx + v dv/dy = -dp/dy + nu * (d2v/dx2 + d2v/dy2)
    double momentum_y = d.u * d.v_x + d.v * d.v_y + d.p_y - nu * (d.v_xx + d.v_yy);

    return continuity * continuity + momentum_x * momentum_x + momentum_y * momentum_y;
}

double PINNModel::boundaryLoss(double x, double y, double u_true, double v_true) {
    auto result = forward(x, y);
    double u = result[0];
    double v = result[1];
    return (u - u_true) * (u - u_true) + (v - v_true) * (v - v_true);
}

double PINNModel::train(double Re, int nSteps) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(-1.0, 3.0);
    std::uniform_real_distribution<> dis_y(-1.0, 1.0);

    double totalLoss = 0.0;

    for (int step = 0; step < nSteps; ++step) {
        double x = dis_x(gen);
        double y = dis_y(gen);

        // Skip inside cylinder (r < 0.5)
        if (x * x + y * y < 0.25) continue;

        double pLoss = physicsLoss(x, y, Re);

        // Boundary conditions
        double bLoss = 0.0;

        // Inlet (x ≈ -1): u = 1, v = 0
        if (std::abs(x + 1) < 0.1) {
            bLoss += boundaryLoss(x, y, 1.0, 0.0);
        }

        // Cylinder surface (r ≈ 0.5): u = 0, v = 0
        double r = std::sqrt(x * x + y * y);
        if (std::abs(r - 0.5) < 0.1) {
            bLoss += boundaryLoss(x, y, 0.0, 0.0) * 10.0;
        }

        // Top/bottom walls (|y| ≈ 1): v = 0
        if (std::abs(std::abs(y) - 1.0) < 0.1) {
            auto result = forward(x, y);
            bLoss += result[1] * result[1];
        }

        double loss = pLoss + bLoss * 5.0;
        totalLoss += loss;

        // Very simple gradient update (last layer only, finite-diff approx)
        updateWeights(x, y, Re);
    }

    return totalLoss / nSteps;
}

void PINNModel::updateWeights(double x, double y, double Re) {
    const double h = 1e-4;
    auto& lastLayer = layers.back();

    // Update only a subset of weights for efficiency
    for (int i = 0; i < 3; ++i) { // 3 outputs (u, v, p)
        for (int j = 0; j < std::min(5, (int)lastLayer.weights.cols); ++j) {
            double original = lastLayer.weights.data[i][j];

            lastLayer.weights.data[i][j] = original + h;
            double lossPlus = physicsLoss(x, y, Re);

            lastLayer.weights.data[i][j] = original - h;
            double lossMinus = physicsLoss(x, y, Re);

            lastLayer.weights.data[i][j] = original;
            double grad = (lossPlus - lossMinus) / (2 * h);

            lastLayer.weights.data[i][j] -= learningRate * grad * 0.1;
        }
    }
}
