#include <iostream>
#include <iomanip>
#include <utility>
#include <vector>

#include "pinn_model.h"
#include "flow_field.h"

int main() {
    std::cout << "Physics-Informed Neural Network for Flow Around Cylinder\n";
    std::cout << "=========================================================\n";

    double Re = 100.0;
    int maxEpochs = 1000;

    std::cout << "Reynolds Number: " << Re << std::endl;
    std::cout << "Training epochs: " << maxEpochs << std::endl << std::endl;

    PINNModel model;

    std::cout << "Training..." << std::endl;
    for (int epoch = 0; epoch < maxEpochs; ++epoch) {
        double loss = model.train(Re, 10);

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << std::setw(4) << epoch
                      << "  Loss: " << std::scientific << std::setprecision(4) << loss << std::endl;

            // Save flow field at intervals
            if (epoch % 200 == 0) {
                std::string filename = "flow_field_epoch_" + std::to_string(epoch) + ".csv";
                generateFlowField(model, filename);
            }
        }
    }

    std::cout << "\nTraining complete!" << std::endl;

    // Generate final flow field
    generateFlowField(model, "flow_field_final.csv");

    // Test at specific points
    std::cout << "\nFlow field at selected points:\n";
    std::cout << "x\ty\tu\tv\tp\n";

    std::vector<std::pair<double, double>> testPoints = {
        {-0.5, 0.0}, // Before cylinder
        {0.5,  0.0}, // After cylinder
        {0.0,  0.6}, // Above cylinder
        {1.0,  0.0}, // Wake region
        {2.0,  0.0}  // Far downstream
    };

    for (const auto& point : testPoints) {
        auto result = model.forward(point.first, point.second);
        std::cout << std::fixed << std::setprecision(3)
                  << point.first << "\t" << point.second << "\t"
                  << result[0]  << "\t" << result[1]    << "\t" << result[2] << std::endl;
    }

    std::cout << "\nVisualization tip: Use Python/MATLAB to plot the CSV files\n";
    std::cout << "Example (Python): import pandas as pd; df = pd.read_csv('flow_field_final.csv')\n";

    return 0;
}
