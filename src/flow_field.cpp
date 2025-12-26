#include "flow_field.h"
#include "pinn_model.h"
#include <fstream>
#include <iostream>
#include <cmath>

void generateFlowField(PINNModel& model, const std::string& filename) {
    const int nx = 60, ny = 30;
    std::ofstream file(filename);
    file << "x,y,u,v,p,speed" << std::endl;

    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            double x = -1.0 + (j / static_cast<double>(nx - 1)) * 4.0;
            double y = -1.0 + (i / static_cast<double>(ny - 1)) * 2.0;

            // Skip cylinder interior (r < 0.5)
            if (x * x + y * y < 0.25) {
                file << x << "," << y << ",0,0,0,0" << std::endl;
                continue;
            }

            auto result = model.forward(x, y);
            double u = result[0];
            double v = result[1];
            double p = result[2];
            double speed = std::sqrt(u * u + v * v);

            file << x << "," << y << "," << u << "," << v << "," << p << "," << speed << std::endl;
        }
    }

    file.close();
    std::cout << "Flow field saved to " << filename << std::endl;
}
