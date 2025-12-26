#pragma once
#include <vector>

class Matrix {
    public: 
        std::vector<std::vector<double>> data; 
        int rows, cols; 

        Matrix(int r, int c); 

        // Initialise with uniform random values in [-scale, scale]
        void randomInit(double scale = 0.5);

        // y = W *x
        std::vector<double> multiply(const std::vector<double>& x) const; 


};