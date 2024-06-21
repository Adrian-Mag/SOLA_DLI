// main.cpp

#include <iostream>
#include "gaussian.h"
#include <Eigen/Dense>
#include <chrono>


int main() {
    Gaussian gaussian(0.0, 1.0);
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(10000000, -5.0, 5.0);

    auto start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd result2 = gaussian.evaluate(x);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    /* for(int i = 0; i < result2.size(); i++) {
        std::cout << "Gaussian function value at x = " << x(i) << ": " << result2(i) << std::endl;
    } */
    return 0;
}
