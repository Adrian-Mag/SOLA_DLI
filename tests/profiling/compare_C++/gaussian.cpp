// gaussian.cpp

#include "gaussian.h"
#include <Eigen/Dense>
#include <cmath>

Gaussian::Gaussian(double mean, double stddev) : mean(mean), stddev(stddev) {}

Eigen::VectorXd Gaussian::evaluate(const Eigen::VectorXd& x) const {
    Eigen::VectorXd exponent = -0.5 * (x.array() - mean).array().square() / (stddev * stddev);
    return (exponent.array().exp() / (stddev * std::sqrt(2.0 * M_PI))).matrix();
}

double Gaussian::getMean() const {
    return mean;
}

double Gaussian::getStdDev() const {
    return stddev;
}
