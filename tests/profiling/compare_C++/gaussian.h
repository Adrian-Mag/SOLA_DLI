#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <Eigen/Dense>

class Gaussian {
private:
    double mean;
    double stddev;

public:
    // Constructor
    Gaussian(double mean, double stddev);

    // Evaluate the Gaussian function at an array of values
    Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const;

    // Getter for mean
    double getMean() const;

    // Getter for standard deviation
    double getStdDev() const;
};

#endif // GAUSSIAN_H
