// gaussian.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;

// Function to evaluate the Gaussian at a given point
double evaluate_gaussian(double x, double mean, double stddev) {
    double coeff = 1.0 / (stddev * std::sqrt(2.0 * M_PI));
    double exponent = -0.5 * std::pow((x - mean) / stddev, 2.0);
    return coeff * std::exp(exponent);
}

// Function to evaluate the Gaussian over an array of points
py::array_t<double> evaluate_gaussian_array(py::array_t<double> xs, double mean, double stddev) {
    auto xs_unchecked = xs.unchecked<1>(); // access data without bounds checking
    ssize_t n = xs_unchecked.shape(0);
    py::array_t<double> result(n);
    auto result_unchecked = result.mutable_unchecked<1>();

    for (ssize_t i = 0; i < n; ++i) {
        result_unchecked(i) = evaluate_gaussian(xs_unchecked(i), mean, stddev);
    }

    return result;
}

PYBIND11_MODULE(gaussian, m) {
    m.def("evaluate_gaussian", &evaluate_gaussian, "Evaluate Gaussian at a point",
          py::arg("x"), py::arg("mean"), py::arg("stddev"));
    m.def("evaluate_gaussian_array", &evaluate_gaussian_array, "Evaluate Gaussian over an array of points",
          py::arg("xs"), py::arg("mean"), py::arg("stddev"));
}
