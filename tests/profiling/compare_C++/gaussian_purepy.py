import numpy as np
import time

# Function to evaluate the Gaussian at a given point
def evaluate_gaussian(x, mean, stddev):
    coeff = 1.0 / (stddev * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((x - mean) / stddev) ** 2
    return coeff * np.exp(exponent)

# Function to evaluate the Gaussian over an array of points
def evaluate_gaussian_array(xs, mean, stddev):
    coeff = 1.0 / (stddev * np.sqrt(2.0 * np.pi))
    exponents = -0.5 * ((xs - mean) / stddev) ** 2
    return coeff * np.exp(exponents)

# Parameters
mean = 0.0
stddev = 1.0
N = 100000000  # Number of points
xmin = -5.0
xmax = 5.0

# Generate points
xs = np.linspace(xmin, xmax, N)

# Evaluate Gaussian over the array of points and measure time
start_time = time.time()
result = evaluate_gaussian_array(xs, mean, stddev)
end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")
