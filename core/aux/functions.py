from core.aux.domains import Domain, HyperParalelipiped

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Callable
from scipy.interpolate import interp1d
import random
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt 


class Function(ABC):
    def __init__(self, domain: Domain) -> None:
        super().__init__()
        self.domain = domain

    @abstractmethod
    def evaluate(self, r, check_if_in_domain=True):
        pass

    def is_compatible_with(self, other):
        if isinstance(other, Function):
            if self.domain == other.domain:
                return True
            else:
                raise Exception('The two functions have different domains')
        else:
            raise Exception('The other function is not of type Function')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return _ScaledFunction(self, other)
        elif self.is_compatible_with(other):
            return _ProductFunction(self, other)

    def __add__(self, other):
        if self.is_compatible_with(other):
            return _SumFunction(self, other)
    
    def __sub__(self, other):
        if self.is_compatible_with(other):
            return _SubtractFunction(self, other)
        
    def __truediv__(self, other):
        if self.is_compatible_with(other):
            return _DivideFunction(self, other)

# 1D FUNCTIONS 
class _ScaledFunction(Function):
    def __init__(self, function: Function, scalar: float):
        super().__init__(function.domain)
        self.function = function
        self.scalar = scalar

    def evaluate(self, r, check_if_in_domain=True):
        eval_function = self.function.evaluate(r, check_if_in_domain)
        return eval_function[0], eval_function[1] * self.scalar

class _DivideFunction(Function):
    def __init__(self, function1: Function, function2: Function) -> None:
        super().__init__()
        self.function1 = function1
        self.function2 = function2
        self.domain = function1.domain

    def evaluate(self, r, check_if_in_domain=True):
        eval1 = self.function1.evaluate(r,check_if_in_domain)
        eval2 = self.function2.evaluate(r, check_if_in_domain)
        return eval1[0], eval1[1] / eval2[1]

class _SubtractFunction(Function):
    def __init__(self, function1: Function, function2: Function) -> None:
        super().__init__()
        self.function1 = function1
        self.function2 = function2
        self.domain = function1.domain

    def evaluate(self, r, check_if_in_domain=True):
        eval1 = self.function1.evaluate(r,check_if_in_domain)
        eval2 = self.function2.evaluate(r, check_if_in_domain)
        return eval1[0], eval1[1] - eval2[1]

class _SumFunction(Function):
    def __init__(self, function1: Function, function2: Function) -> None:
        super().__init__()
        self.function1 = function1
        self.function2 = function2
        self.domain = function1.domain

    def evaluate(self, r, check_if_in_domain=True):
        eval1 = self.function1.evaluate(r,check_if_in_domain)
        eval2 = self.function2.evaluate(r, check_if_in_domain)
        return eval1[0], eval1[1] + eval2[1]

class _ProductFunction(Function):
    def __init__(self, function1: Function, function2: Function) -> None:
        super().__init__()
        self.function1 = function1
        self.function2 = function2
        self.domain = function1.domain

    def evaluate(self, r, check_if_in_domain=True):
        eval1 = self.function1.evaluate(r,check_if_in_domain)
        eval2 = self.function2.evaluate(r, check_if_in_domain)
        return eval1[0], eval1[1] * eval2[1]

class Constant_1D(Function):
    """
    Compute the constant function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.

    Returns:
    - numpy.ndarray: Computed Haar wavelet function values over the domain.
    """
    def __init__(self, domain: HyperParalelipiped):
        super().__init__()
        self.domain = domain

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            return r[in_domain], np.ones_like(r[in_domain])
        else:
            return r, np.ones_like(r)

class Random_1D(Function):
    def __init__(self, domain: Domain, seed: float, continuous: bool=False) -> None:
        super().__init__()
        self.seed = seed
        self.domain = domain
        self.continuous = continuous
        self.function = self._create_function()

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def _create_function(self):
        if self.seed is None:
            self.seed = np.random.randint(0,10000)
        random.seed(self.seed)
        
        if self.domain.dimension == 1:
            # Random number of partitions
            if self.continuous is True:
                segments = 1
            else:
                segments = random.randint(1,10)
            values = np.zeros_like(self.domain.mesh)
            inpoints = [random.uniform(self.domain.bounds[0][0], self.domain.bounds[0][1]) for _ in range(segments-1)]
            allpoints = sorted(list(self.domain.bounds[0]) + inpoints)
            partitions = [(allpoints[i], allpoints[i+1]) for i in range(len(allpoints)-1)]

            for _, partition in enumerate(partitions):
                
                # Random position of the x-shift in the function
                x_shift = random.uniform(partition[0], partition[1])
                # Random coefficients
                a0 = random.uniform(-1,5)
                a1 = random.uniform(-1,5)
                a2 = random.uniform(-1,5)
                a3 = random.uniform(-1,5)
                stretch = np.max(self.domain.mesh)
                model = a0 + a1*(self.domain.mesh + x_shift)/stretch + a2*((self.domain.mesh + x_shift)/stretch)**2 + a3*((self.domain.mesh + x_shift)/stretch)**3
                model[self.domain.mesh<partition[0]] = 0
                model[self.domain.mesh>partition[1]] = 0

                values += model
            return interp1d(self.domain.mesh, values, 
                            kind='linear', fill_value='extrapolate')
    
    def evaluate(self, r, check_if_in_domain=True):
        if isinstance(r, (float, int)):
            r = np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            return r[in_domain], self.function(r[in_domain])
        else:
            return r, self.function(r)

class Interpolation_1D(Function):
    def __init__(self, values, raw_domain, domain: Domain) -> None:
        super().__init__()
        self.values = values
        self.raw_domain = raw_domain
        self.domain = domain

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            return r[in_domain], interp1d(self.raw_domain, self.values, 
                                          kind='linear', fill_value='extrapolate')(r[in_domain])
        else:
            return r, interp1d(self.raw_domain, self.values, 
                                kind='linear', fill_value='extrapolate')(r)

class ComplexExponential_1D(Function):
    """
    Compute the Complex exponential function over a given domain.

    Args:
        frequency (float): Frequency
        domain (HyperParalelipiped): (a,b) type domain.

    Returns:
        np.ndarray: Computed Complex exponential function values over the domain.
    """
    def __init__(self, domain: HyperParalelipiped, frequency):
        super().__init__()
        self.domain = domain
        self.frequency = frequency

    def evaluate(self, r, check_if_in_domain=True):
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            fourier_vector = np.exp(-2 * np.pi * self.frequency * 1j * r[in_domain] / self.domain.total_measure) / self.domain.total_measure
            return r[in_domain], fourier_vector
        else:
            fourier_vector = np.exp(-2 * np.pi * self.frequency * 1j * r / self.domain.total_measure) / self.domain.total_measure
            return r, fourier_vector

class Polynomial_1D(Function):
    def __init__(self, domain: HyperParalelipiped, order, min_val, max_val):
        super().__init__()
        self.order = order
        self.min_val = min_val
        self.max_val = max_val
        self.domain = domain
        self.coefficients = self.generate_random_coefficients()

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def generate_random_coefficients(self):
        if self.order < 0:
            raise ValueError("The order of the polynomial must be non-negative.")

        # Generate random coefficients within the specified range
        coefficients = np.random.uniform(self.min_val, self.max_val, self.order + 1)

        # Set some coefficients to zero to introduce more variation
        num_zero_coefficients = np.random.randint(1, self.order + 1)
        zero_indices = np.random.choice(range(self.order + 1), num_zero_coefficients, replace=False)
        coefficients[zero_indices] = 0

        return coefficients

    def evaluate(self, r, check_if_in_domain=True):
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            poly_function = np.poly1d(self.coefficients)
            return r[in_domain], poly_function(r[in_domain])
        else:
            poly_function = np.poly1d(self.coefficients)
            return r, poly_function(r)

class SinusoidalPolynomial_1D(Function):
    def __init__(self, domain: HyperParalelipiped, order, min_val, max_val, min_f, max_f, seed):
        super().__init__()
        self.order = order
        self.min_val = min_val
        self.max_val = max_val
        self.min_f = min_f
        self.max_f = max_f
        self.seed = seed
        self.domain = domain
        self.coefficients, self.frequencies, self.phases = self.generate_random_parameters()

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def generate_random_parameters(self):
        if self.order < 0:
            raise ValueError("The order of the polynomial must be non-negative.")

        np.random.seed(self.seed)

        # Generate random coefficients within the specified range
        coefficients = np.random.uniform(self.min_val, self.max_val, self.order + 1)

        # Generate random frequencies and phases for the sinusoidal functions
        frequencies = np.random.uniform(self.min_f, self.max_f, self.order + 1)
        phases = np.random.uniform(0, 2 * np.pi, self.order + 1)

        return coefficients, frequencies, phases

    def poly_with_sinusoidal(self, x):
            result = 0
            for i in range(self.order + 1):
                term = self.coefficients[i] * np.sin(self.frequencies[i] * x + self.phases[i])
                result += term
            return result

    def evaluate(self, r, check_if_in_domain=True):
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            return r[in_domain], self.poly_with_sinusoidal(r[in_domain])
        else:
            return r, self.poly_with_sinusoidal(r)

class SinusoidalGaussianPolynomial_1D(Function):
    def __init__(self, domain: HyperParalelipiped, order, min_val, max_val, min_f, max_f, spread, seed=None):
        super().__init__()
        self.order = order
        self.min_val = min_val
        self.max_val = max_val
        self.min_f = min_f
        self.max_f = max_f
        self.spread = spread
        self.seed = seed
        self.domain = domain
        self.coefficients, self.frequencies, self.phases = self.generate_random_parameters()
        self.mean, self.std_dev = self.generate_gaussian_parameters()

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def generate_random_parameters(self):
        if self.order < 0:
            raise ValueError("The order of the polynomial must be non-negative.")
        
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate random coefficients within the specified range
        coefficients = np.random.uniform(self.min_val, self.max_val, self.order + 1)

        # Generate random frequencies and phases for the sinusoidal functions
        frequencies = np.random.uniform(self.min_f, self.max_f, self.order + 1)
        phases = np.random.uniform(0, 2 * np.pi, self.order + 1)

        return coefficients, frequencies, phases

    def generate_gaussian_parameters(self):
        # Generate Gaussian function parameters
        mean = np.random.uniform(self.domain.bounds[0][0], self.domain.bounds[0][-1])
        std_dev = np.random.uniform(self.spread / 2, self.spread * 2)

        return mean, std_dev

    def evaluate(self, r, check_if_in_domain=True):
        def poly_with_sinusoidal(x):
            result = 0
            for i in range(self.order + 1):
                term = self.coefficients[i] * np.sin(self.frequencies[i] * x + self.phases[i])
                result += term
            return result

        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            gaussian_function = (1 / (self.std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r[in_domain] - self.mean) / self.std_dev) ** 2)
            return r[in_domain], poly_with_sinusoidal(r[in_domain]) * gaussian_function
        else:
            gaussian_function = (1 / (self.std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - self.mean) / self.std_dev) ** 2)
            return r, poly_with_sinusoidal(r) * gaussian_function

class NormalModes_1D(Function):
    def __init__(self, domain: HyperParalelipiped, order, spread, max_freq, seed=None):
        super().__init__()
        self.order = order
        self.seed = seed
        self.spread = spread
        self.max_freq = max_freq
        self.domain = domain
        self.coefficients, self.shifts = self.generate_random_parameters()
        self.mean, self.std_dev, self.frequency = self.generate_function_parameters()

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def generate_random_parameters(self):
        if self.order < 0:
            raise ValueError("The order of the polynomial must be non-negative.")
        
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate random coefficients within the specified range
        coefficients = np.random.uniform(-1, 1, self.order)

        # Generate random shifts
        shifts = np.random.uniform(0, 1, self.order)
        
        # Set some coefficients to zero to introduce more variation
        num_zero_coefficients = np.random.randint(1, self.order)  # Random number of coefficients to set to zero
        zero_indices = np.random.choice(range(self.order), num_zero_coefficients, replace=False)
        coefficients[zero_indices] = 0

        return coefficients, shifts

    def generate_function_parameters(self):
        # Generate sinusoidal part
        frequency = np.sqrt(np.random.uniform(0, self.max_freq, 1))

        # Generate Gaussian function parameters
        mean = np.random.uniform(self.domain.bounds[0][0], self.domain.bounds[0][-1])
        std_dev = np.random.uniform(self.spread / 2, self.spread * 2)

        return mean, std_dev, frequency

    def evaluate(self, r, check_if_in_domain=True):
        shifted_poly = np.zeros_like(r)
        for i in range(self.order):
            shifted_domain = r - self.shifts[i]
            shifted_poly += np.power(shifted_domain, i + 1)

        sin_poly = np.sin(r * self.frequency) * shifted_poly

        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            gaussian_function = (1 / (self.std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r[in_domain] - self.mean) / self.std_dev) ** 2)
            return r[in_domain], sin_poly[in_domain] * gaussian_function
        else:
            gaussian_function = (1 / (self.std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - self.mean) / self.std_dev) ** 2)
            return r, sin_poly * gaussian_function

class Gaussian_1D(Function):
    """
    Compute the Gaussian function over a given domain.

    Args:
    - center (float): Center of the Gaussian function.
    - width (float): Width of the Gaussian function.
    - domain (HyperParalelipiped): Array representing the domain for computation.

    Returns:
    - numpy.ndarray: Computed Gaussian function values over the domain.
    """
    def __init__(self, domain:HyperParalelipiped, center, width, unimodularity_precision=1000) -> None:
        super().__init__(domain=domain)
        self.center = center
        self.width = width
        self.unimodularity_precision = unimodularity_precision
        self.spread = self.width / (5 * np.sqrt(2 * np.log(2)))
        self.normalization = self._compute_normalization()
        
    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def _compute_normalization(self):
        precise_mesh = self.domain.dynamic_mesh(self.unimodularity_precision)
        gaussian_vector_full = (1 / (self.spread * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((precise_mesh - self.center) / self.spread) ** 2)
        return np.trapz(gaussian_vector_full, precise_mesh)

    def evaluate(self, r, check_if_in_domain=True):
        try: r[0] 
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            gaussian_vector = (1 / (self.spread * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r[in_domain] - self.center) / self.spread) ** 2)
            return r[in_domain], gaussian_vector / self.normalization
        else:
            gaussian_vector = (1 / (self.spread * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - self.center) / self.spread) ** 2)
            return r, gaussian_vector / self.normalization

class Moorlet_1D(Function):
    """
    Compute the Moorlet function over a given domain.

    Args:
    - center (float): Center of the Moorlet function.
    - spread (float): Spread of the Moorlet function.
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - frequency (float): Frequency parameter for the Moorlet function.

    Returns:
    - numpy.ndarray: Computed Moorlet function values over the domain.
    """
    def __init__(self, domain: HyperParalelipiped, center, spread, frequency, unimodularity_precision=1000):
        super().__init__()
        self.domain = domain
        self.center = center
        self.spread = spread
        self.frequency = frequency
        self.unimodularity_precision = unimodularity_precision
        self.normalization = self._compute_normalization()

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def _compute_normalization(self):
        moorlet_vector = np.cos(self.frequency * (self.domain.dynamic_mesh(self.unimodularity_precision) - self.center)) \
            * np.exp(-0.5 * ((self.domain.dynamic_mesh(self.unimodularity_precision) - self.center) / self.spread) ** 2)
    
        area = np.trapz(moorlet_vector, self.domain.dynamic_mesh(self.unimodularity_precision))
        return area

    def evaluate(self, r, check_if_in_domain=True):
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            moorlet_vector = np.cos(self.frequency * (r[in_domain] - self.center)) \
                * np.exp(-0.5 * ((r[in_domain] - self.center) / self.spread) ** 2)
            return r[in_domain], moorlet_vector / self.normalization
        else:
            moorlet_vector = np.cos(self.frequency * (r - self.center)) \
                * np.exp(-0.5 * ((r - self.center) / self.spread) ** 2)
            return r, moorlet_vector / self.normalization
    
class Haar_1D(Function):
    """
    Compute the Haar wavelet function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Haar wavelet function.
    - width (float): Width of the Haar wavelet function.

    Returns:
    - numpy.ndarray: Computed Haar wavelet function values over the domain.
    """
    def __init__(self, domain: HyperParalelipiped, center, width):
        super().__init__()
        self.domain = domain
        self.center = center
        self.width = width

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            scaled_domain = (r[in_domain] - self.center) / self.width
            haar_vector = 4 * np.where((scaled_domain >= -0.5) & (scaled_domain < 0.5), np.sign(scaled_domain), 0) / self.width**2
            return r[in_domain], haar_vector[in_domain]
        else:
            scaled_domain = (r - self.center) / self.width
            haar_vector = 4 * np.where((scaled_domain >= -0.5) & (scaled_domain < 0.5), np.sign(scaled_domain), 0) / self.width**2
            return r, haar_vector

class Ricker_1D(Function):
    """
    Compute the Ricker wavelet function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Ricker wavelet function.
    - width (float): Width of the Ricker wavelet function.

    Returns:
    - numpy.ndarray: Computed Ricker wavelet function values over the domain.
    """
    def __init__(self, domain: HyperParalelipiped, center, width):
        super().__init__()
        self.domain = domain
        self.center = center
        self.width = width

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        A = 2 / (np.sqrt(3 * self.width) * (np.pi ** 0.25))
        ricker_specific_width = self.width / 7

        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            vector = A * (1 - ((r[in_domain] - self.center) / ricker_specific_width) ** 2) * np.exp(
                -0.5 * ((r[in_domain] - self.center) / ricker_specific_width) ** 2)
            return r[in_domain], vector
        else:
            vector = A * (1 - ((r - self.center) / ricker_specific_width) ** 2) * np.exp(
                -0.5 * ((r - self.center) / ricker_specific_width) ** 2)
            return r, vector

class Dgaussian_1D(Function):
    """
    Compute the Polynomial wavelet function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Polynomial wavelet function.
    - width (float): Width of the Polynomial wavelet function.

    Returns:
    - numpy.ndarray: Computed Polynomial wavelet function values over the domain.
    """
    def __init__(self, domain: HyperParalelipiped, center, width):
        super().__init__()
        self.domain = domain
        self.center = center
        self.width = width
        self.spread = width / (5 * np.sqrt(2 * np.log(2)))

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            Dgaussian_vector = ((r[in_domain] - self.center) / (self.spread ** 3 * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((r[in_domain] - self.center) / self.spread) ** 2)
            return r[in_domain], Dgaussian_vector
        else:
            Dgaussian_vector = ((r - self.center) / (self.spread ** 3 * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((r - self.center) / self.spread) ** 2)
            return r, Dgaussian_vector

class Boxcar_1D(Function):
    """
    Compute the Boxcar function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Boxcar function.
    - width (float): Width of the Boxcar function.

    Returns:
    - numpy.ndarray: Computed Boxcar function values over the domain.
    """
    def __init__(self, domain: HyperParalelipiped, center, width, unimodularity_precision=1000):
        super().__init__()
        self.domain = domain
        self.center = center
        self.width = width
        self.unimodularity_precision = unimodularity_precision

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        if isinstance(r, (int, float)):
            r = np.array([r])  # Convert single value to a numpy array

        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            scaled_domain = (r[in_domain] - self.center) / self.width
            boxcar_vector = np.where(np.abs(scaled_domain) < 0.5, 1 / self.width, 0)
            return r[in_domain], boxcar_vector 
        else:
            scaled_domain = (r - self.center) / self.width
            boxcar_vector = np.where(np.abs(scaled_domain) < 0.5, 1 / self.width, 0)
            return r, boxcar_vector

class Bump_1D(Function):
    """
    Compute the Bump function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Bump function.
    - width (float): Width of the Bump function.

    Returns:
    - numpy.ndarray: Computed Bump function values over the domain.
    """
    def __init__(self, domain: HyperParalelipiped, center, width, unimodularity_precision=1000):
        super().__init__()
        self.domain = domain
        self.center = center
        self.width = width
        self.unimodularity_precision = unimodularity_precision
        self.normalization = self._compute_normalization()

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def _compute_normalization(self):
        limits = [-0.5 * self.width + self.center, 0.5 * self.width + self.center]
        mask = (self.domain.dynamic_mesh(self.unimodularity_precision) >= limits[0]) & (
                    self.domain.dynamic_mesh(self.unimodularity_precision) <= limits[1])
        bump_vector = np.zeros_like(self.domain.dynamic_mesh(self.unimodularity_precision))
        bump_vector[mask] = np.exp(
            1 / ((2 * (self.domain.dynamic_mesh(self.unimodularity_precision)[mask] - self.center) / self.width) ** 2 - 1))
        area = np.trapz(bump_vector[mask],
                        self.domain.dynamic_mesh(self.unimodularity_precision)[mask])
        return area

    def evaluate(self, r, check_if_in_domain=True):
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            limits = [-0.5 * self.width + self.center, 0.5 * self.width + self.center]
            mask = (r[in_domain] >= limits[0]) & (r[in_domain] <= limits[1])
            bump_vector = np.zeros_like(r[in_domain])
            bump_vector[mask] = np.exp(
                1 / ((2 * (r[in_domain][mask] - self.center) / self.width) ** 2 - 1))
            return r[in_domain], bump_vector / self.normalization
        else:
            limits = [-0.5 * self.width + self.center, 0.5 * self.width + self.center]
            mask = (r >= limits[0]) & (r <= limits[1])
            bump_vector = np.zeros_like(r)
            bump_vector[mask] = np.exp(
                1 / ((2 * (r[mask] - self.center) / self.width) ** 2 - 1))
            return r, bump_vector / self.normalization

class Dbump_1D(Function):
    """
    Compute the Bump derivative function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Bump function.
    - width (float): Width of the Bump function.

    Returns:
    - numpy.ndarray: Computed Bump derivative function values over the domain.
    """
    def __init__(self, domain: HyperParalelipiped, center, width, unimodularity_precision=1000):
        super().__init__()
        self.domain = domain
        self.center = center
        self.width = width
        self.unimodularity_precision = unimodularity_precision
        self.area = self._compute_area()

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def _compute_area(self):
        limits = [-0.5 * self.width + self.center, 0.5 * self.width + self.center]
        mask = (self.domain.dynamic_mesh(self.unimodularity_precision) >= limits[0]) & (
                    self.domain.dynamic_mesh(self.unimodularity_precision) <= limits[1])
        bump_vector = np.zeros_like(self.domain.dynamic_mesh(self.unimodularity_precision))
        bump_vector[mask] = np.exp(
            1 / ((2 * (self.domain.dynamic_mesh(self.unimodularity_precision)[mask] - self.center) / self.width) ** 2 - 1))
        area = np.trapz(bump_vector[mask], self.domain.dynamic_mesh(self.unimodularity_precision)[mask])
        return area

    def evaluate(self, r, check_if_in_domain=True):
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            limits = [-0.5 * self.width + self.center, 0.5 * self.width + self.center]
            mask = (r[in_domain] >= limits[0]) & (r[in_domain] <= limits[1])
            bump_vector = np.zeros_like(r[in_domain])
            bump_vector[mask] = np.exp(
                1 / ((2 * (r[in_domain][mask] - self.center) / self.width) ** 2 - 1))
            bump_vector[mask] *= 8 * (self.width**2) * (r[in_domain][mask] - self.center) / (
                    (2 * (r[in_domain][mask] - self.center))**2 - self.width**2)**2
            return r[in_domain], bump_vector / self.area
        else:
            limits = [-0.5 * self.width + self.center, 0.5 * self.width + self.center]
            mask = (r >= limits[0]) & (r <= limits[1])
            bump_vector = np.zeros_like(r)
            bump_vector[mask] = np.exp(
                1 / ((2 * (r[mask] - self.center) / self.width) ** 2 - 1))
            bump_vector[mask] *= 8 * (self.width**2) * (r[mask] - self.center) / (
                    (2 * (r[mask] - self.center))**2 - self.width**2)**2
            return r, bump_vector / self.area
    
class Triangular_1D(Function):
    """
    Compute the Triangular function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Triangular function.
    - width (float): Width of the Triangular function.

    Returns:
    - numpy.ndarray: Computed Triangular function values over the domain.
    """
    def __init__(self, domain: HyperParalelipiped, center, width):
        super().__init__()
        self.domain = domain
        self.center = center
        self.width = width

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        limits = [-0.5 * self.width + self.center, 0.5 * self.width + self.center]
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            mask = (r[in_domain] >= limits[0]) & (r[in_domain] <= limits[1])
            triangular_vector = np.zeros_like(r[in_domain])
            triangular_vector[mask] = 2 / self.width - 4 * np.abs(r[in_domain][mask] - self.center) / self.width**2
            return r[in_domain], triangular_vector
        else:
            mask = (r >= limits[0]) & (r <= limits[1])
            triangular_vector = np.zeros_like(r)
            triangular_vector[mask] = 2 / self.width - 4 * np.abs(r[mask] - self.center) / self.width**2
            return r, triangular_vector