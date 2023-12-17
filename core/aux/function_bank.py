import numpy as np
from core.aux.domains import Domain, HyperParalelipiped
from abc import ABC, abstractmethod
from typing import Union, Callable
from scipy.interpolate import interp1d

import random

class Function(ABC):
    @abstractmethod
    def evaluate(self, r, check_if_in_domain=True):
        pass

# 1D FUNCTIONS 

class Random(Function):
    def __init__(self, domain: Domain, seed: float, continuous: bool=False) -> None:
        super().__init__()
        self.seed = seed
        self.domain = domain
        self.continuous = continuous
        self.function = self._create_function()

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
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            return r[in_domain], self.function(r[in_domain])
        else:
            return r, self.function(r)

class Interpolation(Function):
    def __init__(self, values, raw_domain, domain: Domain) -> None:
        super().__init__()
        self.values = values
        self.raw_domain = raw_domain
        self.domain = domain

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
        super().__init__()
        self.domain = domain
        self.center = center
        self.width = width
        self.unimodularity_precision = unimodularity_precision
        self.spread = self.width / (5 * np.sqrt(2 * np.log(2)))
        self.normalization = self._compute_normalization()
        
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

    def evaluate(self, r, check_if_in_domain=True):

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