import numpy as np
from core.aux.domains import Domain, HyperParalelipiped

# 1D FUNCTIONS 

def complex_exponential_1D(r, domain: HyperParalelipiped, check_if_in_domain, frequency):
    """Compute the Complex explonential function over a given domain

    Args:
        frequency (float): Frequency
        domain (HyperParalelipiped): (a,b) type domain

    Returns:
        np.ndarray: Computed Complex exponential function values over the domain.
    """    
    if check_if_in_domain:
        in_domain = domain.check_if_in_domain(r)
        fourier_vector = np.exp(-2*np.pi*frequency*1j*r[in_domain]/domain.total_measure)/domain.total_measure
        return r[in_domain], fourier_vector
    else: 
        fourier_vector = np.exp(-2*np.pi*frequency*1j*r/domain.total_measure)/domain.total_measure
        return r, fourier_vector

def gaussian_1D(r, domain:HyperParalelipiped, check_if_in_domain, center, width, unimodularity_precision=1000):
    """
    Compute the Gaussian function over a given domain.

    Args:
    - center (float): Center of the Gaussian function.
    - width (float): Width of the Gaussian function.
    - domain (HyperParalelipiped): Array representing the domain for computation.

    Returns:
    - numpy.ndarray: Computed Gaussian function values over the domain.
    """
    spread = width / (5 * np.sqrt(2 * np.log(2)))
    precise_mesh = domain.dynamic_mesh(unimodularity_precision)
    gaussian_vector_full = (1 / (spread * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((precise_mesh - center) / spread) ** 2)
    area = np.trapz(gaussian_vector_full, precise_mesh)
    if check_if_in_domain:
        in_domain = domain.check_if_in_domain(r)
        gaussian_vector = (1 / (spread * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r[in_domain] - center) / spread) ** 2)
        return r[in_domain], gaussian_vector / area
    else:
        gaussian_vector = (1 / (spread * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - center) / spread) ** 2)
        return r, gaussian_vector / area

def moorlet_1D(r, domain:HyperParalelipiped, check_if_in_domain, center, spread, frequency, unimodularity_precision=1000):
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
    moorlet_vector = np.cos(frequency * (domain.dynamic_mesh(unimodularity_precision) - center)) * np.exp(-0.5 * ((domain.dynamic_mesh(unimodularity_precision) - center) / spread) ** 2)
    area = np.trapz(moorlet_vector, domain.dynamic_mesh(unimodularity_precision))
    if check_if_in_domain:
        in_domain = domain.check_if_in_domain(r)
        moorlet_vector = np.cos(frequency * (r[in_domain] - center)) * np.exp(-0.5 * ((r[in_domain] - center) / spread) ** 2)
        return r[in_domain], moorlet_vector / area
    else:
        moorlet_vector = np.cos(frequency * (r - center)) * np.exp(-0.5 * ((r - center) / spread) ** 2)
        return r, moorlet_vector / area
    
    return moorlet_vector / area

def haar_1D(r, domain:HyperParalelipiped, check_if_in_domain, center, width):
    """
    Compute the Haar wavelet function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Haar wavelet function.
    - width (float): Width of the Haar wavelet function.

    Returns:
    - numpy.ndarray: Computed Haar wavelet function values over the domain.
    """
    if check_if_in_domain:
        in_domain = domain.check_if_in_domain(r)
        scaled_domain = (r[in_domain] - center) / width
        haar_vector = 4 * np.where((scaled_domain >= -0.5) & (scaled_domain < 0.5), np.sign(scaled_domain), 0) / width**2
        return r[in_domain], haar_vector
    else:
        scaled_domain = (r - center) / width
        haar_vector = 4 * np.where((scaled_domain >= -0.5) & (scaled_domain < 0.5), np.sign(scaled_domain), 0) / width**2
        return r, haar_vector

def ricker_1D(r, domain:HyperParalelipiped, check_if_in_domain, center, width):
    """
    Compute the Ricker wavelet function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Ricker wavelet function.
    - width (float): Width of the Ricker wavelet function.

    Returns:
    - numpy.ndarray: Computed Ricker wavelet function values over the domain.
    """
    A = 2 / (np.sqrt(3 * width) * (np.pi**0.25))
    ricker_specific_width = width / 7
    if check_if_in_domain:
        in_domain = domain.check_if_in_domain(r)
        vector = A * (1 - ((r[in_domain] - center) / ricker_specific_width)**2) * np.exp(-.5 * ((r[in_domain] - center) / ricker_specific_width)**2)
        return r[in_domain], vector
    else:
        vector = A * (1 - ((r - center) / ricker_specific_width)**2) * np.exp(-.5 * ((r - center) / ricker_specific_width)**2)
        return r, vector

def Dgaussian_1D(r, domain:HyperParalelipiped, check_if_in_domain, center, width):
    """
    Compute the Polynomial wavelet function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Polynomial wavelet function.
    - width (float): Width of the Polynomial wavelet function.

    Returns:
    - numpy.ndarray: Computed Polynomial wavelet function values over the domain.
    """
    spread = width / (5 * np.sqrt(2 * np.log(2)))
    if check_if_in_domain:
        in_domain = domain.check_if_in_domain(r)
        Dgaussian_vector = ((r[in_domain] - center) / (spread**3 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r[in_domain] - center) / spread) ** 2)
        return r[in_domain], Dgaussian_vector
    else:
        Dgaussian_vector = ((domain.mesh - center) / (spread**3 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((domain.mesh - center) / spread) ** 2)
        return r, Dgaussian_vector

def boxcar_1D(r, domain:HyperParalelipiped, check_if_in_domain, center, width, unimodularity_precision=1000):
    """
    Compute the Boxcar function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Boxcar function.
    - width (float): Width of the Boxcar function.

    Returns:
    - numpy.ndarray: Computed Boxcar function values over the domain.
    """
    scaled_domain = (domain.dynamic_mesh(unimodularity_precision) - center) / width
    boxcar_vector = np.where(np.abs(scaled_domain) < 0.5, 1 / width, 0)
    area = np.trapz(boxcar_vector, domain.dynamic_mesh(unimodularity_precision))
    if check_if_in_domain:
        in_domain = domain.check_if_in_domain(r)
        scaled_domain = (r[in_domain] - center) / width
        boxcar_vector = np.where(np.abs(scaled_domain) < 0.5, 1 / width, 0)
        return r[in_domain], boxcar_vector / area
    else:
        scaled_domain = (r - center) / width
        boxcar_vector = np.where(np.abs(scaled_domain) < 0.5, 1 / width, 0)
    return r, boxcar_vector / area

def bump_1D(r, domain:HyperParalelipiped, check_if_in_domain, center, width, unimodularity_precision=1000):
    """
    Compute the Bump function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Bump function.
    - width (float): Width of the Bump function.

    Returns:
    - numpy.ndarray: Computed Bump function values over the domain.
    """
    limits = [-0.5 * width + center, 0.5 * width + center]
    mask = (domain.dynamic_mesh(unimodularity_precision) >= limits[0]) & (domain.dynamic_mesh(unimodularity_precision) <= limits[1])
    bump_vector = np.zeros_like(domain.dynamic_mesh(unimodularity_precision))
    bump_vector[mask] = np.exp(1 / ((2 * (domain.dynamic_mesh(unimodularity_precision)[mask] - center) / width) ** 2 - 1))
    area = np.trapz(bump_vector[mask], domain.dynamic_mesh(unimodularity_precision)[mask])
    if check_if_in_domain:
        in_domain = domain.check_if_in_domain(r)
        mask = (r[in_domain] >= limits[0]) & (r[in_domain] <= limits[1])
        bump_vector = np.zeros_like(r[in_domain])
        bump_vector[mask] = np.exp(1 / ((2 * (r[in_domain][mask] - center) / width) ** 2 - 1))
        return r[in_domain], bump_vector / area 
    else:
        mask = (r >= limits[0]) & (r <= limits[1])
        bump_vector = np.zeros_like(r)
        bump_vector[mask] = np.exp(1 / ((2 * (r[mask] - center) / width) ** 2 - 1))
        return r, bump_vector / area 

def Dbump_1D(r, domain:HyperParalelipiped, check_if_in_domain, center, width, unimodularity_precision=1000):
    """
    Compute the Bump derivative function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Bump function.
    - width (float): Width of the Bump function.

    Returns:
    - numpy.ndarray: Computed Bump derivative function values over the domain.
    """
    limits = [-0.5 * width + center, 0.5 * width + center]
    mask = (domain.dynamic_mesh(unimodularity_precision) >= limits[0]) & (domain.dynamic_mesh(unimodularity_precision) <= limits[1])
    bump_vector = np.zeros_like(domain.dynamic_mesh(unimodularity_precision))
    bump_vector[mask] = np.exp(1 / ((2 * (domain.dynamic_mesh(unimodularity_precision)[mask] - center) / width) ** 2 - 1))
    area = np.trapz(bump_vector[mask], domain.dynamic_mesh(unimodularity_precision)[mask])
    if check_if_in_domain:
        in_domain = domain.check_if_in_domain(r)
        mask = (r[in_domain] >= limits[0]) & (r[in_domain] <= limits[1])
        bump_vector = np.zeros_like(r[in_domain])
        bump_vector[mask] = np.exp(1 / ((2 * (r[in_domain][mask] - center) / width) ** 2 - 1))
        bump_vector[mask] *= 8 * (width**2) * (r[in_domain][mask] - center) / ((2*(r[in_domain][mask] - center))**2 - width**2)**2
        return r[in_domain], bump_vector / area 
    else:
        mask = (r >= limits[0]) & (r <= limits[1])
        bump_vector = np.zeros_like(r)
        bump_vector[mask] = np.exp(1 / ((2 * (r[mask] - center) / width) ** 2 - 1))    
        bump_vector[mask] *= 8 * (width**2) * (r[mask] - center) / ((2*(r[mask] - center))**2 - width**2)**2
        return r, bump_vector / area
    
def triangular_1D(r, domain:HyperParalelipiped, check_if_in_domain, center, width):
    """
    Compute the Triangular function over a given domain.

    Args:
    - domain (HyperParalelipiped): Array representing the domain for computation.
    - center (float): Center of the Trinagular function.
    - width (float): Width of the Triangular function.

    Returns:
    - numpy.ndarray: Computed Triangular function values over the domain.
    """
    limits = [-0.5 * width + center, 0.5 * width + center]
    if check_if_in_domain:
        in_domain = domain.check_if_in_domain(r)
        mask = (r[in_domain] >= limits[0]) & (r[in_domain] <= limits[1])
        bump_vector = np.zeros_like(r[in_domain])
        bump_vector[mask] = 2/width - 4*np.abs(r[in_domain][mask] - center)/width**2
        return r[in_domain], bump_vector
    else:
        mask = (r >= limits[0]) & (r <= limits[1])
        bump_vector = np.zeros_like(r)
        bump_vector[mask] = 2/width - 4*np.abs(r[mask] - center)/width**2
        return r, bump_vector