import numpy as np
from abc import ABC, abstractmethod

class Domain(ABC):
    @abstractmethod
    def sample_domain(self, N=1):
        """
        Samples the domain.

        Args:
        - N (int): Number of points to sample. Defaults to 1.

        Returns:
        - np.ndarray: Sampled points within the domain.
        """
        pass
    
    @abstractmethod
    def check_if_in_domain(self, values):
        pass

    @property
    @abstractmethod    
    def total_measure(self):
        """
        Computes the total measure of the domain.

        Returns:
        - float: Total measure of the domain.
        """
        pass

class HyperParalelipiped(Domain):
    def __init__(self, bounds: list, fineness: int) -> None:
        """
        Initializes the HyperParalelipiped domain.

        Args:
        - bounds (list): List o lists of bounds for each dimension of the domain.
        - fineness (int): Number of points along each dimension for meshing.
        """
        self.dimension = len(bounds)
        self.bounds = bounds
        self.fineness = fineness
        self.axes, self.mesh = self._create_mesh()

    def dynamic_mesh(self, fineness):
        axes = [np.linspace(bound[0], bound[1], fineness) for bound in self.bounds]
        if self.dimension == 1:
            mesh = axes[0]
        else:
            mesh = np.meshgrid(*axes)
        return mesh

    def _create_mesh(self):
        """
        Creates the discretized frame for the domain.

        Returns:
        - tuple: Tuple containing axes and the discretized frame.
        """
        axes = [np.linspace(bound[0], bound[1], self.fineness) for bound in self.bounds]
        if self.dimension == 1:
            mesh = axes[0]
        else:
            mesh = np.meshgrid(*axes)
        return axes, mesh

    def sample_domain(self, N=1):
        """
        Samples the domain.

        Args:
        - N (int): Number of points to sample. Defaults to 1.

        Returns:
        - np.ndarray: Sampled points within the domain.
        """
        if N > 1:
            points = [np.random.uniform(bound[0], bound[1], N) for bound in self.bounds]
            points = np.array(points).T  # Transpose to get points as rows
        else: 
            points = np.array([np.random.uniform(bound[0], bound[1], 1) for bound in self.bounds])
        return points

    def check_if_in_domain(self, values):
        """
        Checks if the given value or values are within the domain bounds.

        Args:
        - value (np.ndarray or float): Value or values to check.

        Returns:
        - bool or np.ndarray: Boolean indicating whether the value(s) are within the domain bounds.
        """
        
        if self.dimension == 1:
            if isinstance(values, (int, float)):
                return (values >= self.bounds[0][0] and values <= self.bounds[0][1])  
            elif isinstance(values, np.ndarray) and values.ndim == 1:
                return ((values >= self.bounds[0][0]) & (values <= self.bounds[0][1]))
            else:
                raise Exception('Wrong dimension or type')
        else:
            if values.ndim == 1:
                if len(values) == self.dimension:
                    if np.all((values >= self.bounds[:, 0]) & (values <= self.bounds[:, 1])):
                        return True
                    else:
                        return False
                else: 
                    raise Exception('Wrong dimension or type')
            else:
                values_in_domain = []
                for value in values:
                    if len(value) == self.dimension:
                        if np.all((value >= self.bounds[:, 0]) & (value <= self.bounds[:, 1])):
                            values_in_domain.append(True)
                        else:
                            values_in_domain.append(False)
                    else:
                        values_in_domain.append(False)
                        raise Exception('Wrong dimension or type')
                return values_in_domain

    @property
    def total_measure(self):
        """
        Computes the total measure of the domain.

        Returns:
        - float: Total measure of the domain.
        """
        return np.prod([bound[1] - bound[0] for bound in self.bounds])
