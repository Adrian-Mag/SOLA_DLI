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
        pass  # pragma: no cover

    @abstractmethod
    def check_if_in_domain(self, values):
        """
        Checks if the given values are within the domain.

        This method should be implemented by any concrete subclass of Domain.
        The implementation should return a boolean indicating whether the given
        values are within the domain defined by the subclass.

        Args:
        - values (np.ndarray or float): The values to check. This could be a
        single value or a numpy array of values.

        Returns:
        - bool or np.ndarray: A boolean indicating whether the values are
        within the domain. If 'values' is a numpy array, the return should be a
        numpy array of booleans with the same shape as 'values', where each
        boolean indicates whether the corresponding value in 'values' is within
        the domain.
        """
        pass  # pragma: no cover

    @property
    @abstractmethod
    def total_measure(self):
        """
        Computes and returns the total measure of the domain.

        The total measure of a domain is a concept that depends on the specific
        type of domain. For example, for a rectangular domain, the total
        measure would be the area of the rectangle. For a three-dimensional
        domain, the total measure would be the volume.

        This is an abstract method that must be implemented by any concrete
        subclass of Domain.

        Returns:
        - float: The total measure of the domain. The exact meaning of this
        value depends on the specific type of domain. For a one-dimensional
        domain, this would be the length of the domain. For a two-dimensional
        domain, this would be the area. For a three-dimensional domain, this
        would be the volume, and so on.
        """
        pass  # pragma: no cover

    @abstractmethod
    def __eq__(self, other):
        """
        Checks if this domain is equal to another domain.

        This is an abstract method that must be implemented by any concrete
        subclass of Domain. The implementation should return a boolean
        indicating whether this domain is considered equal to the 'other'
        domain.

        The exact criteria for equality depends on the specific type of domain.
        For example, two domains might be considered equal if they have the
        same bounds, even if they have different meshes.

        Args:
        - other (Domain): The other domain to compare with this domain.

        Returns:
        - bool: True if this domain is equal to the 'other' domain, False
        otherwise.
        """
        pass  # pragma: no cover


class HyperParalelipiped(Domain):
    def __init__(self, bounds: list, fineness: int = 1000) -> None:
        """
        Initializes a new instance of the HyperParalelipiped class.

        The HyperParalelipiped class represents a domain in the form of a
        hyper-parallelepiped, which is a generalization of a parallelepiped to
        an arbitrary number of dimensions.

        The bounds of the domain along each dimension are specified by the
        'bounds' parameter, and the 'fineness' parameter determines the number
        of points along each dimension when creating a mesh for the domain.

        Args:
        - bounds (list): A list of lists, where each inner list represents the
        bounds along one dimension of the domain. Each inner list should
        contain exactly two elements: the lower bound and the upper bound along
        that dimension. The number of inner lists determines the dimension of
        the domain.
        - fineness (int, optional): The number of points along each dimension
        when creating a mesh for the domain. Higher values result in a more
        detailed mesh, but also require more memory and computational
        resources. Defaults to 1000.

        Attributes:
        - dimension (int): The number of dimensions of the domain. This is
        determined by the length of the 'bounds' list.
        - bounds (list): The bounds of the domain along each dimension.
        - fineness (int): The number of points along each dimension for
        meshing.
        - axes (list): The axes of the mesh grid.
        - mesh (list): The mesh grid.
        """
        self.dimension = len(bounds)
        self.bounds = bounds
        # Check the dimension of the domain. For 1D a 1000 fineness is ok, but
        # for higher dimension we should take the 1/dimension power to keep
        # memory manageable
        self.fineness = int(np.power(fineness, 1 / self.dimension))
        self.axes, self.mesh = self._create_mesh()

    def __eq__(self, other):
        """
        Checks if this HyperParalelipiped instance is equal to another.

        Two HyperParalelipiped instances are considered equal if they have the
        same bounds along each dimension. The order of the bounds is also taken
        into account, i.e., two instances with the same bounds but in a
        different order are not considered equal.

        This method overrides the default `__eq__` method to provide a custom
        equality check for HyperParalelipiped instances.

        Args:
        - other (HyperParalelipiped): Another instance of HyperParalelipiped to
        compare with this instance.

        Returns:
        - bool: True if this instance is equal to the 'other' instance (i.e.,
        if they have the same bounds in the same order), False otherwise.
        """
        if isinstance(other, HyperParalelipiped):
            # Check if the bounds are identical, including order
            if self.bounds == other.bounds:
                return True
        return False

    def dynamic_mesh(self, fineness):
        """
        Generates a dynamic mesh for the HyperParalelipiped domain.

        This method creates a mesh grid with a specified fineness. The mesh
        grid is a multi-dimensional grid of points that covers the domain. The
        fineness parameter controls the number of points along each dimension.

        If the domain is one-dimensional, the method returns a one-dimensional
        numpy array. For domains with two or more dimensions, the method
        returns a list of numpy arrays, where each array represents the points
        along one dimension.

        Args:
        - fineness (int): The number of points along each dimension in the mesh
        grid.

        Returns:
        - mesh: If the domain is one-dimensional, a one-dimensional numpy array
        of points. For domains with two or more dimensions, a list of numpy
        arrays, where each array represents the points along one dimension.
        """
        if fineness <= 0:
            raise ValueError('The number of samples must be greater than 0.')
        axes = [
            np.linspace(bound[0], bound[1], fineness)
            for bound in self.bounds
        ]
        if self.dimension == 1:
            mesh = axes[0]
        else:
            mesh = np.meshgrid(*axes)
        return mesh

    def _create_mesh(self):
        """
        Creates a discretized mesh for the HyperParalelipiped domain.

        This method generates a mesh grid based on the bounds and fineness of
        the domain. The mesh grid is a multi-dimensional grid of points that
        covers the domain. The fineness attribute of the domain controls the
        number of points along each dimension.

        If the domain is one-dimensional, the method returns a one-dimensional
        numpy array. For domains with two or more dimensions, the method
        returns a list of numpy arrays, where each array represents the points
        along one dimension.

        This is a private method, intended to be called internally by the
        class. The resulting mesh is used for numerical computations over the
        domain.

        Returns:
        - tuple: A tuple containing two elements:
        - axes (list): A list of numpy arrays, where each array represents the
        points along one dimension.
        - mesh: If the domain is one-dimensional, a one-dimensional numpy array
        of points.
            For domains with two or more dimensions, a list of numpy arrays,
            where each array represents the points along one dimension.
        """
        axes = [
            np.linspace(bound[0], bound[1], self.fineness)
            for bound in self.bounds
            ]
        if self.dimension == 1:
            mesh = axes[0]
        else:
            mesh = np.meshgrid(*axes)
        return axes, mesh

    def sample_domain(self, N=1):
        """
        Randomly samples points from the HyperParalelipiped domain.

        This method generates 'N' random points within the domain. Each point
        is represented as a numpy array of coordinates, one for each dimension
        of the domain. The points are uniformly distributed within the bounds
        of the domain.

        If 'N' is greater than 1, the method returns a 2D numpy array where
        each row represents a point. If 'N' is 1, the method returns a 1D numpy
        array representing a single point.

        Args:
        - N (int, optional): The number of points to sample from the domain.
        Defaults to 1.

        Returns:
        - np.ndarray: A numpy array containing the sampled points. Each point
        is represented as a numpy array of coordinates. If 'N' is greater than
        1, the returned array is 2D (with points as rows). If 'N' is 1, the
        returned array is 1D.
        """
        if N > 1:
            points = [
                np.random.uniform(bound[0], bound[1], N)
                for bound in self.bounds
                ]
            points = np.array(points).T  # Transpose to get points as rows
        else:
            points = np.array([
                np.random.uniform(bound[0], bound[1], 1)
                for bound in self.bounds
                ]).T
        return points

    def check_if_in_domain(self, values):
        """
        Checks if the given value or array of values are within the domain
        bounds.

        This method checks whether a given value or array of values lies within
        the bounds of the domain. The method supports both single values and
        numpy arrays of values. For single values, the method returns a boolean
        indicating whether the value is within the domain. For arrays of
        values, the method returns a list of booleans, where each boolean
        indicates whether the corresponding value is within the domain.

        The method supports domains of any dimension. For one-dimensional
        domains, the method accepts both single values and one-dimensional
        arrays. For multi-dimensional domains, the method accepts arrays of
        values, where each value is an array of coordinates.

        Args:
        - values (np.ndarray or float): The value or array of values to check.
        For multi-dimensional domains, each value should be an array of
        coordinates.

        Returns:
        - bool or list of bool: For single values, a boolean indicating whether
        the value is within the domain. For arrays of values, a list of
        booleans, where each boolean indicates whether the corresponding value
        is within the domain.

        Raises:
        - Exception: If the dimension or type of the values is not compatible
        with the domain.
        """

        if self.dimension == 1:
            if isinstance(values, (int, float)):
                return (values >= self.bounds[0][0] and
                        values <= self.bounds[0][1])
            elif isinstance(values, np.ndarray) and values.ndim == 1:
                return ((values >= self.bounds[0][0]) &
                        (values <= self.bounds[0][1]))
            else:
                raise Exception('Wrong dimension or type')
        else:
            if values.ndim == 1:
                if len(values) == self.dimension:
                    if (np.all(values >= np.array(self.bounds)[:, 0]) and
                            np.all(values <= np.array(self.bounds)[:, 1])):
                        return True
                    else:
                        return False
                else:
                    raise Exception('Wrong dimension or type')
            else:
                values_in_domain = []
                for value in values:
                    if len(value) == self.dimension:
                        if (np.all(value >= np.array(self.bounds)[:, 0]) and
                                np.all(value <= np.array(self.bounds)[:, 1])):
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
        Computes the total measure (volume) of the HyperParalelipiped domain.

        This property calculates the total measure of the domain, which is
        equivalent to the volume in multi-dimensional space. The measure is
        calculated as the product of the lengths of the domain along each
        dimension.

        The lengths are calculated as the difference between the upper and
        lower bounds along each dimension. The product of these lengths gives
        the total measure of the domain.

        This is a read-only property. It's intended to be used for getting the
        total measure of the domain, not for setting it.

        Returns:
        - float: The total measure (volume) of the domain. This is the product
        of the lengths of the domain along each dimension.
        """
        return np.prod([bound[1] - bound[0] for bound in self.bounds])
