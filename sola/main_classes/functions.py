from sola.main_classes.domains import Domain, HyperParalelipiped

from typing import Tuple, Union
import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg')


class Function(ABC):
    """
    Abstract base class representing a mathematical function. This class
    provides a structure for defining mathematical functions and performing
    operations on them, such as addition, subtraction, multiplication, and
    division.

    Attributes:
    - domain (Domain): The domain of the function. This is the set of all
    possible inputs to the function.

    Methods:
    - evaluate(r, check_if_in_domain=True): Abstract method to evaluate the
    function at given points.
    - is_compatible_with(other): Check compatibility with another function. Two
    functions are compatible if they have the same domain.
    """
    def __init__(self, domain: Domain) -> None:
        """
        Initialize the Function object with a specified domain.

        Parameters:
        - domain (Domain): The domain of the function. This should be an
        instance of the Domain class.
        """
        super().__init__()
        self.domain = domain

    @abstractmethod
    def evaluate(self, r, check_if_in_domain=True):
        """
        Abstract method to evaluate the function at given points. This method
        should be implemented by any concrete subclass.

        Parameters:
        - r: Points at which to evaluate the function. This can be a single
        point or an array of points.
        - check_if_in_domain (bool): Whether to check if points are in the
        function's domain. Default is True. If this is True and a point is not
        in the domain, an exception will be raised.

        Returns:
        - Tuple: A tuple containing the points and the evaluated values. The
        exact structure of this tuple will depend on the specific function.
        """
        pass

    def is_compatible_with(self, other):
        """
        Check compatibility with another function. Two functions are compatible
        if they have the same domain.

        Parameters:
        - other: Another function to check compatibility. This should be an
        instance of the Function class.

        Returns:
        - bool: True if the functions are compatible, otherwise raises an
        exception. An exception is raised if the other function is not of type
        Function or if the two functions have different domains.
        """
        if isinstance(other, Function):
            if self.domain == other.domain:
                return True
            else:
                raise Exception('The two functions have different domains')
        else:
            raise Exception('The other function is not of type Function')

    @abstractmethod
    def __str__(self) -> str:
        """
        Abstract method to convert the function to a string. This method should
        be implemented by any concrete subclass.

        Returns:
        - str: A string representation of the function.
        """
        pass

    def __rmul__(self, other):
        """
        Handle multiplication of the function from the right by a scalar or
        another function. This method delegates to the __mul__ method.

        Parameters:
        - other: The scalar or function to multiply by.

        Returns:
        - Function: The result of the multiplication.
        """
        return self.__mul__(other)

    def __mul__(self, other):
        """
        Multiply the function by a scalar or another compatible function. If
        other is a scalar, the function is scaled by that scalar. If other is a
        function, the two functions are multiplied pointwise.

        Parameters:
        - other: Scalar or another compatible function.

        Returns:
        - Function: Result of the multiplication. This is a new function that
        represents the product of the original function and the scalar or other
        function.
        """
        if isinstance(other, (float, int)):
            return _ScaledFunction(self, other)
        elif self.is_compatible_with(other):
            return _ProductFunction(self, other)

    def __add__(self, other):
        """
        Add another compatible function. The two functions are added pointwise.

        Parameters:
        - other: Another compatible function.

        Returns:
        - Function: Result of the addition. This is a new function that
        represents the sum of the original function and the other function.
        """
        if self.is_compatible_with(other):
            return _SumFunction(self, other)

    def __sub__(self, other):
        """
        Subtract another compatible function. The other function is subtracted
        from the original function pointwise.

        Parameters:
        - other: Another compatible function.

        Returns:
        - Function: Result of the subtraction. This is a new function that
        represents the difference between the original function and the other
        function.
        """
        if self.is_compatible_with(other):
            return _SubtractFunction(self, other)

    def __truediv__(self, other):
        """
        Divide by another compatible function. The original function is divided
        by the other function pointwise.

        Parameters:
        - other: Another compatible function.

        Returns:
        - Function: Result of the division. This is a new function that
        represents the quotient of the original function and the other
        function.
        """
        if self.is_compatible_with(other):
            return _DivideFunction(self, other)


# 1D FUNCTIONS
class _ScaledFunction(Function):
    """
    The _ScaledFunction class inherits from the Function class and represents a
    scaled version of a function. The scaling is performed by multiplying the
    function's output by a scalar value.

    Attributes:
    - function (Function): An instance of the Function class or its subclasses.
    Represents the function to be scaled.
    - scalar (float): A float value by which the function's output will be
    multiplied.

    Methods:
    - evaluate(r, check_if_in_domain=True): Evaluates the scaled function at
    given points.
    """
    def __init__(self, function: Function, scalar: float):
        """
        Constructs all the necessary attributes for the _ScaledFunction object.

        Parameters:
        - function (Function): An instance of the Function class or its
        subclasses. Represents the function to be scaled.
        - scalar (float): A float value by which the function's output will be
        multiplied.
        """
        super().__init__(function.domain)
        self.function = function
        self.scalar = scalar

    def evaluate(self, r: Union[float, int, list, tuple],
                 check_if_in_domain: bool = True,
                 return_in_domain: bool = False) -> Tuple:
        """
        Evaluates the scaled function at given points.

        Parameters:
        - r: Points at which to evaluate the function. Can be a single point
        (float or int) or multiple points (list or tuple).
        - check_if_in_domain (bool): If True, checks if the points are within
        the function's domain before evaluation. Default is True.

        Returns:
        - Tuple: A tuple containing the points and the evaluated values of the
        scaled function. The evaluated values are obtained by multiplying the
        function's output at the points by the scalar attribute.
        """
        eval_function = self.function.evaluate(r, check_if_in_domain,
                                               return_in_domain)
        if return_in_domain:
            return eval_function[0], eval_function[1] * self.scalar
        else:
            return eval_function * self.scalar

    def __str__(self) -> str:
        """
        Returns a string representation of the _ScaledFunction object.

        Returns:
        - str: A string that details the scalar value and the function being
        scaled.
        """
        return (f'_ScaledFunction (scalar={self.scalar}, '
                f'function={self.function})')


class _DivideFunction(Function):
    """
    The _DivideFunction class inherits from the Function class and represents
    the division of two functions.

    Attributes:
    - function1 (Function): An instance of the Function class or its
    subclasses. Represents the numerator function.
    - function2 (Function): An instance of the Function class or its
    subclasses. Represents the denominator function.

    Methods:
    - evaluate(r, check_if_in_domain=True): Evaluates the division of the two
    functions at given points.
    """
    def __init__(self, function1: Function, function2: Function) -> None:
        """
        Constructs all the necessary attributes for the _DivideFunction object.

        Parameters:
        - function1 (Function): An instance of the Function class or its
        subclasses. Represents the numerator function.
        - function2 (Function): An instance of the Function class or its
        subclasses. Represents the denominator function.
        """
        super().__init__(function1.domain)
        self.function1 = function1
        self.function2 = function2

    def evaluate(self, r: Union[float, int, list, tuple],
                 check_if_in_domain: bool = True) -> Tuple:
        """
        Evaluates the division of the two functions at given points.

        Parameters:
        - r: Points at which to evaluate the function. Can be a single point
        (float or int) or multiple points (list or tuple).
        - check_if_in_domain (bool): If True, checks if the points are within
        the function's domain before evaluation. Default is True.

        Returns:
        - Tuple: A tuple containing the points and the evaluated values of the
        division of the two functions.

        Raises:
        - ZeroDivisionError: If the denominator function evaluates to zero at
        any point.
        """
        eval1 = self.function1.evaluate(r, check_if_in_domain)
        eval2 = self.function2.evaluate(r, check_if_in_domain)
        if 0 in eval2[1]:
            raise ZeroDivisionError("Denominator function evaluates to zero at"
                                    " some points.")
        return eval1[0], eval1[1] / eval2[1]

    def __str__(self) -> str:
        """
        Returns a string representation of the _DivideFunction object.

        Returns:
        - str: A string that details the numerator and denominator functions.
        """
        return (f'_DivideFunction (numerator={self.function1}, '
                'denominator={self.function2})')


class _SubtractFunction(Function):
    """
    The _SubtractFunction class inherits from the Function class and represents
    the subtraction of two functions.

    Attributes:
    - function1 (Function): An instance of the Function class or its
    subclasses. Represents the minuend function.
    - function2 (Function): An instance of the Function class or its
    subclasses. Represents the subtrahend function.

    Methods:
    - evaluate(r, check_if_in_domain=True): Evaluates the subtraction of the
    two functions at given points.
    """
    def __init__(self, function1: Function, function2: Function) -> None:
        """
        Constructs all the necessary attributes for the _SubtractFunction
        object.

        Parameters:
        - function1 (Function): An instance of the Function class or its
        subclasses. Represents the minuend function.
        - function2 (Function): An instance of the Function class or its
        subclasses. Represents the subtrahend function.
        """
        super().__init__(function1.domain)
        self.function1 = function1
        self.function2 = function2

    def evaluate(self, r: Union[float, int, list, tuple],
                 check_if_in_domain: bool = True) -> Tuple:
        """
        Evaluates the subtraction of the two functions at given points.

        Parameters:
        - r: Points at which to evaluate the function. Can be a single point
        (float or int) or multiple points (list or tuple).
        - check_if_in_domain (bool): If True, checks if the points are within
        the function's domain before evaluation. Default is True.

        Returns:
        - Tuple: A tuple containing the points and the evaluated values of the
        subtraction of the two functions.
        """
        eval1 = self.function1.evaluate(r, check_if_in_domain)
        eval2 = self.function2.evaluate(r, check_if_in_domain)
        return eval1[0], eval1[1] - eval2[1]

    def __str__(self) -> str:
        """
        Returns a string representation of the _SubtractFunction object.

        Returns:
        - str: A string that details the minuend and subtrahend functions.
        """
        return (f'_SubtractFunction (minuend={self.function1}, '
                'subtrahend={self.function2})')


class _SumFunction(Function):
    """
    The _SumFunction class inherits from the Function class and represents the
    sum of two functions.

    Attributes:
    - function1 (Function): An instance of the Function class or its
    subclasses. Represents the first function to be summed.
    - function2 (Function): An instance of the Function class or its
    subclasses. Represents the second function to be summed.

    Methods:
    - evaluate(r, check_if_in_domain=True): Evaluates the sum of the two
    functions at given points.
    """
    def __init__(self, function1: Function, function2: Function) -> None:
        """
        Constructs all the necessary attributes for the _SumFunction object.

        Parameters:
        - function1 (Function): An instance of the Function class or its
        subclasses. Represents the first function to be summed.
        - function2 (Function): An instance of the Function class or its
        subclasses. Represents the second function to be summed.
        """
        super().__init__(function1.domain)
        self.function1 = function1
        self.function2 = function2

    def evaluate(self, r: Union[float, int, list, tuple],
                 check_if_in_domain: bool = True) -> Tuple:
        """
        Evaluates the sum of the two functions at given points.

        Parameters:
        - r: Points at which to evaluate the function. Can be a single point
        (float or int) or multiple points (list or tuple).
        - check_if_in_domain (bool): If True, checks if the points are within
        the function's domain before evaluation. Default is True.

        Returns:
        - Tuple: A tuple containing the points and the evaluated values of the
        sum of the two functions.
        """
        eval1 = self.function1.evaluate(r, check_if_in_domain)
        eval2 = self.function2.evaluate(r, check_if_in_domain)
        return eval1[0], eval1[1] + eval2[1]

    def __str__(self) -> str:
        """
        Returns a string representation of the _SumFunction object.

        Returns:
        - str: A string that details the first and second functions to be
        summed.
        """
        return (f'_SumFunction (function1={self.function1}, '
                'function2={self.function2})')


class _ProductFunction(Function):
    """
    The _ProductFunction class inherits from the Function class and represents
    the product of two functions.

    Attributes:
    - function1 (Function): An instance of the Function class or its
    subclasses. Represents the first function to be multiplied.
    - function2 (Function): An instance of the Function class or its
    subclasses. Represents the second function to be multiplied.

    Methods:
    - evaluate(r, check_if_in_domain=True): Evaluates the product of the two
    functions at given points.
    """
    def __init__(self, function1: Function, function2: Function) -> None:
        """
        Constructs all the necessary attributes for the _ProductFunction
        object.

        Parameters:
        - function1 (Function): An instance of the Function class or its
        subclasses. Represents the first function to be multiplied.
        - function2 (Function): An instance of the Function class or its
        subclasses. Represents the second function to be multiplied.
        """
        super().__init__(function1.domain)
        self.function1 = function1
        self.function2 = function2

    def evaluate(self, r: Union[float, int, list, tuple],
                 check_if_in_domain: bool = True) -> Tuple:
        """
        Evaluates the product of the two functions at given points.

        Parameters:
        - r: Points at which to evaluate the function. Can be a single point
        (float or int) or multiple points (list or tuple).
        - check_if_in_domain (bool): If True, checks if the points are within
        the function's domain before evaluation. Default is True.

        Returns:
        - Tuple: A tuple containing the points and the evaluated values of the
        product of the two functions.
        """
        eval1 = self.function1.evaluate(r, check_if_in_domain)
        eval2 = self.function2.evaluate(r, check_if_in_domain)
        return eval1[0], eval1[1] * eval2[1]

    def __str__(self) -> str:
        """
        Returns a string representation of the _ProductFunction object.

        Returns:
        - str: A string that details the first and second functions to be
        multiplied.
        """
        return (f'_ProductFunction (function1={self.function1}, '
                'function2={self.function2})')


class Piecewise_1D(Function):
    """
    The Piecewise_1D class inherits from the Function class and represents a
    piecewise function in 1D.

    Attributes:
    - domain (HyperParalelipiped): The domain of the function.
    - intervals (np.ndarray): The intervals of the piecewise function.
    - values (np.ndarray): The values of the function at each interval.

    Methods:
    - evaluate(r, check_if_in_domain=True): Evaluates the piecewise function at
    given points.
    - plot(): Plots the piecewise function.
    """
    def __init__(self, domain: HyperParalelipiped, intervals: np.ndarray,
                 values: np.ndarray) -> None:
        """
        Constructs all the necessary attributes for the Piecewise_1D object.

        Parameters:
        - domain (HyperParalelipiped): The domain of the function.
        - intervals (np.ndarray): The intervals of the piecewise function.
        - values (np.ndarray): The values of the function at each interval.
        """
        super().__init__(domain)
        self.intervals = intervals
        self.values = values

    def _evaluate_piecewise(self, r: np.ndarray) -> np.ndarray:
        """
        Evaluates the piecewise function at given points.

        Parameters:
        - r (np.ndarray): Points at which to evaluate the function.

        Returns:
        - np.ndarray: The evaluated values of the piecewise function.
        """
        # Initialize the result array
        result = np.zeros_like(r, dtype=float)

        # Evaluate the piecewise function for each interval
        for i in range(len(self.values)):
            mask = np.logical_and(self.intervals[i] <= r,
                                  r < self.intervals[i + 1])
            result[mask] = self.values[i]

        # Handle values outside the specified intervals
        result[r < self.intervals[0]] = self.values[0]
        result[r >= self.intervals[-1]] = self.values[-1]

        return result

    def evaluate(self, r: Union[float, int, np.ndarray],
                 check_if_in_domain: bool = True,
                 return_in_domain: bool = False) -> Union[Tuple, np.ndarray]:
        """
        Evaluates the piecewise function at given points.

        Parameters:
        - r: Points at which to evaluate the function. Can be a single point
        (float or int) or multiple points (list or tuple).
        - check_if_in_domain (bool): If True, checks if the points are within
        the function's domain before evaluation. Default is True.

        Returns:
        - Tuple: A tuple containing the points and the evaluated values of the
        piecewise function.
        """
        if isinstance(r, (float, int)):
            r = np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            if return_in_domain:
                return r[in_domain], self._evaluate_piecewise(r[in_domain])
            else:
                return self._evaluate_piecewise(r[in_domain])
        else:
            if return_in_domain:
                return r, self._evaluate_piecewise(r)
            else:
                return self._evaluate_piecewise(r)

    def plot(self):
        """
        Plots the piecewise function.
        """
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh,
                                                 check_if_in_domain=False,
                                                 return_in_domain=True))
        plt.show()

    def __str__(self) -> str:
        """
        Returns a string representation of the Piecewise_1D object.

        Returns:
        - str: A string that details the intervals and values of the piecewise
        function.
        """
        return (f'Piecewise_1D (intervals={self.intervals}, '
                'values={self.values})')


class Null_1D(Function):
    """
    This class represents a null function in one dimension. It is initialized
    with a domain, and can evaluate the null function over this domain. It can
    also plot the function.

    Args:
        domain (HyperParalelipiped): The domain over which the function is
        defined.
    """

    def __init__(self, domain: HyperParalelipiped):
        """
        Initialize the Null_1D function with a given domain.

        Args:
            domain (HyperParalelipiped): The domain over which the function is
            defined.
        """
        super().__init__(domain)

    def plot(self):
        """
        Plot the null function over its domain.
        """
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh))
        plt.title("Plot of the Null_1D function")
        plt.xlabel("Domain")
        plt.ylabel("Function values")
        plt.show()

    def evaluate(self, r, check_if_in_domain=True, return_in_domain=False):
        """
        Evaluate the null function at a given point or array of points.

        Args:
            r (float or numpy.ndarray): The point(s) at which to evaluate the
            function.
            check_if_in_domain (bool, optional): Whether to check if the points
            are in the domain. Defaults to True.

        Returns:
            tuple: A tuple containing the points at which the function was
            evaluated and the function values.
        """
        r = np.array(r, ndmin=1)
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            if return_in_domain:
                return r[in_domain], np.zeros_like(r[in_domain])
            else:
                return np.zeros_like(r[in_domain])
        else:
            if return_in_domain:
                return r, np.zeros_like(r)
            else:
                return np.zeros_like(r)

    def __str__(self) -> str:
        """
        Return a string representation of the function.

        Returns:
            str: The string representation of the function.
        """
        return 'Null_1D'


class Constant_1D(Function):
    """
    This class represents a constant function in one dimension. It is
    initialized with a domain, and can evaluate the constant function over this
    domain. It can also plot the function.

    Args:
        domain (HyperParalelipiped): The domain over which the function is
        defined.
        value (float, optional): The constant value of the function. Defaults
        to 1.
    """

    def __init__(self, domain: HyperParalelipiped, value: float = 1):
        """
        Initialize the Constant_1D function with a given domain and value.

        Args:
            domain (HyperParalelipiped): The domain over which the function is
            defined.
            value (float, optional): The constant value of the function.
            Defaults to 1.
        """
        super().__init__(domain)
        self.value = value

    def plot(self):
        """
        Plot the constant function over its domain.
        """
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh))
        plt.title("Plot of the Constant_1D function")
        plt.xlabel("Domain")
        plt.ylabel("Function values")
        plt.show()

    def evaluate(self, r, check_if_in_domain=True, return_in_domain=False):
        """
        Evaluate the constant function at a given point or array of points.

        Args:
            r (float or numpy.ndarray): The point(s) at which to evaluate the
            function.
            check_if_in_domain (bool, optional): Whether to check if the points
            are in the domain. Defaults to True.

        Returns:
            tuple: A tuple containing the points at which the function was
            evaluated and the function values.
        """
        r = np.array(r, ndmin=1)
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            if return_in_domain:
                return r[in_domain], np.full_like(r[in_domain], self.value)
            else:
                return np.full_like(r[in_domain], self.value)
        else:
            if return_in_domain:
                return r, np.full_like(r, self.value)
            else:
                return np.full_like(r, self.value)

    def __str__(self) -> str:
        """
        Return a string representation of the function.

        Returns:
            str: The string representation of the function.
        """
        return 'Constant_1D'


class Random_1D(Function):
    """
    A class used to represent a one-dimensional random function.

    Attributes
    ----------
    domain : Domain
        The domain of the function.
    seed : float, optional
        The seed for the random number generator.
    continuous : bool, optional
        Whether the function is continuous.
    boundaries : list, optional
        The boundaries of the function.
    function : callable
        The actual function that is created.

    Methods
    -------
    plot():
        Plots the function.
    _create_function():
        Creates the function.
    _determine_segments():
        Determines the number of segments in the function.
    _determine_inpoints(segments):
        Determines the inpoints of the function.
    _create_partitions(inpoints):
        Creates the partitions of the function.
    _create_model(partition):
        Creates a model for a partition of the function.
    evaluate(r, check_if_in_domain=True, return_in_domain=False):
        Evaluates the function at a given point.
    """

    def __init__(self, domain: Domain, seed: float = None,
                 continuous: bool = False, boundaries: list = None) -> None:
        """
        Constructs all the necessary attributes for the Random_1D object.

        Parameters
        ----------
        domain : Domain
            The domain of the function.
        seed : float, optional
            The seed for the random number generator.
        continuous : bool, optional
            Whether the function is continuous.
        boundaries : list, optional
            The boundaries of the function.
        """
        super().__init__(domain)
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        self.continuous = continuous
        self.boundaries = boundaries
        self.function = self._create_function()

    def plot(self):
        """Plots the function."""
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh))
        plt.show()

    def _create_function(self):
        """Creates the function."""
        np.random.seed(self.seed)

        if self.domain.dimension == 1:
            values = np.zeros_like(self.domain.mesh)
            segments = self._determine_segments()
            inpoints = self._determine_inpoints(segments)
            partitions = self._create_partitions(inpoints)

            for _, partition in enumerate(partitions):
                values += self._create_model(partition)

            return interp1d(self.domain.mesh, values*0.1, kind='linear',
                            fill_value='extrapolate')

    def _determine_segments(self):
        """Determines the number of segments in the function."""
        if self.continuous:
            return 1
        elif self.boundaries is not None:
            return len(self.boundaries) + 1
        else:
            return np.random.randint(1, 10)

    def _determine_inpoints(self, segments):
        """
        Determines the inpoints of the function.

        Parameters
        ----------
        segments : int
            The number of segments in the function.
        """
        if self.boundaries is not None and not self.continuous:
            return self.boundaries
        else:
            lower_bound, upper_bound = self.domain.bounds[0]
            return [np.random.uniform(lower_bound,
                                      upper_bound) for _ in range(segments-1)]

    def _create_partitions(self, inpoints):
        """
        Creates the partitions of the function.

        Parameters
        ----------
        inpoints : list
            The inpoints of the function.
        """
        allpoints = sorted(list(self.domain.bounds[0]) + inpoints)
        return list(zip(allpoints, allpoints[1:]))

    def _create_model(self, partition):
        """
        Creates a model for a partition of the function.

        Parameters
        ----------
        partition : tuple
            The start and end points of the partition.
        """
        x_shift = np.random.uniform(partition[0], partition[1])
        a0, a1, a2, a3 = np.random.uniform(-1, 1, 4)
        stretch = np.max(self.domain.mesh)
        shifted_mesh = (self.domain.mesh + x_shift) / stretch
        model = a0
        model += a1 * shifted_mesh
        model += a2 * shifted_mesh**2
        model += a3 * shifted_mesh**3
        model[self.domain.mesh < partition[0]] = 0
        model[self.domain.mesh > partition[1]] = 0
        return model

    def evaluate(self, r, check_if_in_domain=True, return_in_domain=False):
        """
        Evaluates the function at a given point.

        Parameters
        ----------
        r : float
            The point at which to evaluate the function.
        check_if_in_domain : bool, optional
            Whether to check if the point is in the domain of the function.
        return_in_domain : bool, optional
            Whether to return the point if it is in the domain of the function.
        """
        r = np.array(r, ndmin=1)
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            if return_in_domain:
                return r[in_domain], self.function(r[in_domain])
            else:
                return self.function(r[in_domain])
        else:
            if return_in_domain:
                return r, self.function(r)
            else:
                return self.function(r)

    def __str__(self) -> str:
        """Returns the string representation of the function."""
        return 'random1d'


class Interpolation_1D(Function):
    """
    A class used to represent a 1D interpolation function.

    ...

    Attributes
    ----------
    values : array_like
        The y-coordinates of the data points.
    raw_domain : array_like
        The x-coordinates of the data points.
    domain : Domain
        The domain of the function.

    Methods
    -------
    plot():
        Plots the interpolation function.
    evaluate(r, check_if_in_domain=True):
        Evaluates the interpolation function at the points r.
    """

    def __init__(self, values, raw_domain, domain: Domain) -> None:
        """
        Constructs all the necessary attributes for the Interpolation_1D
        object.

        Parameters
        ----------
            values : array_like
                The y-coordinates of the data points.
            raw_domain : array_like
                The x-coordinates of the data points.
            domain : Domain
                The domain of the function.
        """
        super().__init__(domain)
        self.values = values
        self.raw_domain = raw_domain

    def plot(self):
        """Plots the interpolation function."""
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True, return_in_domain=False):
        """
        Evaluates the interpolation function at the points r.

        Parameters
        ----------
            r : array_like
                The points at which to evaluate the interpolation function.
            check_if_in_domain : bool, optional
                Whether to check if the points r are in the domain (default is
                True).
            return_in_domain : bool, optional
                Whether to return the points r if they are in the domain
                (default is False).

        Returns
        -------
            array_like
                The values of the interpolation function at the points r.
        """
        r = np.array(r, ndmin=1)
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            if return_in_domain:
                return r[in_domain], interp1d(self.raw_domain, self.values,
                                              kind='linear',
                                              fill_value='extrapolate'
                                              )(r[in_domain])
            else:
                return interp1d(self.raw_domain, self.values, kind='linear',
                                fill_value='extrapolate')(r[in_domain])
        else:
            if return_in_domain:
                return r, interp1d(self.raw_domain, self.values, kind='linear',
                                   fill_value='extrapolate')(r)
            else:
                return interp1d(self.raw_domain, self.values, kind='linear',
                                fill_value='extrapolate')(r)

    def __str__(self) -> str:
        """Returns the string representation of the Interpolation_1D object."""
        return 'interpolation_1d'


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
        super().__init__(domain)
        self.frequency = frequency

    def evaluate(self, r, check_if_in_domain=True):
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            fourier_vector = np.exp(-2 * np.pi * self.frequency * 1j * r[in_domain] / self.domain.total_measure) / self.domain.total_measure
            return r[in_domain], fourier_vector
        else:
            fourier_vector = np.exp(-2 * np.pi * self.frequency * 1j * r / self.domain.total_measure) / self.domain.total_measure
            return r, fourier_vector

    def __str__(self) -> str:
        return 'ComplExponential_1D'

class Polynomial_1D(Function):
    def __init__(self, domain: HyperParalelipiped, order, min_val, max_val):
        super().__init__(domain)
        self.order = order
        self.min_val = min_val
        self.max_val = max_val
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
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            poly_function = np.poly1d(self.coefficients)
            return r[in_domain], poly_function(r[in_domain])
        else:
            poly_function = np.poly1d(self.coefficients)
            return r, poly_function(r)

    def __str__(self) -> str:
        return 'Polynomial_1D'

class SinusoidalPolynomial_1D(Function):
    def __init__(self, domain: HyperParalelipiped, order, min_val, max_val, min_f, max_f, seed):
        super().__init__(domain)
        self.order = order
        self.min_val = min_val
        self.max_val = max_val
        self.min_f = min_f
        self.max_f = max_f
        self.seed = seed
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
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            return r[in_domain], self.poly_with_sinusoidal(r[in_domain])
        else:
            return r, self.poly_with_sinusoidal(r)

    def __str__(self) -> str:
        return 'SinusoidalPolynomiall_1D'

class SinusoidalGaussianPolynomial_1D(Function):
    def __init__(self, domain: HyperParalelipiped, order, min_val, max_val, min_f, max_f, spread, seed=None):
        super().__init__(domain)
        self.order = order
        self.min_val = min_val
        self.max_val = max_val
        self.min_f = min_f
        self.max_f = max_f
        self.spread = spread
        self.seed = seed
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
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            gaussian_function = (1 / (self.std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r[in_domain] - self.mean) / self.std_dev) ** 2)
            return r[in_domain], poly_with_sinusoidal(r[in_domain]) * gaussian_function
        else:
            gaussian_function = (1 / (self.std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - self.mean) / self.std_dev) ** 2)
            return r, poly_with_sinusoidal(r) * gaussian_function

    def __str__(self) -> str:
        return 'SinusoidalGaussianPolynomial_1D'

class NormalModes_1D(Function):
    def __init__(self, domain: HyperParalelipiped, order, spread, max_freq,
                 no_sensitivity_regions: np.array=None, seed=None):
        super().__init__(domain)
        self.order = order
        self.seed = seed
        self.spread = spread
        self.max_freq = max_freq
        self.no_sensitivity_regions = no_sensitivity_regions
        self.coefficients, self.shifts = self.generate_random_parameters()
        self.mean, self.std_dev, self.frequency, self.shift = self.generate_function_parameters()

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
        shift = np.random.uniform(0, np.pi)

        # Generate Gaussian function parameters
        mean = np.random.uniform(self.domain.bounds[0][0], self.domain.bounds[0][-1])
        std_dev = np.random.uniform(self.spread / 2, self.spread * 2)

        return mean, std_dev, frequency, shift

    def evaluate(self, r, check_if_in_domain=True):
        shifted_poly = np.zeros_like(r)
        for i in range(self.order):
            shifted_domain = r - self.shifts[i]*(np.max(r) - np.min(r))
            shifted_poly += np.power(shifted_domain, i + 1)
        try: r[0]
        except: r=np.array([r])
        sin_poly = np.sin(r * self.frequency + self.shift/(np.max(r) - np.min(r))) * shifted_poly

        if self.no_sensitivity_regions is not None:
            for region in self.no_sensitivity_regions:
                sin_poly[(r >= region[0]) & (r <= region[1])] = 0

        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            gaussian_function = (1 / (self.std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r[in_domain] - self.mean) / self.std_dev) ** 2)
            return r[in_domain], sin_poly[in_domain] * gaussian_function
        else:
            gaussian_function = (1 / (self.std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - self.mean) / self.std_dev) ** 2)
            return r, sin_poly * gaussian_function

    def __str__(self) -> str:
        return 'NormalModes_1D'

class Gaussian_Bump_1D(Function):
    """
    Compute the a Gaussian lookinf bump function over a given domain.

    Args:
    - center (float): Center of the compact domain.
    - width (float): Width of the compact domain.
    - domain (HyperParalelipiped): Array representing the domain for computation.

    Returns:
    - numpy.ndarray: Computed function values over the domain.
    """
    def __init__(self, domain:HyperParalelipiped, center, width, pointiness=2,
                 unimodularity_precision=1000) -> None:
        super().__init__(domain=domain)
        self.center = center
        self.width = width
        self.pointiness = pointiness
        self.unimodularity_precision = unimodularity_precision
        self._normalization_stored = None

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    @property
    def normalization(self):
        if self._normalization_stored is None:
            r = np.linspace(-1,1,self.unimodularity_precision)
            bump = np.exp(1/(r**2 - 1) - self.pointiness * r**2)
            bump = np.where(np.isinf(bump),0,bump)
            return (self.width / 2) * np.trapz(bump, r)
        else:
            return self._normalization_stored

    def evaluate(self, r, check_if_in_domain=True):
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            where_compact = np.where((r>(self.center - self.width/2)) & (r<(self.center + self.width/2)))
            r_compact = r[in_domain][where_compact]
            r_compact_centered = r_compact - self.center
            bump = np.zeros_like(r[in_domain])
            bump[where_compact] = np.exp(((self.width/2)**4 - self.pointiness * r_compact_centered**2 * (r_compact_centered**2 - (self.width/2)**2))/ \
                          ((self.width/2)**2 * r_compact_centered**2 - (self.width/2)**4))
            bump = np.where(np.isinf(bump),0,bump)
            bump /= self.normalization
            return r[in_domain], bump
        else:
            where_compact = np.where((r>(self.center - self.width/2)) & (r<(self.center + self.width/2)))
            r_compact = r[where_compact]
            r_compact_centered = r_compact - self.center
            bump = np.zeros_like(r)
            bump[where_compact] = np.exp(((self.width/2)**4 - self.pointiness * r_compact_centered**2 * (r_compact_centered**2 - (self.width/2)**2))/ \
                          ((self.width/2)**2 * r_compact_centered**2 - (self.width/2)**4))
            bump = np.where(np.isinf(bump),0,bump)
            bump /= self.normalization
            return r, bump

    def __str__(self) -> str:
        return 'Gaussian_Bump_1D'

class Dgaussian_Bump_1D(Function):
    """
    Compute the derivative of Gaussian bump function over a given domain.

    Args:
    - center (float): Center of the Gaussian function.
    - width (float): Width of the Gaussian function.
    - domain (HyperParalelipiped): Array representing the domain for computation.

    Returns:
    - numpy.ndarray: Computed Gaussian function values over the domain.
    """
    def __init__(self, domain:HyperParalelipiped, center, width, pointiness=2,
                 unimodularity_precision=1000) -> None:
        super().__init__(domain=domain)
        self.center = center
        self.width = width
        self.pointiness = pointiness
        self.unimodularity_precision = unimodularity_precision
        self._normalziation_stored = None

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            where_compact = np.where((r[in_domain]>(self.center - self.width/2)) & (r[in_domain]<(self.center + self.width/2)))
            r_compact = r[in_domain][where_compact]
            dbump = np.zeros_like(r[in_domain])
            r_compact_centered = r_compact - self.center
            multiplier = (-(2 * self.pointiness * r_compact_centered)/(self.width/2)**2 - \
                        (2 * (self.width/2)**2 * r_compact_centered)/((r_compact_centered**2 - (self.width/2)**2)**2))
            bump = Gaussian_Bump_1D(domain=self.domain, center=self.center,
                                    width=self.width, pointiness=self.pointiness)
            dbump[where_compact] = -multiplier * bump.evaluate(r_compact)[1]

            return r[in_domain], dbump
        else:
            where_compact = np.where((r>(self.center - self.width/2)) & (r<(self.center + self.width/2)))
            r_compact = r[where_compact]
            r_compact_centered = r_compact - self.center
            dbump = np.zeros_like(r)
            multiplier =  (-(2 * self.pointiness * (r_compact_centered))/(self.width/2)**2 - \
                                     (2 * (self.width/2)**2 * (r_compact_centered))/(((r_compact_centered)**2 - (self.width/2)**2)**2))
            bump = Gaussian_Bump_1D(domain=self.domain, center=self.center,
                                    width=self.width, pointiness=self.pointiness)
            dbump[where_compact] = -multiplier * bump.evaluate(r_compact, check_if_in_domain=False)[1]
            return r, dbump

    def __str__(self) -> str:
        return 'Dgaussian_Bump_1D'

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

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            gaussian_vector = (1 / (self.spread * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r[in_domain] - self.center) / self.spread) ** 2)
            return r[in_domain], gaussian_vector
        else:
            gaussian_vector = (1 / (self.spread * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - self.center) / self.spread) ** 2)
            return r, gaussian_vector

    def __str__(self) -> str:
        return 'Gaussian_1D'

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
        super().__init__(domain)
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
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            moorlet_vector = np.cos(self.frequency * (r[in_domain] - self.center)) \
                * np.exp(-0.5 * ((r[in_domain] - self.center) / self.spread) ** 2)
            return r[in_domain], moorlet_vector / self.normalization
        else:
            moorlet_vector = np.cos(self.frequency * (r - self.center)) \
                * np.exp(-0.5 * ((r - self.center) / self.spread) ** 2)
            return r, moorlet_vector / self.normalization

    def __str__(self) -> str:
        return 'Moorlet_1D'

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
        super().__init__(domain)
        self.center = center
        self.width = width

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            scaled_domain = (r[in_domain] - self.center) / self.width
            haar_vector = 4 * np.where((scaled_domain >= -0.5) & (scaled_domain < 0.5), np.sign(scaled_domain), 0) / self.width**2
            return r[in_domain], haar_vector[in_domain]
        else:
            scaled_domain = (r - self.center) / self.width
            haar_vector = 4 * np.where((scaled_domain >= -0.5) & (scaled_domain < 0.5), np.sign(scaled_domain), 0) / self.width**2
            return r, haar_vector

    def __str__(self) -> str:
        return 'Haar_1D'

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
        super().__init__(domain)
        self.center = center
        self.width = width

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        A = 2 / (np.sqrt(3 * self.width) * (np.pi ** 0.25))
        ricker_specific_width = self.width / 7
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            vector = A * (1 - ((r[in_domain] - self.center) / ricker_specific_width) ** 2) * np.exp(
                -0.5 * ((r[in_domain] - self.center) / ricker_specific_width) ** 2)
            return r[in_domain], vector
        else:
            vector = A * (1 - ((r - self.center) / ricker_specific_width) ** 2) * np.exp(
                -0.5 * ((r - self.center) / ricker_specific_width) ** 2)
            return r, vector

    def __str__(self) -> str:
        return 'Ricker_1D'

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
        super().__init__(domain)
        self.center = center
        self.width = width
        self.spread = width / (5 * np.sqrt(2 * np.log(2)))

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            Dgaussian_vector = ((r[in_domain] - self.center) / (self.spread ** 3 * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((r[in_domain] - self.center) / self.spread) ** 2)
            return r[in_domain], Dgaussian_vector
        else:
            Dgaussian_vector = ((r - self.center) / (self.spread ** 3 * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((r - self.center) / self.spread) ** 2)
            return r, Dgaussian_vector

    def __str__(self) -> str:
        return 'Dgaussian_1D'

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
    def __init__(self, domain: HyperParalelipiped, center, width,
                 unimodularity_precision=1000):
        super().__init__(domain)
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

    def __str__(self) -> str:
        return 'Boxcar_1D'

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
        super().__init__(domain)
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
        try: r[0]
        except: r=np.array([r])
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

    def __str__(self) -> str:
        return 'Bump_1D'

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
        super().__init__(domain)
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
        try: r[0]
        except: r=np.array([r])
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

    def __str__(self) -> str:
        return 'Dbump_1D'

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
        super().__init__(domain)
        self.center = center
        self.width = width

    def plot(self):
        plt.plot(self.domain.mesh, self.evaluate(self.domain.mesh)[1])
        plt.show()

    def evaluate(self, r, check_if_in_domain=True):
        try: r[0]
        except: r=np.array([r])
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

    def __str__(self) -> str:
        return 'Triangular_1D'

class Fourier(Function):
    """
    Compute Furier basis functions on a domain of type [0, P]
    """
    def __init__(self, domain: HyperParalelipiped, type: str, order: int) -> None:
        super().__init__(domain)
        self.type = type
        self.order = order
        self.period = self.domain.bounds[0][1] - self.domain.bounds[0][0]

    def evaluate(self, r, check_if_in_domain=True):
        try: r[0]
        except: r=np.array([r])
        if check_if_in_domain:
            in_domain = self.domain.check_if_in_domain(r)
            if self.order == 0:
                return r[in_domain], np.ones_like(r[in_domain]) / np.sqrt(self.period)
            else:
                if self.type == 'sin':
                    return r[in_domain], np.sin(2*np.pi*self.order*r[in_domain]/self.period) * np.sqrt(2/self.period)
                else:
                    return r[in_domain], np.cos(2*np.pi*self.order*r[in_domain]/self.period) * np.sqrt(2/self.period)
        else:
            if self.order == 0:
                return r, np.ones_like(r) / np.sqrt(self.period)
            else:
                if self.type == 'sin':
                    return r, np.sin(2*np.pi*self.order*r/self.period) * np.sqrt(2/self.period)
                else:
                    return r, np.cos(2*np.pi*self.order*r/self.period) * np.sqrt(2/self.period)

    def __str__(self) -> str:
        return 'Fourier'