import numpy as np
from abc import ABC, abstractclassmethod, abstractproperty
from sola.main_classes.domains import HyperParalelipiped
from sola.main_classes import functions
from sola.aux import function_creator
import scipy


class Space(ABC):

    @abstractclassmethod
    def random_member(self):
        pass

    @abstractclassmethod
    def inner_product(self, member1, member2):
        pass

    @abstractclassmethod
    def norm(self, member):
        pass

    @abstractproperty
    def zero(self):
        pass


class DirectSumSpace(Space):
    def __init__(self, spaces: tuple) -> None:
        super().__init__()
        self.spaces = spaces

    def random_member(self, args_list: list):
        list_of_random_members = []
        for space, space_args in zip(self.spaces, args_list):
            list_of_random_members.append(space.random_member(*space_args))

        return tuple(list_of_random_members)

    def inner_product(self, member1: tuple, member2: tuple):
        inner_product = 0
        for sub_member1, sub_member2, space in zip(member1, member2, self.spaces):
            inner_product += space.inner_product(sub_member1, sub_member2)

        return inner_product

    def norm(self, member):
        norm = 0
        for sub_member, space in zip(member, self.spaces):
            norm += space.norm(sub_member)

        return norm

    @property
    def zero(self):
        list_of_zeros = []
        for space in self.spaces:
            list_of_zeros.append(space.zero)

        return tuple(list_of_zeros)


class PCb(Space):
    """
    A class representing a space of functions defined on a given domain.

    Attributes
    ----------
    domain : HyperParalelipiped
        The domain on which the functions are defined.
    members : dict
        A dictionary mapping member names to their corresponding functions.

    Methods
    -------
    draw_member(min_y: float, max_y: float) -> np.ndarray:
        Draws a function on the domain and returns its interpolated values.
    random_member(seed=None, continuous=False, boundaries=None) -> np.ndarray:
        Returns a random function defined on the domain.
    add_member(member_name, member: functions.Function):
        Adds a function to the space.
    inner_product(member1, member2, fineness=None) -> float:
        Calculates the inner product of two functions.
    norm(member) -> float:
        Calculates the norm of a function.
    """

    def __init__(self, domain: HyperParalelipiped) -> None:
        """
        Initializes the space with a given domain.

        Parameters
        ----------
        domain : HyperParalelipiped
            The domain on which the functions are defined.
        """
        self.domain = domain
        self.members = {}

    def draw_member(self, min_y: float, max_y: float):
        """
        Draws a function on the domain and returns its interpolated values.

        Parameters
        ----------
        min_y : float
            The minimum y-value of the function.
        max_y : float
            The maximum y-value of the function.

        Returns
        -------
        np.ndarray
            The interpolated values of the function.
        """
        function = function_creator.FunctionDrawer(domain=self.domain,
                                                   min_y=min_y, max_y=max_y)
        function.draw_function()
        function.interpolate_function()
        return function.interpolated_values

    def random_member(self, seed=None,
                      continuous=False, boundaries=None) -> np.ndarray:
        """
        Returns a random function defined on the domain.

        Parameters
        ----------
        seed : int, optional
            The seed for the random number generator.
        continuous : bool, optional
            Whether the function should be continuous.
        boundaries : tuple, optional
            The boundaries for the function values.

        Returns
        -------
        np.ndarray
            The values of the random function.
        """
        return functions.Random_1D(domain=self.domain, seed=seed,
                                   continuous=continuous,
                                   boundaries=boundaries)

    def add_member(self, member_name, member: functions.Function):
        """
        Adds a function to the space.

        Parameters
        ----------
        member_name : str
            The name of the function.
        member : functions.Function
            The function to add.

        Raises
        ------
        Exception
            If the function is not an instance of functions.Function or if the
            domain of the function does not match the domain of the space.
        """
        if not isinstance(member, functions.Function):
            raise Exception('Only functions can be added as members.')
        if self.domain != member.domain:
            raise Exception('The domain of the function does'
                            ' not match the domain of the space')
        self.members[member_name] = member

    def inner_product(self, member1, member2) -> float:
        """
        Calculates the inner product of two functions.

        Parameters
        ----------
        member1 : functions.Function
            The first function.
        member2 : functions.Function
            The second function.
        fineness : int, optional
            The fineness of the mesh for the integration.

        Returns
        -------
        float
            The inner product of the two functions.

        Raises
        ------
        Exception
            If the inner product is not implemented for the domain.
        """
        if (isinstance(self.domain, HyperParalelipiped) and
           self.domain.dimension == 1):
            # quad requires as input a function that returns only a scalar. My
            # functions always return numpy arrays because they also check
            # wether the input is in the domain and eliminates those that are
            # not in the domain. For this reason, I must transform my function
            # using a lambda to make sure that it returns only a scalar for
            # quad. This should not encounter any issues because quad will
            # input only one element to my function and if it is not in the
            # domain I will get an error for that and if it is in the domain
            # then all will workk fine.
            return scipy.integrate.quad(
                lambda x: (member1.evaluate(x) * member2.evaluate(x))[0],
                self.domain.bounds[0][0], self.domain.bounds[0][1])[0]
        else:
            raise Exception('The inner product is not '
                            'implemented for this domain')

    def norm(self, member) -> float:
        """
        Calculates the norm of a function.

        Parameters
        ----------
        member : functions.Function
            The function.

        Returns
        -------
        float
            The norm of the function.
        """
        return np.sqrt(self.inner_product(member, member))

    @property
    def zero(self):
        """
        Returns the zero function defined on the domain.

        Returns
        -------
        functions.Null_1D
            The zero function.
        """
        return functions.Null_1D(domain=self.domain)


class RN(Space):
    """
    A class to represent a real number space.

    Attributes
    ----------
    dimension : int
        The dimension of the space.
    members : dict
        The members of the space.

    Methods
    -------
    check_if_member(member)
        Checks if a given member is part of the space.
    random_member(N=1)
        Generates a random member of the space.
    add_member(member_name, member)
        Adds a member to the space.
    inner_product(member1, member2)
        Calculates the inner product of two members.
    norm(member)
        Calculates the norm of a member.
    zero
        Returns the zero vector of the space.
    """
    def __init__(self, dimension: int) -> None:
        """
        Constructs all the necessary attributes for the RN object.

        Parameters
        ----------
        dimension : int
            The dimension of the space.

        Examples
        --------
        >>> space = RN(3)
        """
        self.dimension = dimension
        self.members = {}

    def check_if_member(self, member):
        """
        Checks if a given member is part of the space.

        A member is considered part of the space if it is a scalar (for
        1-dimensional spaces) or if it is a numpy array with a shape of
        (self.dimension, 1) for spaces with dimension greater than 1. The dtype
        of the numpy array must be integer or floating point.

        Parameters
        ----------
        member : int, float, or np.ndarray
            The member to check.

        Returns
        -------
        bool
            True if the member is part of the space, False otherwise.

        Examples
        --------
        >>> space = RN(3)
        >>> space.check_if_member(np.array([[1], [2], [3]]))
        True
        """
        if np.isscalar(member):
            return self.dimension == 1
        elif isinstance(member, np.ndarray):
            if member.shape != (self.dimension, 1):
                return False
            if (not np.issubdtype(member.dtype, np.integer) and not
               np.issubdtype(member.dtype, np.floating)):
                return False
            return True
        return False

    def random_member(self, N=1) -> np.ndarray:
        """
        Generates a random member of the space.

        Parameters
        ----------
        N : int, optional
            The number of random members to generate (default is 1).

        Returns
        -------
        np.ndarray
            The generated random member(s).

        Examples
        --------
        >>> space = RN(3)
        >>> space.random_member()
        array([[ 1.12345678],
               [-2.34567891],
               [ 3.45678912]])
        """
        if N > 1:
            if self.dimension > 1:
                return np.array([
                    np.random.uniform(-100, 100, self.dimension)[:, np.newaxis]
                    for _ in range(N)])
            else:
                return np.random.uniform(-100, 100, N)
        else:
            return np.reshape(np.random.uniform(-100, 100, self.dimension),
                              (self.dimension, 1))

    def add_member(self, member_name, member):
        """
        Adds a member to the space.

        Parameters
        ----------
        member_name : str
            The name of the member.
        member : int, float, or np.ndarray
            The member to add.

        Raises
        ------
        Exception
            If the member is not part of the space.

        Examples
        --------
        >>> space = RN(3)
        >>> space.add_member('v', np.array([1, 2, 3]))
        """
        if not self.check_if_member(member):
            raise Exception('Not a member')
        self.members[member_name] = member

    def inner_product(self, member1, member2) -> float:
        """
        Calculates the inner product of two members.

        Parameters
        ----------
        member1 : int, float, or np.ndarray
            The first member.
        member2 : int, float, or np.ndarray
            The second member.

        Returns
        -------
        float
            The inner product of the two members.

        Raises
        ------
        Exception
            If either member is not part of the space.

        Examples
        --------
        >>> space = RN(3)
        >>> space.inner_product(np.array([1, 2, 3]), np.array([4, 5, 6]))
        32.0
        """
        if (not self.check_if_member(member1) or not
           self.check_if_member(member2)):
            raise Exception('Both elements must be members of the space.')
        if self.dimension == 1:
            return member1*member2
        else:
            return np.dot(member1.T, member2)[0, 0]

    def norm(self, member) -> float:
        """
        Calculates the norm of a member.

        Parameters
        ----------
        member : int, float, or np.ndarray
            The member.

        Returns
        -------
        float
            The norm of the member.

        Raises
        ------
        Exception
            If the member is not part of the space.

        Examples
        --------
        >>> space = RN(3)
        >>> space.norm(np.array([1, 2, 3]))
        3.7416573867739413
        """
        if not self.check_if_member(member):
            raise Exception('Element must be a member of the space.')
        return np.linalg.norm(member)

    @property
    def zero(self):
        """
        Returns the zero vector of the space.

        Returns
        -------
        np.ndarray
            The zero vector of the space.

        Examples
        --------
        >>> space = RN(3)
        >>> space.zero
        array([[0.],
               [0.],
               [0.]])
        """
        return np.zeros((self.dimension, 1))
