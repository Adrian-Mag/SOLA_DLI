import numpy as np
from abc import ABC, abstractclassmethod
from core.aux.domains import Domain, HyperParalelipiped
from core.aux.function_creator import FunctionDrawer
import random
import scipy
import logging 
# Configure logging
logging.basicConfig(level=logging.WARNING)  # Set the logging level
from typing import Union, Callable

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


class L2Space(Space):
    def __init__(self, domain:HyperParalelipiped) -> None:
        self.domain = domain
        self.true_model = None
        self.members = {}

    def add_true_model(self, method, args):
        self.true_model = method(*args)

    def draw_function(self, min_y: float, max_y: float):
        function = FunctionDrawer(domain=self.domain, min_y=min_y, max_y=max_y)
        function.draw_function()
        function.interpolate_function()
        save_model = input('Save model? [y/n]: ')
        if save_model == 'y':
            name = input('Model name: ')
            function.save_function(name=name)
        return function.interpolated_values

    def random_member(self, seed=None, continuous=False) -> np.ndarray:
        if seed is None:
            seed = np.random.randint(0,10000)
        random.seed(seed)
    
        if self.domain.dimension == 1:
            # Random number of partitions
            if continuous is True:
                segments = 1
            else:
                segments = random.randint(1,10)
            values = np.zeros_like(self.domain.mesh)
            inpoints = [random.uniform(self.domain.bounds[0][0], self.domain.bounds[0][1]) for _ in range(segments-1)]
            allpoints = sorted(self.domain.bounds[0] + inpoints)
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
            return values

    def validate_callable(self, func):
        # Validate if the callable function follows specified rules
        x_test = np.array([0.5])  # Sample input to test function arguments
        try:
            # Try calling the function with a sample input
            func_value = func(x_test)
            # Check if the output matches the expected shape
            if np.asarray(func_value).shape != x_test.shape:
                return False
        except Exception as e:
            return False
        return True

    def add_member(self, member_name, member: Union[Callable, np.ndarray], domain=None):
        if callable(member):
            if self.validate_callable(member):
                self.members[member_name] = member
            else:
                raise ValueError(
                                "Invalid function format. "
                                "Must accept a float or a 1D ndarray. "
                                "Try using functools.partial to make your function dependent of a single argument."
                                )
        elif isinstance(member, np.ndarray) and member.ndim == 1:
            self.members[member_name] = scipy.interpolate.interp1d(domain, 
                                                                   member, 
                                                                   kind='linear',
                                                                   fill_values='extrapolate')
        else:
            raise ValueError("Invalid member type. Must be a 1D ndarray or a function.")

    def inner_product(self, member1, member2) -> float:
        if type(self.domain) == HyperParalelipiped and self.domain.dimension == 1:
            return scipy.integrate.simpson(member1 * member2, self.domain.mesh)

    def norm(self, member) -> float:
        return np.sqrt(self.inner_product(member, member))


class RN(Space):
    def __init__(self, dimension:int) -> None:
        self.dimensinon = dimension
        self.true_model = None

    def random_member(self, N=1) -> np.ndarray:
        if N > 1:
            return np.vstack([np.random.uniform(-100, 100, self.dimensinon) for _ in range(N)])
        else:
            return np.random.uniform(-100, 100, self.dimensinon)

    def read_member(self, member):
        if len(member) == self.dimensinon:
            return member
        else:
            raise Exception('This cannot be a member of this space.')

    def add_true_model(self, method, args):
        self.true_model = method(*args)

    def inner_product(self, member1, member2) -> float:
        return np.dot(member1, member2)
    
    def norm(self, member) -> float:
        return np.linalg.norm(member)