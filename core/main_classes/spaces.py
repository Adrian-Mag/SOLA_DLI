import numpy as np
from abc import ABC, abstractclassmethod
from core.aux.domains import Domain, HyperParalelipiped
from core.aux.functions import *
from core.aux.function_creator import *
import random
import scipy
import logging 
# Configure logging
logging.basicConfig(level=logging.WARNING)  # Set the logging level
from typing import Union, Callable

class Space(ABC):
    @abstractmethod
    def add_member(self, member_name, member):
        pass

    @abstractclassmethod
    def random_member(self):
        pass

    @abstractclassmethod
    def inner_product(self, member1, member2): 
        pass

    @abstractclassmethod
    def norm(self, member):
        pass


class PCb(Space):
    def __init__(self, domain:HyperParalelipiped) -> None:
        self.domain = domain
        self.members = {}

    def draw_member(self, min_y: float, max_y: float):
        function = FunctionDrawer(domain=self.domain, min_y=min_y, max_y=max_y)
        function.draw_function()
        function.interpolate_function()
        save_model = input('Save model? [y/n]: ')
        if save_model == 'y':
            name = input('Model name: ')
            function.save_function(name=name)
        return function.interpolated_values

    def random_member(self, seed=None, continuous=False) -> np.ndarray:
        return Random_1D(domain=self.domain, seed=seed, continuous=continuous)

    def add_member(self, member_name, member: Function, domain=None):
        if isinstance(member, Function):
            if self.domain == member.domain:
                self.members[member_name] = member
            else: 
                raise Exception('The domain of the function does not match the domain of the space')
        else:
            raise Exception('Only functions can be added as members.')

    def inner_product(self, member1, member2, fineness=None) -> float:
        if fineness is None:
            mesh = self.domain.mesh
        else:
            mesh = self.domain.dynamic_mesh(fineness)
        if type(self.domain) == HyperParalelipiped and self.domain.dimension == 1:
            return scipy.integrate.simpson((member1*member2).evaluate(mesh)[1], mesh)

    def norm(self, member) -> float:
        return np.sqrt(self.inner_product(member, member))


class RN(Space):
    def __init__(self, dimension:int) -> None:
        self.dimension = dimension
        self.members = {}

    def check_if_member(self, member):
        if isinstance(member, (float, int)):
            if self.dimension == 1:
                return True
            else:
                return False
        elif isinstance(member, np.ndarray):
            if member.ndim == 1 or (member.ndim == 2 and member.shape[0] == 1):
                logging.warn('The member is a row vector')
                return False
            else:
                if np.issubdtype(member.dtype, np.integer) or np.issubdtype(member.dtype, np.floating):
                    return True
                else:
                    return False
        else:
            return False

    def random_member(self, N=1) -> np.ndarray:
        if N > 1:
            if self.dimensinon > 1:
                return np.vstack([np.random.uniform(-100, 100, self.dimensinon) for _ in range(N)])
            else:
                return np.random.uniform(-100, 100, N)
        else:
            return np.random.uniform(-100, 100, self.dimension)

    def add_member(self, member_name, member):
        if self.check_if_member(member):
            self.members[member_name] = member
        else:
            raise Exception('Not a member')

    def inner_product(self, member1, member2, check_if_member=False) -> float:
        if check_if_member:
            if self.check_if_member(member1) and self.check_if_member(member2):
                return np.dot(member1, member2)
            else: 
                raise Exception('Both elements must be members of the space.')
        else:            
            return np.dot(member1, member2)
    
    def norm(self, member, check_if_member=False) -> float:
        if check_if_member:
            if self.check_if_member(member):
                return np.linalg.norm(member)
            else: 
                raise Exception('Both elements must be members of the space.')
        else:            
            return np.linalg.norm(member)