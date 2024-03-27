import numpy as np
from abc import ABC, abstractclassmethod, abstractproperty
from sola.main_classes.domains import HyperParalelipiped
from sola.main_classes.functions import *
from sola.aux.function_creator import *
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

    def random_member(self, seed=None, continuous=False, boundaries=None) -> np.ndarray:
        return Random_1D(domain=self.domain, seed=seed, 
                         continuous=continuous, boundaries=boundaries)

    def add_member(self, member_name, member: Function):
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

    @property
    def zero(self):
        return Null_1D(domain=self.domain)

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
                if self.dimension == 1:
                    if member.shape[0] == 1 and (np.issubdtype(member.dtype, np.integer) or np.issubdtype(member.dtype, np.floating)):
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                if np.issubdtype(member.dtype, np.integer) or np.issubdtype(member.dtype, np.floating):
                    return True
                else:
                    return False
        else:
            return False

    def random_member(self, N=1) -> np.ndarray:
        if N > 1:
            if self.dimension > 1:
                return np.array([np.random.uniform(-100, 100, self.dimension)[:, np.newaxis] for _ in range(N)])
            else:
                return np.random.uniform(-100, 100, N)
        else:
            return np.reshape(np.random.uniform(-100, 100, self.dimension), (self.dimension, 1))

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
            if self.dimension == 1:
                return member1*member2
            else:
                return np.dot(member1.T, member2)[0,0]
    
    def norm(self, member, check_if_member=False) -> float:
        if check_if_member:
            if self.check_if_member(member):
                return np.linalg.norm(member)
            else: 
                raise Exception('Both elements must be members of the space.')
        else:            
            return np.linalg.norm(member)
        
    @property
    def zero(self):
        return np.zeros((self.dimension, 1))