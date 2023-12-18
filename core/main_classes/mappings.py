import numpy as np
from abc import ABC, abstractmethod

from core.main_classes.spaces import *

class Mapping(ABC):
    @abstractmethod
    def map(self, member):
        pass

    @abstractmethod
    def adjoint_map(self, member):
        pass


class IntegralMapping(Mapping):
    def __init__(self, domain: PCb, codomain: RN, kernels: list) -> None:
        self.domain = domain
        self.codomain = codomain
        self.kernels = kernels
        self.GramMatrix = self._compute_GramMatrix()

    def map(self, member: PCb, fineness=None):
        if fineness is None:
            mesh = self.domain.domain.mesh
        else:
            mesh = self.domain.domain.dynamic_mesh(fineness)
        result = np.empty((self.codomain.dimension, 1))
        for index, kernel in enumerate(self.kernels):
            result[index, 0] = scipy.integrate.simpson((kernel*member).evaluate(mesh)[1], mesh)
        return result
    
    def adjoint_map(self, member: RN):
        result = 0*Constant(domain=self.domain.domain)
        for index, kernel in enumerate(self.kernels):
            result = result + member[index] * kernel
        return result

    def _compute_GramMatrix(self):
        GramMatrix = np.empty((self.codomain.dimension,self.codomain.dimension))
        for i in range(self.codomain.dimension):
            for j in range(self.codomain.dimension):
                entry = self.domain.inner_product(self.kernels[i], self.kernels[j])
                GramMatrix[i,j] = entry
                if i!= j:
                    GramMatrix[j, i] = entry
        self.GramMatrix = FiniteLinearMapping(domain=self.codomain, 
                                              codomain=self.codomain, 
                                              matrix=GramMatrix)
        self.GramMatrix.compute_inverse()

class FiniteLinearMapping(Mapping):
    def __init__(self, domain: RN, codomain: RN, matrix: np.ndarray) -> None:
        self.domain = domain
        self.codomain = codomain
        self.matrix = matrix
        self.inverse = None

    def map(self, member: RN):
        return np.dot(self.matrix, member)
    
    def adjoint_map(self, member: RN):
        return np.dot(self.matrix.T, member)
    
    def compute_inverse(self):
        condition_number = np.linalg.cond(self.matrix)
        if  condition_number > 1e15:
            logging.warning(f'The matrix has a high conditioning number ({condition_number}) '
                '-> it will not be inverted. Use np.linalg.solve instead.')
            self.inverse = None
        else:
            self.inverse = np.linalg.inv(self.matrix)

    def inverse_map(self, member: RN):
        if self.inverse is None:
            return np.linalg.solve(self.matrix, member)
        else:
            return np.dot(self.inverse, member)
