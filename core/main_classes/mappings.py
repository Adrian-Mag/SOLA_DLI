import numpy as np
from abc import ABC, abstractmethod

from core.main_classes.spaces import *

class Mapping(ABC):
    def __init__(self, domain: Space, codomain: Space) -> None:
        super().__init__()
        self.domain = domain
        self.codomain = codomain

    @abstractmethod
    def map(self, member):
        pass

    @abstractmethod
    def adjoint(self):
        pass


class IntegralMapping(Mapping):
    def __init__(self, domain: PCb, codomain: RN, kernels: list) -> None:
        super().__init__(domain, codomain)
        self.kernels = kernels
        self._compute_GramMatrix()
        self.pseudo_inverse = None

    def pseudoinverse_map(self, member: RN):
        result = 0*Constant_1D(domain=self.domain.domain)
        intermediary_result = self.GramMatrix.inverse_map(member)
        for index, kernel in enumerate(self.kernels):
            result = result + intermediary_result[index,0] * kernel
        return result

    def map(self, member: PCb, fineness=None):
        if fineness is None:
            mesh = self.domain.domain.mesh
        else:
            mesh = self.domain.domain.dynamic_mesh(fineness)
        result = np.empty((self.codomain.dimension, 1))
        for index, kernel in enumerate(self.kernels):
            result[index, 0] = scipy.integrate.simpson((kernel*member).evaluate(mesh)[1], mesh)
        return result
    
    def adjoint(self):
        return FunctionMapping(domain=self.codomain, codomain=self.domain,
                               kernels=self.kernels)

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

class FunctionMapping(Mapping):
    def __init__(self, domain: RN, codomain: PCb, kernels: list) -> None:
        super().__init__(domain, codomain)
        self.kernels = kernels

    def map(self, member: RN):
        if self.domain.check_if_member(member):
            result = 0*Constant_1D(domain=self.kernels[0].domain)
            for index, member_i in enumerate(member):
                result = result + member_i[0] * self.kernels[index]
            return result
        else:
            raise Exception('Not a member of RN')
    
    def adjoint(self):
        return IntegralMapping(domain=self.codomain, 
                               codomain=self.domain, 
                               kernels=self.kernels)

class FiniteLinearMapping(Mapping):
    def __init__(self, domain: RN, codomain: RN, matrix: np.ndarray) -> None:
        super().__init__(domain, codomain)
        self.matrix = matrix
    
    def map(self, member: RN):
        return np.dot(self.matrix, member)

    def invert(self):
        if self.domain.dimension == self.codomain.dimension:
            if self.determinant() != 0:
                condition_number = np.linalg.cond(self.matrix)
                if condition_number > 1e10:
                    logging.warning(f'The matrix has a high conditioning number ({condition_number})')
                    inverse = np.linalg.solve(self.matrix, np.eye(self.domain.dimension))
                    return FiniteLinearMapping(domain=self.codomain,
                                                    codomain=self.domain,
                                                    matrix=inverse)
                else:
                    return FiniteLinearMapping(domain=self.codomain, 
                                            codomain=self.domain,
                                            matrix=np.linalg.inv(self.matrix))
            else:
                raise Exception('This linear map has 0 determinant.')
        else:
            raise Exception('Only square matrices may have inverse.')

    def determinant(self):
        return np.linalg.det(self.matrix)

    def adjoint(self):
        return FiniteLinearMapping(domain=self.codomain, 
                                   codomain=self.domain,
                                   matrix=self.matrix.T)

    def __mul__(self, other: Mapping):
        if isinstance(other, FiniteLinearMapping):
            return FiniteLinearMapping(domain=other.domain,
                                       codomain=self.codomain,
                                       matrix=np.dot(self.matrix, other.matrix))
        else:
            raise Exception('Other mapping must also be a FiniteLinearMapping')

    def __add__(self, other: Mapping):
        if isinstance(other, FiniteLinearMapping):
            return FiniteLinearMapping(domain=self.domain,
                                       codomain=self.codomain,
                                       matrix=self.matrix + other.matrix)

    def __sub__(self, other: Mapping):
        if isinstance(other, FiniteLinearMapping):
            return FiniteLinearMapping(domain=self.domain,
                                       codomain=self.codomain,
                                       matrix=self.matrix - other.matrix)