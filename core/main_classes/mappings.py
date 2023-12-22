import numpy as np
from abc import ABC, abstractmethod

from core.main_classes.spaces import *

class Mapping(ABC):
    @abstractmethod
    def map(self, member):
        pass

    @abstractmethod
    def adjoint(self):
        pass


class IntegralMapping(Mapping):
    def __init__(self, domain: PCb, codomain: RN, kernels: list) -> None:
        self.domain = domain
        self.codomain = codomain
        self.kernels = kernels
        self._compute_GramMatrix()
        self.pseudo_inverse = None

    def pseudoinverse_map(self, member: RN):
        result = 0*Constant(domain=self.domain.domain)
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
    
    def map(self, member: RN):
        return np.dot(self.matrix, member)

    def invert(self):
        if self.domain.dimension == self.codomain.dimension:
            if self.determinant() != 0:
                condition_number = np.linalg.cond(self.matrix)
                if condition_number > 1e10:
                    logging.warning(f'The matrix has a high conditioning number ({condition_number}) '
                        '-> it will not be inverted. Will use np.linalg.solve instead.')
                    return _InverseFiniteLinearMapping(domain=self.codomain,
                                                    codomain=self.domain,
                                                    inverse=self.matrix)
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

class _InverseFiniteLinearMapping(Mapping):
    def __init__(self, domain: RN, codomain: RN, inverse: np.ndarray) -> None:
        self.domain = domain
        self.codomain = codomain
        self.inverse = inverse
    
    def map(self, member: RN):
        return np.linalg.solve(self.inverse, member)

    def determinant(self):
        return 1 / np.linalg.det(self.inverse)
    
    def invert(self):
        return FiniteLinearMapping(domain=self.codomain,
                                   codomain=self.domain,
                                   matrix=self.inverse)

    def adjoint(self):
        # Assuming the inverse matrix is real-valued,
        # the adjoint is the transpose of the inverse matrix.
        return _InverseFiniteLinearMapping(domain=self.codomain,
                                           codomain=self.domain,
                                           inverse=self.inverse.T)
        
    def __mul__(self, other: Mapping):
        if isinstance(other, FiniteLinearMapping):
            # Perform the composition of mappings.
            # _InverseFiniteLinearMapping * FiniteLinearMapping
            # This should result in an instance of FiniteLinearMapping
            inverse_matrix = self.map(np.eye(self.domain.dimension))
            composed_matrix = np.dot(inverse_matrix, other.matrix)
            return FiniteLinearMapping(domain=self.domain, 
                                        codomain=other.codomain,
                                        matrix=composed_matrix)
        elif isinstance(other, _InverseFiniteLinearMapping):
            return _InverseFiniteLinearMapping(domain=other.domain,
                                               codomain=self.codomain,
                                               inverse=np.dot(other.inverse, self.inverse))
        else:
            raise Exception('Other mapping must also be a FiniteLinearMapping or _InverseFiniteLinearMapping')