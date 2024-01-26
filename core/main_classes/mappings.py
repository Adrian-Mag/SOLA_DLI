import numpy as np
from abc import ABC, abstractmethod, abstractproperty

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


class DirectSumMapping(Mapping):
    def __init__(self, domain: Space, codomain: Space, mappings: tuple) -> None:
        super().__init__(domain, codomain)
        self.mappings = mappings

    def map(self, member: tuple):
        ans = self.codomain.zero
        for sub_member, mapping in zip(member, self.mappings):
            ans += mapping.map(sub_member)

        return ans

    def adjoint(self):
        adjoint_mappings = []
        for mapping in self.mappings:
            adjoint_mappings.append(mapping.adjoint())
        
        return DirectSumMappingAdj(domain=self.codomain,
                                   codomain=self.domain,
                                   mappings=tuple(adjoint_mappings))        
    def _compute_GramMatrix(self, return_matrix_only=False):
        matrix = np.zeros((self.codomain.dimension, self.codomain.dimension))
        for mapping in self.mappings:
            matrix += mapping._compute_GramMatrix(return_matrix_only=True)
        if return_matrix_only:
            return matrix
        else:
            return FiniteLinearMapping(domain=self.codomain, 
                                   codomain=self.codomain,
                                   matrix=matrix)

class DirectSumMappingAdj(Mapping):
    def __init__(self, domain: Space, codomain: Space, mappings: tuple) -> None:
        super().__init__(domain, codomain)
        self.mappings = mappings

    def map(self, member):
        ans = []
        for mapping in self.mappings:
            ans.append(mapping.map(member))

        return tuple(ans)

    def adjoint(self):
        adjoint_mappings = []
        for mapping in self.mappings:
            adjoint_mappings.append(mapping.adjoint())
        
        return DirectSumMappingAdj(domain=self.codomain,
                                   codomain=self.domain,
                                   mappings=tuple(adjoint_mappings)) 


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

    def _compute_GramMatrix(self, return_matrix_only=False):
        GramMatrix = np.empty((self.codomain.dimension,self.codomain.dimension))
        for i in range(self.codomain.dimension):
            for j in range(self.codomain.dimension):
                entry = self.domain.inner_product(self.kernels[i], self.kernels[j])
                GramMatrix[i,j] = entry
                if i!= j:
                    GramMatrix[j, i] = entry
        if return_matrix_only:
            return GramMatrix
        else:
            return FiniteLinearMapping(domain=self.codomain, 
                                    codomain=self.codomain, 
                                    matrix=GramMatrix)
        
    def __mul__(self, other: Mapping):
        if isinstance(other, FunctionMapping):
            # Compute the matrix defining this composition
            matrix = np.empty((len(self.kernels), len(other.kernels)))
            for i, ker1 in enumerate(self.kernels):
                for j, ker2 in enumerate(other.kernels):
                    # The kernels of both must live in the same space on which an inner product is defined
                    matrix[i, j] = self.domain.inner_product(ker1, ker2)
            return FiniteLinearMapping(domain=other.domain, codomain=self.codomain, matrix=matrix)
        else:
            raise Exception('Other mapping must also be a FuncionMapping')


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

    def invert(self, check_determinant=False):
        if self.domain.dimension == self.codomain.dimension:
            if check_determinant:
                if self.determinant != 0:
                    condition_number = np.linalg.cond(self.matrix)
                    if condition_number > 1e10:
                        return ImplicitInvFiniteLinearMapping(domain=self.codomain,
                                                        codomain=self.domain,
                                                        inverse_matrix=self.matrix)
                    else:
                        return FiniteLinearMapping(domain=self.codomain, 
                                                codomain=self.domain,
                                                matrix=np.linalg.inv(self.matrix))
                else:
                    raise Exception('This linear map has 0 determinant.')
            else:
                condition_number = np.linalg.cond(self.matrix)
                if condition_number > 1e10:
                    return ImplicitInvFiniteLinearMapping(domain=self.codomain,
                                                    codomain=self.domain,
                                                    inverse_matrix=self.matrix)
                else:
                    return FiniteLinearMapping(domain=self.codomain, 
                                            codomain=self.domain,
                                            matrix=np.linalg.inv(self.matrix))
        else:
            raise Exception('Only square matrices may have inverse.')

    def _compute_GramMatrix(self, return_matrix_only=False):
        matrix = np.dot(self.matrix, self.matrix.T)
        if return_matrix_only:
            return matrix
        else:
            return FiniteLinearMapping(domain=self.codomain, 
                                   codomain=self.codomain,
                                   matrix=matrix)
    @property
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
        elif isinstance(other, ImplicitInvFiniteLinearMapping):
            other_transposed = other.inverse_matrix.T
            ans = np.ones((self.matrix.shape[0], other.inverse_matrix.shape[0]))
            for i, row in enumerate(self.matrix):
                ans[i, :] = np.linalg.solve(other_transposed, row.reshape(row.shape[0],1)).T
            return FiniteLinearMapping(domain=other.domain,
                                       codomain=self.codomain,
                                       matrix=ans)
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
        
class ImplicitInvFiniteLinearMapping(Mapping):
    def __init__(self, domain: RN, codomain: RN, inverse_matrix: np.ndarray) -> None:
        super().__init__(domain, codomain)
        self.inverse_matrix = inverse_matrix

    def invert(self):
        return FiniteLinearMapping(domain=self.codomain,
                                   codomain=self.domain,
                                   matrix=self.inverse_matrix)
    
    def map(self, member: RN):
        return np.linalg.solve(self.inverse_matrix, member)
    
    def adjoint(self):
        return ImplicitInvFiniteLinearMapping(domain=self.codomain,
                                              codomain=self.domain,
                                              inverse_matrix=self.inverse_matrix.T)

    def __mul__(self, other: Mapping):
        if isinstance(other, FiniteLinearMapping):
            answer = np.ones((self.inverse_matrix.shape[0], other.matrix.shape[1]))
            for i, column in enumerate(other.matrix.T):
                answer[:, i] = np.linalg.solve(self.inverse_matrix, column)

            return FiniteLinearMapping(domain=other.domain, 
                                    codomain=self.codomain, 
                                    matrix=answer)
        elif isinstance(other, ImplicitInvFiniteLinearMapping):
            return ImplicitInvFiniteLinearMapping(domain=other.domain,
                                                  codomain=self.codomain,
                                                  inverse_matrix=np.dot(other.inverse_matrix, self.inverse_matrix))
    