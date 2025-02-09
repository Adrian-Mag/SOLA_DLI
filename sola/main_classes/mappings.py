import numpy as np
from abc import ABC, abstractmethod
import scipy
from multiprocessing import Pool

from sola.main_classes import spaces, functions, mappings


class Mapping(ABC):
    """
    Abstract base class for mathematical mappings between spaces.

    Attributes:
    - domain (Space): The domain space of the mapping.
    - codomain (Space): The codomain space of the mapping.
    """

    def __init__(self, domain: spaces.Space, codomain: spaces.Space) -> None:
        """
        Initialize a Mapping object.

        Parameters:
        - domain (Space): The domain space of the mapping.
        - codomain (Space): The codomain space of the mapping.
        """
        super().__init__()
        self.domain = domain
        self.codomain = codomain

    @abstractmethod
    def map(self, member):
        """
        Abstract method for mapping a member from the domain to the codomain.

        Parameters:
        - member: The input member to be mapped.

        Returns:
        - The result of mapping the input member.
        """
        pass

    @abstractmethod
    def adjoint(self):
        """
        Abstract method for computing the adjoint of the mapping.

        Returns:
        - The adjoint mapping.
        """
        pass


class Projection(Mapping):
    """
    A class representing a projection mapping.

    Attributes:
        domain (spaces.Space): The domain of the projection.
        codomain (spaces.Subspace): The codomain of the projection.
        basis (list): The basis vectors of the codomain.
        basis_map: The mapping of the basis vectors.
        basis_map_adj: The adjoint of the basis mapping.
        gram_matrix: The Gram matrix of the subspace.
        gram_matrix_inv: The inverse of the Gram matrix.

    Methods:
        map(member): Maps a member of the domain to the codomain.
        _compute_gram_matrix(): Computes the Gram matrix of the subspace.
        adjoint(): Returns the adjoint of the projection.
    """

    def __init__(self, domain: spaces.Space,
                 codomain: spaces.Subspace) -> None:
        """
        Constructs a new Projection object.

        Args:
            domain (spaces.Space): The domain of the projection.
            codomain (spaces.Subspace): The codomain of the projection.
        """
        super().__init__(domain, codomain)
        self.basis = codomain.basis

        if isinstance(domain, spaces.RN):
            self.basis_map = mappings.FiniteLinearMapping(domain=self.domain,
                                                          codomain=spaces.RN(dimension=len(self.basis)), # noqa
                                                          matrix=np.array(codomain.basis)) # noqa
        elif isinstance(domain, spaces.PCb):
            self.basis_map = mappings.IntegralMapping(domain=self.domain,
                                                      codomain=spaces.RN(dimension=len(self.basis)), # noqa
                                                      kernels=self.basis)
        self.basis_map_adj = self.basis_map.adjoint
        self.gram_matrix = self._compute_gram_matrix()
        self.gram_matrix_inv = self.gram_matrix.invert()

    def map(self, member):
        """
        Maps a member of the domain to the codomain.

        Args:
            member: The member of the domain.

        Returns:
            The mapped member in the codomain.
        """
        projected = self.domain.zero
        for base in self.basis:
            projected += self.domain.inner_product(member, base) * base
        return projected

    def _compute_gram_matrix(self):
        """
        Computes the Gram matrix of the subspace.

        Returns:
            The Gram matrix of the subspace.
        """
        gram_matrix = np.zeros((len(self.basis), len(self.basis)))
        for i in range(len(self.basis)):
            for j in range(len(self.basis)):
                if i <= j:
                    gram_matrix[i, j] = self.domain.inner_product(self.basis[i], self.basis[j]) # noqa
                else:
                    gram_matrix[i, j] = gram_matrix[j, i]
        return mappings.FiniteLinearMapping(domain=spaces.RN(dimension=len(self.basis)), # noqa
                                            codomain=spaces.RN(dimension=len(self.basis)), # noqa
                                            matrix=gram_matrix)

    def adjoint(self):
        """
        Returns the adjoint of the projection.

        Returns:
            The adjoint of the projection.
        """
        pass


class DirectSumMapping(Mapping):
    """
    Mapping representing the direct sum of multiple mappings.

    Attributes:
    - domain (Space): The domain space of the direct sum mapping.
    - codomain (Space): The codomain space of the direct sum mapping.
    - mappings (tuple): Tuple of mappings representing the direct sum
    components.
    """

    def __init__(self, domain: spaces.Space, codomain: spaces.Space,
                 mappings: tuple) -> None:
        """
        Initialize a DirectSumMapping object.

        Parameters:
        - domain (Space): The domain space of the direct sum mapping.
        - codomain (Space): The codomain space of the direct sum mapping.
        - mappings (tuple): Tuple of mappings representing the direct sum
        components.
        """
        super().__init__(domain, codomain)
        self.mappings = mappings
        self.kernels = self._obtain_kernels()

        self._adjoint_stored = None
        self._gram_matrix_stored = None

    def _obtain_kernels(self):
        kernels = []
        for index in range(self.codomain.dimension):
            kernels.append(tuple([mapping.kernels[index] for mapping in self.mappings])) # noqa
        return kernels

    def map(self, member: tuple):
        """
        Map a tuple member from the domain to the codomain using direct sum.

        Parameters:
        - member (tuple): The input tuple member.

        Returns:
        - The result of mapping the input tuple member.
        """
        ans = self.codomain.zero
        for sub_member, mapping in zip(member, self.mappings):
            ans += mapping.map(sub_member)

        return ans

    @property
    def adjoint(self):
        """
        Compute the adjoint of the DirectSumMapping.

        Returns:
        - The adjoint DirectSumMapping.
        """
        if self._adjoint_stored is not None:
            return self._adjoint_stored
        else:
            adjoint_mappings = []
            for mapping in self.mappings:
                adjoint_mappings.append(mapping.adjoint)
            self._adjoint_stored = DirectSumMappingAdj(domain=self.codomain,
                                                       codomain=self.domain,
                                                       mappings=tuple(adjoint_mappings)) # noqa
            return self._adjoint_stored

    def pseudoinverse_map(self, member: spaces.RN):
        intermediate = self.gram_matrix.invert().map(member)
        return self.adjoint.map(intermediate)

    @property
    def gram_matrix(self):
        if self._gram_matrix_stored is not None:
            return self._gram_matrix_stored
        else:
            self._gram_matrix_stored = self._compute_GramMatrix()
            return self._gram_matrix_stored

    def _compute_GramMatrix(self, return_matrix_only=False):
        """
        Compute the Gram matrix associated with the DirectSumMapping.

        Parameters:
        - return_matrix_only (bool): If True, only the Gram matrix is returned.

        Returns:
        - If return_matrix_only is True, the Gram matrix; else, a
        FiniteLinearMapping.
        """
        matrix = np.zeros((self.codomain.dimension, self.codomain.dimension))
        for mapping in self.mappings:
            matrix += mapping._compute_GramMatrix(return_matrix_only=True)
        if return_matrix_only:
            return matrix
        else:
            return FiniteLinearMapping(domain=self.codomain,
                                       codomain=self.codomain,
                                       matrix=matrix)

    def _compute_GramMatrix_diag(self):
        """
        Compute the diagonal of the Gram matrix associated with the
        DirectSumMapping.

        Returns:
        - The diagonal of the Gram matrix.
        """
        matrix = np.zeros((self.codomain.dimension, 1))
        for mapping in self.mappings:
            matrix += mapping._compute_GramMatrix_diag()
        return matrix

    def __mul__(self, other: Mapping):
        """
        Multiply the DirectSumMapping by another mapping.

        Parameters:
        - other (Mapping): The mapping to multiply with.

        Returns:
        - The result of the multiplication as a FiniteLinearMapping.
        """
        if isinstance(other, DirectSumMappingAdj):
            matrix = np.zeros((self.codomain.dimension,
                               other.domain.dimension))
            for sub_mapping_1, sub_mapping_2 in zip(self.mappings, other.mappings): # noqa
                matrix += (sub_mapping_1 * sub_mapping_2).matrix

            return FiniteLinearMapping(domain=other.domain,
                                       codomain=self.codomain,
                                       matrix=matrix)

    def project_on_ImG_adj(self, member: tuple):
        """Takes a member of the integra mappings' domain and
            projects it to Im(G*) where G is this mapping.

        Args:
            member (PCb): domain member to be projects

        Returns:
            Function: projection on Im(A*)
        """
        d = self.map(member)
        return self.pseudoinverse_map(d)

    def project_on_kernel(self, member: tuple):
        """Takes a member of the integral mappings' domain and
        projects it to Ker|G where G is this mapping

        Args:
            member (PCb): domain member to be projected

        Returns:
            Function: projection on Ker|G
        """
        member_projected_on_ImG_adj = self.project_on_ImG_adj(member)
        return tuple([f1 - f2 for f1, f2 in zip(member,
                                                member_projected_on_ImG_adj)])


class DirectSumMappingAdj(Mapping):
    """
    Adjoint of the DirectSumMapping.

    Attributes:
    - domain (Space): The domain space of the adjoint mapping.
    - codomain (Space): The codomain space of the adjoint mapping.
    - mappings (tuple): Tuple of mappings representing the direct sum
    components of the adjoint.
    """

    def __init__(self, domain: spaces.Space, codomain: spaces.Space,
                 mappings: tuple) -> None:
        """
        Initialize a DirectSumMappingAdj object.

        Parameters:
        - domain (Space): The domain space of the adjoint mapping.
        - codomain (Space): The codomain space of the adjoint mapping.
        - mappings (tuple): Tuple of mappings representing the direct sum
        components of the adjoint.
        """
        super().__init__(domain, codomain)
        self.mappings = mappings

    def map(self, member):
        """
        Map a member from the domain to the codomain using the adjoint of the
        direct sum.

        Parameters:
        - member: The input member.

        Returns:
        - The result of mapping the input member.
        """
        ans = []
        for mapping in self.mappings:
            ans.append(mapping.map(member))

        return tuple(ans)

    def adjoint(self):
        """
        Compute the adjoint of the DirectSumMappingAdj.

        Returns:
        - The adjoint DirectSumMappingAdj.
        """
        adjoint_mappings = []
        for mapping in self.mappings:
            adjoint_mappings.append(mapping.adjoint())

        return DirectSumMappingAdj(domain=self.codomain,
                                   codomain=self.domain,
                                   mappings=tuple(adjoint_mappings))


class IntegralMapping(Mapping):
    """
    Mapping representing a vector of integrals over a function space.

    Attributes:
    - domain (PCb): The domain space of the integral.
    - codomain (RN): The codomain space of the integral.
    - kernels (list): List of kernels representing the integrand.
    - pseudo_inverse: Placeholder for the pseudo-inverse, initially set to
    None.
    """

    def __init__(self, domain: spaces.PCb, codomain: spaces.RN,
                 kernels: list) -> None:
        """
        Initialize an IntegralMapping object.

        Parameters:
        - domain (PCb): The domain space of the integral.
        - codomain (RN): The codomain space of the integral.
        - kernels (list): List of kernels representing the integrand.
        """
        super().__init__(domain, codomain)
        self.kernels = kernels
        self.pseudo_inverse = None

        self._adjoint_stored = None
        self._gram_matrix_stored = None

    def pseudoinverse_map(self, member: spaces.RN):
        """
        Map a member using the Right Moore-Penrose pseudo-inverse of the
        mapping.

        Parameters:
        - member (RN): The input member.

        Returns:
        - The result of mapping the input member using the pseudoinverse.
        """
        result = functions.Null_1D(domain=self.domain.domain)
        intermediary_result = self.GramMatrix.inverse().map(member)
        for index, kernel in enumerate(self.kernels):
            result = result + intermediary_result[index, 0] * kernel
        return result

    def map(self, member: spaces.PCb, fineness=1000):
        """
        Map a member from the domain to the codomain using integration.

        Parameters:
        - member (PCb): The input member.
        - fineness: Optional parameter for mesh fineness.

        Returns:
        - The result of mapping the input member.
        """
        result = np.empty((self.codomain.dimension, 1))
        for index, kernel in enumerate(self.kernels):
            result[index, 0] = self.domain.inner_product(kernel, member) # noqa
        return result

    @property
    def adjoint(self):
        """
        Compute the adjoint of the IntegralMapping.

        Returns:
        - The adjoint FunctionMapping.
        """
        if self._adjoint_stored is not None:
            return self._adjoint_stored
        else:
            self._adjoint_stored = FunctionMapping(domain=self.codomain,
                                                   codomain=self.domain,
                                                   kernels=self.kernels)
            return self._adjoint_stored

    @property
    def gram_matrix(self):
        if self._gram_matrix_stored is not None:
            return self._gram_matrix_stored
        else:
            self._gram_matrix_stored = self._compute_GramMatrix()
            return self._gram_matrix_stored

    def _compute_GramMatrix(self, return_matrix_only: bool = False, parallel: bool = True):
        """
        Compute the Gram matrix associated with the IntegralMapping.

        Parameters:
        - return_matrix_only (bool): If True, only the Gram matrix is returned.

        Returns:
        - If return_matrix_only is True, the Gram matrix; else, a
        FiniteLinearMapping.
        """
        GramMatrix = np.empty((self.codomain.dimension,
                               self.codomain.dimension))
        if not parallel:
            for i in range(self.codomain.dimension):
                for j in range(i, self.codomain.dimension):
                    entry = self.domain.inner_product(self.kernels[i],
                                                    self.kernels[j])
                    GramMatrix[i, j] = entry
                    if i != j:
                        GramMatrix[j, i] = entry
        else:
            with Pool() as p:
                entries = p.starmap(self._compute_GramMatrix_entry,
                                    [(i, j) for i in range(self.codomain.dimension) for j in range(i, self.codomain.dimension)])
            for entry in entries:
                GramMatrix[entry[0], entry[1]] = entry[2]
                if entry[0] != entry[1]:
                    GramMatrix[entry[1], entry[0]] = entry[2]

        if return_matrix_only:
            return GramMatrix
        else:
            return FiniteLinearMapping(domain=self.codomain,
                                       codomain=self.codomain,
                                       matrix=GramMatrix)

    def _compute_GramMatrix_entry(self, i, j):
        entry = self.domain.inner_product(self.kernels[i], self.kernels[j])
        return i, j , entry

    def _compute_GramMatrix_diag(self):
        """
        Compute the diagonal of the Gram matrix associated with the
        IntegralMapping.

        Returns:
        - The diagonal of the Gram matrix.
        """
        GramMatrix_diag = np.empty((self.codomain.dimension, 1))
        for i in range(self.codomain.dimension):
            entry = self.domain.inner_product(self.kernels[i], self.kernels[i])
            GramMatrix_diag[i] = entry
        return GramMatrix_diag

    def __mul__(self, other: Mapping):
        """
        Multiply the IntegralMapping by another mapping.

        Parameters:
        - other (Mapping): The mapping to multiply with.

        Returns:
        - The result of the multiplication as a FiniteLinearMapping.
        """
        if isinstance(other, FunctionMapping):
            # Compute the matrix defining this composition
            matrix = np.empty((len(self.kernels), len(other.kernels)))
            for i, ker1 in enumerate(self.kernels):
                for j, ker2 in enumerate(other.kernels):
                    # The kernels of both must live in the same space on which
                    # an inner product is defined
                    matrix[i, j] = self.domain.inner_product(ker1, ker2)
            return FiniteLinearMapping(domain=other.domain,
                                       codomain=self.codomain,
                                       matrix=matrix)
        else:
            raise Exception('Other mapping must be a FunctionMapping')

    def project_on_ImG_adj(self, member: spaces.PCb):
        """Takes a member of the integra mappings' domain and
            projects it to Im(G*) where G is this mapping.

        Args:
            member (PCb): domain member to be projects

        Returns:
            Function: projection on Im(A*)
        """
        d = self.map(member)
        return self.pseudoinverse_map(d)

    def project_on_kernel(self, member: spaces.PCb):
        """Takes a member of the integral mappings' domain and
        projects it to Ker|G where G is this mapping

        Args:
            member (PCb): domain member to be projected

        Returns:
            Function: projection on Ker|G
        """
        return member - self.project_on_ImG_adj(member)


class FunctionMapping(Mapping):
    """
    Mapping representing a linear combination of basis functions.

    Attributes:
    - domain (RN): The domain space of the function mapping.
    - codomain (PCb): The codomain space of the function mapping.
    - kernels (list): List of basis functions representing the mapping.
    """

    def __init__(self, domain: spaces.RN, codomain: spaces.PCb,
                 kernels: list) -> None:
        """
        Initialize a FunctionMapping object.

        Parameters:
        - domain (RN): The domain space of the function mapping.
        - codomain (PCb): The codomain space of the function mapping.
        - kernels (list): List of basis functions representing the mapping.
        """
        super().__init__(domain, codomain)
        self.kernels = kernels

    def map(self, member: spaces.RN):
        """
        Map a member from the domain to the codomain using the basis functions.

        Parameters:
        - member (RN): The input member.

        Returns:
        - The result of mapping the input member using the basis functions.
        """
        if self.domain.check_if_member(member):
            result = 0 * functions.Constant_1D(domain=self.kernels[0].domain)
            if type(member) == int or type(member) == float:
                return member * self.kernels[0]
            for index, member_i in enumerate(member):
                result = result + member_i[0] * self.kernels[index]
            return result
        else:
            raise Exception('Not a member of RN')

    def adjoint(self):
        """
        Compute the adjoint of the FunctionMapping.

        Returns:
        - The adjoint IntegralMapping.
        """
        return IntegralMapping(domain=self.codomain,
                               codomain=self.domain,
                               kernels=self.kernels)


class FiniteLinearMapping(Mapping):
    """
    Finite-dimensional linear mapping between two vector spaces (in simpler
    terms, this is just a matrix).

    Attributes:
    - domain (RN): The domain space of the linear mapping.
    - codomain (RN): The codomain space of the linear mapping.
    - matrix (np.ndarray): The matrix representing the linear transformation.
    """

    def __init__(self, domain: spaces.RN, codomain: spaces.RN,
                 matrix: np.ndarray) -> None:
        """
        Initialize a FiniteLinearMapping object.

        Parameters:
        - domain (RN): The domain space of the linear mapping.
        - codomain (RN): The codomain space of the linear mapping.
        - matrix (np.ndarray): The matrix representing the linear
        transformation.
        """
        super().__init__(domain, codomain)
        self.matrix = matrix
        self._adjoint_stored = None

        # For a matrix each row is considered a kernel
        self.kernels = self._get_kernels()

    def map(self, member: spaces.RN):
        """
        Map a member from the domain to the codomain using the linear
        transformation.

        Parameters:
        - member (RN): The input member.

        Returns:
        - The result of mapping the input member.
        """
        if not self.domain.check_if_member(member):
            raise ValueError('Not a member of RN')
        return np.dot(self.matrix, member)

    def invert(self, check_determinant=False):
        """
        Compute the inverse of the linear mapping.

        Parameters:
        - check_determinant (bool): If True, check if the determinant is
        nonzero.

        Returns:
        - The inverse linear mapping as a FiniteLinearMapping or
        ImplicitInvFiniteLinearMapping.
        """
        if self.domain.dimension != self.codomain.dimension:
            raise ValueError('Only square matrices may have inverse.')

        if check_determinant and self.determinant == 0:
            raise ValueError('This linear map has 0 determinant.')

        condition_number = np.linalg.cond(self.matrix)
        if condition_number > 1e10:
            return ImplicitInvFiniteLinearMapping(domain=self.codomain,
                                                  codomain=self.domain,
                                                  inverse_matrix=self.matrix)
        else:
            return FiniteLinearMapping(domain=self.codomain,
                                       codomain=self.domain,
                                       matrix=np.linalg.inv(self.matrix))

    def _get_kernels(self):
        """
        Get the kernels of the linear mapping.

        Returns:
        - The kernels of the linear mapping.
        """
        return [row for row in self.matrix]

    def _compute_GramMatrix(self, return_matrix_only=False):
        """
        Compute the Gram matrix associated with the linear mapping.

        Parameters:
        - return_matrix_only (bool): If True, only the Gram matrix is returned.

        Returns:
        - If return_matrix_only is True, the Gram matrix; else, a
        FiniteLinearMapping.
        """
        matrix = np.dot(self.matrix, self.matrix.T)
        if return_matrix_only:
            return matrix
        else:
            return FiniteLinearMapping(domain=self.codomain,
                                       codomain=self.codomain,
                                       matrix=matrix)

    @property
    def determinant(self):
        """
        Get the determinant of the matrix representing the linear mapping.

        Returns:
        - The determinant of the matrix.
        """
        if self.domain.dimension != self.codomain.dimension:
            raise ValueError('Only square matrices have determinant.')
        return np.linalg.det(self.matrix)

    @property
    def adjoint(self):
        """
        Compute the adjoint of the FiniteLinearMapping.

        Returns:
        - The adjoint FiniteLinearMapping.
        """
        if self._adjoint_stored is not None:
            return self._adjoint_stored
        else:
            self._adjoint_stored = FiniteLinearMapping(domain=self.codomain,
                                                       codomain=self.domain,
                                                       matrix=self.matrix.T)
        return self._adjoint_stored

    def __mul__(self, other: Mapping):
        """
        Multiply the FiniteLinearMapping by another mapping.

        Parameters:
        - other (Mapping): The mapping to multiply with.

        Returns:
        - The result of the multiplication as a FiniteLinearMapping.
        """
        if isinstance(other, FiniteLinearMapping):
            return FiniteLinearMapping(domain=other.domain,
                                       codomain=self.codomain,
                                       matrix=np.dot(self.matrix,
                                                     other.matrix))
        elif isinstance(other, ImplicitInvFiniteLinearMapping):
            other_transposed = other.inverse_matrix.T
            ans = np.ones((self.matrix.shape[0],
                           other.inverse_matrix.shape[0]))
            for i, row in enumerate(self.matrix):
                ans[i, :] = np.linalg.solve(other_transposed,
                                            row.reshape(row.shape[0], 1)).T
            return FiniteLinearMapping(domain=other.domain,
                                       codomain=self.codomain,
                                       matrix=ans)
        elif isinstance(other, IntegralMapping):
            new_kernels = np.dot(self.matrix, other.kernels)
            return IntegralMapping(domain=other.domain, codomain=self.codomain,
                                   kernels=new_kernels)
        elif isinstance(other, DirectSumMapping):
            # Only works when all the constituent mappings are integral
            # mappings
            new_mappings = []
            for map in other.mappings:
                new_kernels = np.dot(self.matrix, map.kernels)
                new_map = IntegralMapping(domain=map.domain,
                                          codomain=self.codomain,
                                          kernels=new_kernels)
                new_mappings.append(new_map)
            return DirectSumMapping(domain=other.domain,
                                    codomain=self.codomain,
                                    mappings=tuple(new_mappings))
        else:
            raise Exception('Other mapping must also be a FiniteLinearMapping')

    def __add__(self, other: Mapping):
        """
        Add another mapping to the FiniteLinearMapping.

        Parameters:
        - other (Mapping): The mapping to add.

        Returns:
        - The result of the addition as a FiniteLinearMapping.
        """
        if isinstance(other, FiniteLinearMapping):
            return FiniteLinearMapping(domain=self.domain,
                                       codomain=self.codomain,
                                       matrix=self.matrix + other.matrix)

    def __sub__(self, other: Mapping):
        """
        Subtract another mapping from the FiniteLinearMapping.

        Parameters:
        - other (Mapping): The mapping to subtract.

        Returns:
        - The result of the subtraction as a FiniteLinearMapping.
        """
        if isinstance(other, FiniteLinearMapping):
            return FiniteLinearMapping(domain=self.domain,
                                       codomain=self.codomain,
                                       matrix=self.matrix - other.matrix)


class ImplicitInvFiniteLinearMapping(Mapping):
    """
    Mapping representing the inverse of a linear transformation implicitly.

    Attributes:
    - domain (RN): The domain space of the linear mapping.
    - codomain (RN): The codomain space of the linear mapping.
    - inverse_matrix (np.ndarray): The inverse matrix of the linear
    transformation.
    """

    def __init__(self, domain: spaces.RN, codomain: spaces.RN,
                 inverse_matrix: np.ndarray) -> None:
        """
        Initialize an ImplicitInvFiniteLinearMapping object.

        Parameters:
        - domain (RN): The domain space of the linear mapping.
        - codomain (RN): The codomain space of the linear mapping.
        - inverse_matrix (np.ndarray): The inverse matrix of the linear
        transformation.
        """
        super().__init__(domain, codomain)
        self.inverse_matrix = inverse_matrix

    def invert(self):
        """
        Compute the inverse of the linear mapping.

        Returns:
        - The inverse linear mapping as a FiniteLinearMapping.
        """
        return FiniteLinearMapping(domain=self.codomain,
                                   codomain=self.domain,
                                   matrix=self.inverse_matrix)

    def map(self, member: spaces.RN):
        """
        Map a member from the domain to the codomain using the inverse linear
        transformation.

        Parameters:
        - member (RN): The input member.

        Returns:
        - The result of mapping the input member using the inverse linear
        transformation.
        """
        return np.linalg.solve(self.inverse_matrix, member)

    def adjoint(self):
        """
        Compute the adjoint of the ImplicitInvFiniteLinearMapping.

        Returns:
        - The adjoint ImplicitInvFiniteLinearMapping.
        """
        return ImplicitInvFiniteLinearMapping(domain=self.codomain,
                                              codomain=self.domain,
                                              inverse_matrix=self.inverse_matrix.T) # noqa

    def __mul__(self, other: Mapping):
        """
        Multiply the ImplicitInvFiniteLinearMapping by another mapping.

        Parameters:
        - other (Mapping): The mapping to multiply with.

        Returns:
        - The result of the multiplication as a FiniteLinearMapping or
        ImplicitInvFiniteLinearMapping.
        """
        if isinstance(other, FiniteLinearMapping):
            answer = np.ones((self.inverse_matrix.shape[0],
                              other.matrix.shape[1]))
            for i, column in enumerate(other.matrix.T):
                answer[:, i] = np.linalg.solve(self.inverse_matrix, column)

            return FiniteLinearMapping(domain=other.domain,
                                       codomain=self.codomain,
                                       matrix=answer)
        elif isinstance(other, ImplicitInvFiniteLinearMapping):
            return ImplicitInvFiniteLinearMapping(domain=other.domain,
                                                  codomain=self.codomain,
                                                  inverse_matrix=np.dot(other.inverse_matrix, # noqa
                                                                        self.inverse_matrix)) # noqa
