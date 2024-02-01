from core.main_classes.spaces import PCb, DirectSumSpace, RN
from core.main_classes.mappings import *


class Problem():
    def __init__(self, M: Space, D: Space, P: Space, G: Mapping,
                 T: Mapping, norm_bound: float, data: np.ndarray=None) -> None:
        self.M = M
        self.D = D
        self.P = P
        self.G = G
        self.T = T
        self.data = data
        self.norm_bound = norm_bound

        self.G_adjoint = G.adjoint()
        self.T_adjoint = T.adjoint()

        self.Lambda = None
        self.Lambda_inv = None
        self.Gamma = None
        self.sdata = None
        self.least_norm = None
        self.least_norm_solution = None
        self.X = None

    def change_M(self, new_M: Space, new_G: Mapping, new_T: Mapping):
        self.M = new_M
        self.G = new_G
        self.T = new_T
        self.G_adjoint = new_G.adjoint()
        self.T_adjoint = new_T.adjoint()

        self.Lambda = None
        self.Lambda_inv = None
        self.Gamma = None
        self.sdata = None
        self.least_norm = None
        self.least_norm_solution = None
        self.X = None
    
    def _compute_Lambda(self):
        self.Lambda = self.G._compute_GramMatrix()

    def _compute_Lambda_inv(self):
        self.Lambda_inv = self.Lambda.invert()

    def _compute_sdata(self):
        if self.data is not None and self.Lambda_inv is not None:
            self.sdata = self.Lambda_inv.map(self.data)
        elif self.data is None:
            raise TypeError('The current problem does not have any data. Please add data')
        elif self.Lambda_inv is None:
            self._compute_Lambda()
            self.sdata = self.Lambda_inv.map(self.data)

    def _compute_least_norm(self):
        if self.sdata is not None:
            self.least_norm = self.D.inner_product(self.data, self.sdata)
        else:
            self._compute_sdata()
            self.least_norm = self.D.inner_product(self.data, self.sdata)

    def _compute_least_norm_solution(self):
        if self.sdata is not None:
            self.least_norm_solution = self.G_adjoint.map(self.sdata)
        else:
            self.sdata = self._compute_sdata()
            self.least_norm_solution = self.G_adjoint.map(self.sdata)

    def _compute_Gamma(self):
        self.Gamma = self.T*self.G_adjoint
        self.Gamma_inv = self.Gamma.invert()

    def _compute_X(self):
        self.X = self.Gamma * self.Lambda_inv

    