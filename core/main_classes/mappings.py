import numpy as np
from abc import ABC, abstractclassmethod

from core.main_classes.spaces import Space, L2Space, RN

class Mapping(ABC):
    @abstractclassmethod
    def map(self):
        pass

class IntegralMapping():
    def __init__(self, domain: L2Space, codomain: RN, kernels: np.ndarray = None) -> None:
        self.domain = domain
        self.codomain = codomain
        self.kernels = kernels

    def map(self, member):
        pass
