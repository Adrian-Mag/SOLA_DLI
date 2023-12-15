import numpy as np

class Problem():
    def __init__(self, Model_space='L2', Data_space='RN', Property_space='RN', G='Integral', T='Integral') -> None:
        self.Model_space = Model_space
        self.Data_space = Data_space
        self.Property_space = Property_space
        self.G = G
        self.T = T

    