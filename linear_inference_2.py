from core.aux.domains import HyperParalelipiped
from core.main_classes.spaces import *
from core.aux.functions import *
from core.aux.function_creator import FunctionDrawer, as_function
from core.main_classes.mappings import *

import functools
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Create domain for functions
bounds = [[0,1]]
fineness = 1000
my_domain = HyperParalelipiped(bounds=bounds, fineness=fineness)

# Create model space
M = PCb(domain=my_domain)

# Create data space
N = 10
D = RN(dimension=N)

# Generate Kernel functions:
kernels = []
for i in range(N):
    kernels.append(NormalModes_1D(domain=my_domain, order=5,
                        spread=0.3, max_freq=1000))
    
# Show the kernels
for i in range(N):
    plt.plot(my_domain.mesh, kernels[i].evaluate(my_domain.mesh)[1])
plt.show()

# Create G data mapping
G = IntegralMapping(domain=M, codomain=D, kernels=kernels)

# Play
# Make some members
box = Boxcar_1D(domain=my_domain, center=0.5, width=0.2)
vector = D.random_member()

func = G.adjoint_map(vector)
plt.plot(my_domain.mesh, func.evaluate(my_domain.mesh)[1])
plt.show()