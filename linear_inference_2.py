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
D_dim = 10
D = RN(dimension=D_dim)

# Create Property space
P_dim = 5
P = RN(dimension=P_dim)

# Generate Kernel functions:
kernels = []
for i in range(D_dim):
    kernels.append(SinusoidalGaussianPolynomial_1D(domain=my_domain, order=5,
                                                   min_val=-5, max_val=5,
                                                   min_f=5, max_f=20,
                                                   spread=0.01))
# Show the kernels
for i in range(D_dim):
    plt.plot(my_domain.mesh, kernels[i].evaluate(my_domain.mesh)[1])
plt.show()
# Create G data mapping
G = IntegralMapping(domain=M, codomain=D, kernels=kernels)

# Create Targets
centers = np.linspace(0,1,P_dim)
width = 0.2
targets = []
for center in centers:
    targets.append(Gaussian_1D(domain=my_domain,
                               center=center,
                               width=width))
# Show the Targets
for i in range(P_dim):
    plt.plot(my_domain.mesh, targets[i].evaluate(my_domain.mesh)[1])
plt.show()
# Create Target mapping
T = IntegralMapping(domain=M, codomain=P, kernels=targets)

# Create a true model
true_model = Random_1D(domain=my_domain, seed=0)
# Create error-free fake data
data = G.map(true_model)

# Least Norm solution via the pseudoinverse
func = G.pseudoinverse_map(data)
plt.plot(my_domain.mesh, true_model.evaluate(my_domain.mesh)[1])
plt.plot(my_domain.mesh, func.evaluate(my_domain.mesh)[1])
plt.show()

# Compute property to data mapping Tau
tau = np.empty((P_dim,D_dim))
for i in range(P_dim):
    for j in range(D_dim):
        tau[i,j] = M.inner_product(targets[i], kernels[j])
Tau = FiniteLinearMapping(domain=D, codomain=P, matrix=tau)

# Compute R
