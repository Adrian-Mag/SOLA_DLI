from core.aux.domains import HyperParalelipiped
from core.main_classes.spaces import L2Space, RN
from core.aux.function_bank import *
from core.aux.function_creator import FunctionDrawer, as_function

import functools
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


bounds = np.array([[-10,10]])
fineness = 1000
my_domain = HyperParalelipiped(bounds=bounds, fineness=fineness)

gaussian = Triangular_1D(domain=my_domain, center=5, width=5)
plt.plot(gaussian.domain.mesh, gaussian.evaluate(my_domain.mesh)[1])
plt.show()