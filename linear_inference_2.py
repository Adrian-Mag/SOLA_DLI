from core.aux.domains import HyperParalelipiped
from core.main_classes.spaces import L2Space, RN
from core.aux.function_bank import *

import functools
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


bounds = np.array([[-10,10]])
fineness = 1000
my_domain = HyperParalelipiped(bounds=bounds, fineness=fineness)
r = np.linspace(-10,10,100)
my_fun = functools.partial(boxcar_1D, center=5, width=5)
r, fun = my_fun(r, my_domain, False)
plt.plot(r, fun)
plt.show()
""" my_space = L2Space(domain=my_domain)

my_space.add_member(member_name='gaussian', 
                    member=functools.partial(gaussian_1D, 
                                             center=0,
                                             width=5)) """