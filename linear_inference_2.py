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

# Create domain for functions
bounds = np.array([[0,1]])
fineness = 1000
my_domain = HyperParalelipiped(bounds=bounds, fineness=fineness)

# Create model space
M = L2Space(domain=my_domain)

# Populate with some functions
""" f = FunctionDrawer(domain=my_domain, min_y=0, max_y=10)
f.draw_function()
member1 = Interpolation(values=f.values, 
                        raw_domain=f.raw_domain, 
                        domain=my_domain)
M.add_member(member_name='model1', member=member1)

f.draw_function()
member2 = Interpolation(values=f.values, 
                        raw_domain=f.raw_domain, 
                        domain=my_domain)

M.add_member(member_name='model2', member=member2) """

for i in range(5):
    M.add_member(str(i), M.random_member())

""" for member in M.members.values():
    plt.plot(my_domain.mesh, member.evaluate(my_domain.mesh)[1])
plt.show() """

# Create data space
D = RN(dimension=1)