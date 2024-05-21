from sola.main_classes import domains
from sola.main_classes import spaces
from sola.aux.normal_data import load_normal_data
from sola.main_classes import functions
from sola.main_classes import mappings
from sola.main_classes.SOLA_DLI import Problem
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Set global parameters for matplotlib
plt.rcParams.update({'font.size': 15})  # Set default font size
plt.rcParams['axes.linewidth'] = 1.5  # Set the thickness of the axes lines

####################
# Create model space
####################
# Edit region -------------
physical_parameters = ['m_1', 'm_2', 'm_3']

# Edit region -------------

physical_parameters_symbols = {'m_1': '$m^1$', 'm_2': '$m^2$', 'm_3': '$m^3$'}
no_of_params = len(physical_parameters)
domain = domains.HyperParalelipiped(bounds=[[0, 1]], fineness=1000)
constituent_models_spaces = [spaces.PCb(domain=domain) for _ in physical_parameters]
# Create a dictionary with physical_parameters as keys
models_dict = {param: model_space for param, model_space in zip(physical_parameters, constituent_models_spaces)}
M = spaces.DirectSumSpace(tuple(constituent_models_spaces))

###################
# Create Data space
###################
# Edit region -------------
how_many_data = 150
# Edit region -------------

D = spaces.RN(dimension=how_many_data)