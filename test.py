from core.main_classes.domains import HyperParalelipiped
from core.main_classes.spaces import PCb, DirectSumSpace, RN
from core.aux.normal_data import load_normal_data
from core.main_classes.functions import *
from core.main_classes.mappings import *
from core.main_classes.SOLA_DLI import Problem
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

# Set global parameters for matplotlib
plt.rcParams.update({'font.size': 15})  # Set default font size
plt.rcParams['axes.linewidth'] = 1.5  # Set the thickness of the axes lines

# Create color Palette
colors = sns.color_palette('YlGnBu', n_colors=100)

####################
# Create model space
####################
# Edit region -------------
physical_parameters = ['m_1', 'm_2', 'm_3']

# Edit region -------------

physical_parameters_symbols = {'m_1': '$m_1$', 'm_2': '$m_2$', 'm_3': '$m_3$'}
no_of_params = len(physical_parameters)
domain = HyperParalelipiped(bounds=[[0, 1]], fineness=1000)
constituent_models_spaces = [PCb(domain=domain) for _ in physical_parameters]
# Create a dictionary with physical_parameters as keys
models_dict = {param: model_space for param, model_space in zip(physical_parameters, constituent_models_spaces)}
M = DirectSumSpace(tuple(constituent_models_spaces))

###################
# Create Data space
###################
# Edit region -------------
how_many_data = 150
# Edit region -------------

D = RN(dimension=how_many_data)

###########################
# Create model-data mapping 
###########################
# Make them into functions via interpolation
sensitivity_dict = {}
for i, param in enumerate(physical_parameters):
    sensitivity_dict[param] = []
    for index in range(how_many_data):
        if i in [0, 2]:
            sensitivity_dict[param].append(NormalModes_1D(domain=domain, order=3, spread=0.05,
                                                        max_freq=10, seed=index + i*how_many_data))
        else:
            sensitivity_dict[param].append(NormalModes_1D(domain=domain, order=3, spread=0.05,
                                                        max_freq=10, seed=index + i*how_many_data, 
                                                        no_sensitivity_regions=[[0.5, 0.75]]))

constituent_mappings = [IntegralMapping(domain=models_dict[param], codomain=D, 
                                        kernels=sensitivity_dict[param]) for param in physical_parameters]
mappings_dict = {param: mapping for param, mapping in zip(physical_parameters, constituent_mappings)}
G = DirectSumMapping(domain=M, codomain=D, mappings=tuple(constituent_mappings))

###################################
# Create property mapping and space
###################################
# Edit region -------------
target_types = {'m_1': Null_1D,
                'm_2': Gaussian_1D,
                'm_3': Null_1D}
max_spread = 1
min_spread = 1e-2
N_enquiry_points = 20
N_widths = 20
# Edit region -------------
how_many_targets = N_enquiry_points * N_widths
enquiry_points = np.linspace(domain.bounds[0][0], 
                             domain.bounds[0][1], 
                             N_enquiry_points)
widths = np.logspace(np.log10(min_spread), np.log10(max_spread), N_widths) # same units as domain (km here)
combinations = list(product(enquiry_points, widths))
enquiry_points_list, widths_list = zip(*combinations)
enquiry_points_list = list(enquiry_points_list)
widths_list = list(widths_list)

P = RN(dimension=how_many_targets)
targets_dict = {}
for param, target_type in target_types.items():
    targets_dict[param] = []
    for i in range(how_many_targets):
        if target_type == Gaussian_1D: # MODIFY HERE THE SPECIAL TARGET AS WELL!!!!
            targets_dict[param].append(target_type(domain=domain,
                                                   center=enquiry_points_list[i],
                                                   width=widths_list[i]))
        else: 
            targets_dict[param].append(target_type(domain=domain))
constituent_mappings = [IntegralMapping(domain=models_dict[param], codomain=P, 
                                        kernels=targets_dict[param]) for param in physical_parameters]
T = DirectSumMapping(domain=M, codomain=P, mappings=tuple(constituent_mappings))


#################################
# Create fake true model and data
#################################
true_model = M.random_member(args_list=[(4,), (3,), (13,)])
data = G.map(true_model)


####################
# Compute norm bound
####################
# Edit region -------------
# Places where the true model will be evaluated
intervals = np.array([0,0.1, 0.25, 0.7, domain.bounds[0][1]])
# Edit region -------------
upper_bounds = []
for model in true_model:
    values = np.abs(model.evaluate(intervals[1:])[1])*1.5
    upper_bound = Piecewise_1D(domain=model.domain,
                               intervals=intervals,
                               values=values)
    upper_bounds.append(upper_bound)
norm_bound = M.norm(tuple(upper_bounds))

problem = Problem(M=M, D=D, P=P, G=G, T=T, norm_bound=norm_bound, data=data)

problem.solve()

problem._compute_least_norm_property()

# Compute True property
true_property = T.map(true_model)

""" problem.plot_multi_widths_errors(domain=domain, enquiry_points=enquiry_points,
                                 widths=widths, error_type='relative2') """

problem.plot_necessary_norm_bounds(relative_error=0.01, domain=domain,
                                   enquiry_points=enquiry_points, widths=widths)