from core.main_classes.domains import HyperParalelipiped
from core.main_classes.spaces import PCb, DirectSumSpace, RN
from core.aux.normal_data import load_normal_data
from core.main_classes.functions import *
from core.main_classes.mappings import *
import numpy as np


####################
# Create model space
####################
# Edit region -------------
physical_parameters = ['vs', 'vp', 'rho']
# Edit region -------------

no_of_params = len(physical_parameters)
EarthDomain = HyperParalelipiped(bounds=[[0, 6371]], fineness=1000)
constituent_models_spaces = [PCb(domain=EarthDomain) for _ in physical_parameters]
# Create a dictionary with physical_parameters as keys
models_dict = {param: model_space for param, model_space in zip(physical_parameters, constituent_models_spaces)}
M = DirectSumSpace(tuple(constituent_models_spaces))

###################
# Create Data space
###################
# Edit region -------------
data_directory = '/disks/data/PhD/BGSOLA/SOLA_DLI/kernels_modeplotaat_Adrian'
which_data = list(np.arange(0, 100))
# Edit region -------------

# Import sensitivity data
how_many_data = len(which_data)
raw_sensitivity_dict = {}
raw_sensitivity_domains_dict = {}
for param in physical_parameters:
    raw_sensitivity_domain, raw_sensitivity = load_normal_data(param, data_directory)
    raw_sensitivity_dict[param] = np.array(raw_sensitivity)[which_data]
    if param not in raw_sensitivity_domains_dict:
        raw_sensitivity_domains_dict[param] = raw_sensitivity_domain
D = RN(dimension=how_many_data)

###########################
# Create model-data mapping 
###########################
# Make them into functions via interpolation
sensitivity_dict = {}
for param in physical_parameters:
    sensitivity_dict[param] = []
    for discrete_sensitivity_kernel in raw_sensitivity_dict[param]:
        sensitivity_dict[param].append(Interpolation_1D(values=discrete_sensitivity_kernel,
                                                        raw_domain=raw_sensitivity_domains_dict[param],
                                                        domain=EarthDomain))
constituent_mappings = [IntegralMapping(domain=models_dict[param], codomain=D, 
                                        kernels=sensitivity_dict[param]) for param in physical_parameters]
mappings_dict = {param: mapping for param, mapping in zip(physical_parameters, constituent_mappings)}
G = DirectSumMapping(domain=M, codomain=D, mappings=tuple(constituent_mappings))

#########################
# Create property mapping
#########################
# Edit region -------------
target_types = {'vs': Gaussian_1D,
                'vp': Null_1D,
                'rho': Null_1D}
width = 1000 # same units as domain (km here)
how_many_targets = 10
enquiry_points = np.linspace(EarthDomain.bounds[0][0], 
                             EarthDomain.bounds[0][1], 
                             how_many_targets)
# Edit region -------------
targets_dict = {}
for param, target_type in target_types.items():
    targets_dict[param] = []
    for i in range(how_many_targets):
        if target_type == Gaussian_1D:
            targets_dict[param].append(target_type(domain=EarthDomain,
                                                   center=enquiry_points[i],
                                                   width=width))
        else: 
            targets_dict[param].append(target_type(domain=EarthDomain))