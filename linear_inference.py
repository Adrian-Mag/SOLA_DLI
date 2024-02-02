from core.main_classes.domains import HyperParalelipiped
from core.main_classes.spaces import PCb, DirectSumSpace, RN
from core.aux.normal_data import load_normal_data
from core.main_classes.functions import *
from core.main_classes.mappings import *
from core.main_classes.SOLA_DLI import Problem
import numpy as np
import logging
import time

def log_and_time(section_name, start_time):
    elapsed_time = time.time() - start_time
    logging.info(f"{section_name}: Done. Time taken: {elapsed_time:.2f} seconds")


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

####################
# Create model space
####################
# Edit region -------------
physical_parameters = ['vs', 'vp', 'rho']
# Edit region -------------
start_time = time.time()

no_of_params = len(physical_parameters)
EarthDomain = HyperParalelipiped(bounds=[[0, 6371]], fineness=1000)
constituent_models_spaces = [PCb(domain=EarthDomain) for _ in physical_parameters]
# Create a dictionary with physical_parameters as keys
models_dict = {param: model_space for param, model_space in zip(physical_parameters, constituent_models_spaces)}
M = DirectSumSpace(tuple(constituent_models_spaces))
log_and_time('Create model space', start_time)

###################
# Create Data space
###################
# Edit region -------------
data_directory = '/disks/data/PhD/BGSOLA/SOLA_DLI/kernels_modeplotaat_Adrian'
which_data = list(np.arange(0, 100))
# Edit region -------------

start_time = time.time()
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
log_and_time('Created data space', start_time)

###########################
# Create model-data mapping 
###########################
start_time = time.time()
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
log_and_time('Created model-data mapping', start_time)


###################################
# Create property mapping and space
###################################
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
start_time = time.time()
P = RN(dimension=how_many_targets)
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
constituent_mappings = [IntegralMapping(domain=models_dict[param], codomain=P, 
                                        kernels=targets_dict[param]) for param in physical_parameters]
T = DirectSumMapping(domain=M, codomain=P, mappings=tuple(constituent_mappings))
log_and_time('Created property space and data-property mapping', start_time)

#################################
# Create fake true model and data
#################################
start_time = time.time()
true_model = M.random_member(args_list=[(1,), (2,), (3,)])
data = G.map(true_model)
log_and_time('Compute fake model and data', start_time)

####################
# Compute norm bound
####################
# Edit region -------------
# Places where the true model will be evaluated
intervals = np.array([0,1000, 2000, 5000, EarthDomain.bounds[0][1]])
# Edit region -------------
start_time = time.time()
upper_bounds = []
for model in true_model:
    values = model.evaluate(intervals[1:])[1]*1.2
    upper_bound = Piecewise_1D(domain=model.domain,
                               intervals=intervals,
                               values=values)
    upper_bounds.append(upper_bound)
norm_bound = M.norm(tuple(upper_bounds))
log_and_time('Compute fake model and data', start_time)

###################
# SOlve the Problem
###################
problem = Problem(M=M, D=D, P=P, G=G, T=T, norm_bound=norm_bound, data=data)
problem._compute_Lambda()
