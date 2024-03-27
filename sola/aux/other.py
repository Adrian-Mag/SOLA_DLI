import numpy as np
from sola.main_classes.mappings import *

def round_to_sf(number, sf):
    if number == 0:
        return 0.0
    else:
        return round(number, -int(np.floor(np.log10(abs(number)))) + (sf - 1))
    
def simple_property(M: Space, P: Space, target_types: dict, domain: Domain, 
                    enquiry_points: list, widths: list, models_dict: dict):
    how_many_targets = len(enquiry_points)
    physical_parameters = list(target_types.keys())
    
    targets_dict = {}
    for param, target_type in target_types.items():
        targets_dict[param] = []
        for i in range(how_many_targets):
            if target_type != Null_1D: # MODIFY HERE THE SPECIAL TARGET AS WELL!!!!
                targets_dict[param].append(target_type(domain=domain,
                                                    center=enquiry_points[i],
                                                    width=widths[i]))
            else: 
                targets_dict[param].append(target_type(domain=domain))
    constituent_mappings = [IntegralMapping(domain=models_dict[param], codomain=P, 
                                            kernels=targets_dict[param]) for param in physical_parameters]
    return DirectSumMapping(domain=M, codomain=P, mappings=tuple(constituent_mappings))