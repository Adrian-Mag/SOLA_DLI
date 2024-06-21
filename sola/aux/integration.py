import scipy
import numpy as np
import scipy.integrate
from sola.main_classes import functions


def integrate(function: functions.Function, fineness):
    computation_domain = function.domain.dynamic_mesh(fineness)
    return scipy.integrate.simpson(function.evaluate(computation_domain),
                                   computation_domain)
