import scipy
import numpy as np
import scipy.integrate
from sola.main_classes import functions


def integrate(function: functions.Function):
    return scipy.integrate.simpson(function.evaluate(function.domain.mesh),
                                   function.domain.mesh)
