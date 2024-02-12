import numpy as np

def round_to_sf(number, sf):
    if number == 0:
        return 0.0
    else:
        return round(number, -int(np.floor(np.log10(abs(number)))) + (sf - 1))