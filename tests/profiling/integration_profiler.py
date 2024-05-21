from sola.main_classes import functions
from sola.main_classes import domains

domain = domains.HyperParalelipiped([[-1, 1]])

f = functions.Gaussian_1D(domain=domain, center=0, width=0.5)
g = functions.Moorlet_1D(domain=domain, center=.1, spread=0.5, frequency=1)
f.plot()
g.plot()
