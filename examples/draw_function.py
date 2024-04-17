from sola.main_classes.domains import HyperParalelipiped
from sola.main_classes.spaces import PCb

domain = HyperParalelipiped([[0, 1]])
space = PCb(domain)

""" draw_member will create a Interpolation_1D function from the points drawn
by the user."""
member = space.draw_member(min_y=-1, max_y=1)
member.plot()
