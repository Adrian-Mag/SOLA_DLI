import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sola.main_classes.domains import HyperParalelipiped
from sola.main_classes.functions import Interpolation_1D
matplotlib.use('tkagg')


def as_function(values: np.ndarray, domain: np.ndarray) -> callable:
    if np.array_equal(values.shape, domain.shape):
        return interp1d(domain, values, kind='linear',
                        fill_value='extrapolate')


class FunctionDrawer:
    def __init__(self, domain: HyperParalelipiped, min_y: float, max_y: float):
        self.points = []
        self.domain = domain
        self.min_y, self.max_y = min_y, max_y
        self.drawing = False  # Track whether the mouse button is pressed

    def draw_function(self):
        self.points = []
        length = self.domain.total_measure

        # Prepare the plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(self.domain.bounds[0][0] - length * 0.1,
                         self.domain.bounds[0][1] + length * 0.1)
        self.ax.set_ylim(self.min_y, self.max_y)
        self.ax.set_title('Draw your function')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('close_event', self.process_points)
        plt.show()

    def process_points(self, event):
        # Filter out points outside x_domain
        self.points = [(x, y) for x, y in self.points if
                       x >= self.domain.bounds[0][0] and # noqa
                       x <= self.domain.bounds[0][1]]
        self.raw_domain, self.values = zip(*self.points)

        # Find unique elements in self.raw_domain and get corresponding values
        unique_raw_domain, indices = np.unique(np.array(self.raw_domain),
                                               return_index=True)
        unique_values = np.array(self.values)[indices]

        self.raw_domain = unique_raw_domain.copy()
        self.values = unique_values.copy()

    def on_click(self, event):
        if event.button == 1:  # Check if the left mouse button is pressed
            self.drawing = True  # Start drawing when left click is pressed

    def on_motion(self, event):
        if self.drawing and event.xdata is not None and event.ydata is not None: # noqa
            self.points.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, 'ro')
            if len(self.points) > 1:
                self.ax.plot([self.points[-2][0], event.xdata],
                             [self.points[-2][1], event.ydata], 'b-')
            self.fig.canvas.draw()

    def on_release(self, event):
        if event.button == 1:  # Check if the left mouse button is released
            self.drawing = False  # Stop drawing when left click is released

    def plot_function(self):
        if self.points is not None:
            plt.plot(self.raw_domain, self.values)
            plt.title('Function')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show(block=True)

    def save_function(self, name):
        with open(name + '_function.txt', 'w') as file:
            for point in self.points:
                file.write(f'{point[0]},{point[1]}\n')

    def open_function(self, filename):
        with open(filename, 'r') as file:
            self.points = [tuple(map(float, line.strip().split(','))) for line in file] # noqa

        self.raw_domain, self.values = zip(*self.points)
        self.raw_domain, self.values = np.array(self.domain), np.array(self.values) # noqa

    def interpolate_function(self):
        return Interpolation_1D(raw_domain=self.raw_domain,
                                values=self.values, domain=self.domain)
