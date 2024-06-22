from sola.main_classes import mappings
from sola.main_classes import spaces
from sola.main_classes import domains

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sola.aux.other import round_to_sf
import plotly.graph_objects as go
from itertools import product
from matplotlib.colors import LogNorm, Normalize  # Import LogNorm from colors module
import seaborn as sns
from matplotlib.ticker import LogFormatter
import numpy as np
import matplotlib


class DependencyTree:
    """
    A class representing a dependency tree.

    Attributes:
    - item_aliases (dict): A dictionary mapping item aliases to their original
    names.
    - aliases_item (dict): A dictionary mapping original item names to their
    aliases.
    - G (DiGraph): The directed graph representing dependencies between items.
    - start_node (str): The starting node for operations like finding reachable
    or dependent nodes.

    Methods:
    - plot_dependency_tree(): Plot the entire dependency tree and optionally
    save the plot.
    - find_reachable_nodes(start_node, plot_reachable_graph=False): Find and
    return all nodes reachable from the given start node.
      Optionally plot and save the reachable graph.
    - find_dependent_nodes(start_node, plot_dependent_tree=False): Find and
      return all nodes dependent on the given start node. Optionally plot and
      save the dependent tree.
    """

    def __init__(self):
        items = ['$\mathcal{M}$', '$\mathcal{D}$', '$\mathcal{P}$', 'G', 'T', # noqa
                 'd', '$\Lambda$', '$\Lambda^{-1}$', '$\Gamma$', # noqa
                 '$\Lambda^{-1}d$', '|$\widetilde{m}$|', '$\widetilde{m}$', # noqa
                 'M', '$\mathcal{H}_{ii}$', '$\chi_{ii}$', 'X', 'npf', # noqa
                 '$\epsilon_i$', '$\widetilde{p}$', 'sol', 'A', # noqa
                 '$\epsilon_r$', '$\epsilon_{r2}$'] # noqa

        aliases = ['M', 'D', 'P', 'G', 'T', 'd',
                   'Lambda', 'Lambda_inv', 'Gamma', 'sdata', 'least_norm',
                   'least_norm_solution', 'norm_bound', 'H_diag', 'chi_diag',
                   'X', 'npf', 'epsilon', 'least_norm_property', 'solution',
                   'A', 'relative_errors', 'relative_errors2']

        self.item_aliases = dict(zip(items, aliases))
        self.aliases_item = dict(zip(aliases, items))

        dependencies = [('G', 'Lambda'), ('Lambda', 'Lambda_inv'),
                        ('Lambda_inv', 'X'), ('T', 'Gamma'), ('G', 'Gamma'),
                        ('Gamma', 'X'), ('d', 'sdata'),
                        ('Lambda_inv', 'sdata'), ('sdata', 'least_norm'),
                        ('sdata', 'least_norm_solution'), ('M', 'G'),
                        ('T', 'chi_diag'), ('chi_diag', 'H_diag'), ('M', 'T'),
                        ('least_norm', 'npf'), ('norm_bound', 'npf'),
                        ('D', 'G'), ('npf', 'epsilon'), ('X', 'H_diag'),
                        ('D', 'd'), ('P', 'T'), ('G', 'D'), ('T', 'P'),
                        ('Gamma', 'H_diag'), ('X', 'least_norm_property'),
                        ('d', 'least_norm_property'), ('H_diag', 'epsilon'),
                        ('epsilon', 'solution'),
                        ('least_norm_property', 'solution'), ('X', 'A'),
                        ('G', 'A'), ('least_norm_property', 'relative_errors'),
                        ('epsilon', 'relative_errors'),
                        ('least_norm_property', 'relative_errors2'),
                        ('epsilon', 'relative_errors2')]

        self.G = nx.DiGraph()
        self.G.add_nodes_from(items)
        self.G.add_edges_from([(self.aliases_item[src], self.aliases_item[end])
                               for src, end in dependencies])
        self.start_node = None

    def plot_dependency_tree(self):
        """
        Plot the entire dependency tree and optionally save the plot.
        """
        pos = graphviz_layout(self.G, prog='dot')

        # Set node colors and edge colors
        node_colors = []
        for node in self.G.nodes:
            if node == self.start_node:
                node_colors.append('gold')  # Color for the starting node
            elif node in self._alias_to_item(['norm_bound', 'M', 'D',
                                              'G', 'T', 'P', 'd']):
                node_colors.append('red')
            elif node in self._alias_to_item(['least_norm_property',
                                              'least_norm_solution',
                                              'solution', 'A', 'epsilon',
                                              'relative_errors',
                                              'relative_errors2']):
                node_colors.append('green')
            else:
                node_colors.append('skyblue')

        # Draw nodes with different colors
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(self.G, pos, node_size=700,
                               node_color=node_colors, edgecolors='black',
                               linewidths=1, alpha=0.8)

        # Draw edges and labels
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', arrowsize=20,
                               connectionstyle='arc3,rad=0.1', width=1.0)
        nx.draw_networkx_labels(self.G, pos, font_weight='bold',
                                font_color='black', font_size=10)

        plt.show()

    def find_reachable_nodes(self, start_nodes, plot_reachable_graph=False):
        """
        Find and return all nodes reachable from the given start nodes.
        Optionally plot and save the reachable graph.

        Parameters:
        - start_nodes (list): The list of starting nodes.
        - plot_reachable_graph (bool): Whether to plot and save the reachable
        graph. Default is False.

        Returns:
        - set: A set of reachable nodes.
        """
        self.start_node = [self.aliases_item[start_node]
                           for start_node in start_nodes]
        reachable_nodes = set()

        def dfs(node):
            reachable_nodes.add(node)
            for neighbor in self.G.neighbors(node):
                if neighbor not in reachable_nodes:
                    dfs(neighbor)

        for start_node in self.start_node:
            dfs(start_node)

        if plot_reachable_graph:
            reachable_graph = self.G.subgraph(reachable_nodes)
            pos = graphviz_layout(reachable_graph, prog='dot')

            # Set node colors and edge colors
            reachable_node_colors = []
            for node in reachable_graph.nodes:
                if node in self.start_node:
                    reachable_node_colors.append('gold')  # Color for the starting node # noqa
                elif node in self._alias_to_item(['norm_bound', 'M', 'D',
                                                  'G', 'T', 'P', 'd']):
                    reachable_node_colors.append('red')
                elif node in self._alias_to_item(['P', 'M', 'relative_errors',
                                                  'solution', 'A', 'epsilon',
                                                  'relative_errors2']):
                    reachable_node_colors.append('green')
                else:
                    reachable_node_colors.append('skyblue')

            # Draw nodes with different colors
            plt.figure(figsize=(12, 8))
            nx.draw_networkx_nodes(reachable_graph, pos, node_size=700,
                                   node_color=reachable_node_colors,
                                   edgecolors='black',
                                   linewidths=1, alpha=0.8)

            # Draw edges and labels
            nx.draw_networkx_edges(reachable_graph, pos, edge_color='gray',
                                   arrowsize=20, width=1.0,
                                   connectionstyle='arc3,rad=0.1')

            nx.draw_networkx_labels(reachable_graph, pos, font_weight='bold',
                                    font_color='black', font_size=10)

            plt.show()

        return set(self._item_to_alias(reachable_nodes))

    def _alias_to_item(self, aliases: list):
        # Given some aliases it returns the corresponding items
        return [self.aliases_item[alias] for alias in aliases]

    def _item_to_alias(self, items: list):
        # Given some items it returns the corresponding aliases
        return [self.item_aliases[item] for item in items]

    def find_dependent_nodes(self, start_node, plot_dependent_tree=False):
        """
        Find and return all nodes dependent on the given start node.
        Optionally plot and save the dependent tree.

        Parameters:
        - start_node (str): The starting node.
        - plot_dependent_tree (bool): Whether to plot and save the dependent
        tree. Default is False.

        Returns:
        - set: A set of dependent nodes.
        """
        self.start_node = self.aliases_item[start_node]
        dependent_nodes = set()

        # These items cannot be turned to None
        def dfs(node):
            dependent_nodes.add(node)
            for predecessor in self.G.predecessors(node):
                if predecessor not in dependent_nodes:
                    dfs(predecessor)

        dfs(self.aliases_item[start_node])

        if plot_dependent_tree:
            dependent_tree = self.G.subgraph(dependent_nodes)
            pos = graphviz_layout(dependent_tree, prog='dot')

            # Set node colors and edge colors
            dependent_node_colors = []
            for node in dependent_tree.nodes:
                if node == self.start_node:
                    dependent_node_colors.append('gold')  # Color for the starting node # noqa
                elif node in self._alias_to_item(['norm_bound', 'M', 'D',
                                                  'G', 'T', 'P', 'd']):
                    dependent_node_colors.append('red')
                elif node in self._alias_to_item(['P', 'M', 'relative_errors',
                                                  'solution', 'A', 'epsilon',
                                                  'relative_errors2']):
                    dependent_node_colors.append('green')
                else:
                    dependent_node_colors.append('skyblue')

            # Draw nodes with different colors
            plt.figure(figsize=(12, 8))
            nx.draw_networkx_nodes(dependent_tree, pos, node_size=700,
                                   node_color=dependent_node_colors,
                                   edgecolors='black',
                                   linewidths=1, alpha=0.8)

            # Draw edges and labels
            nx.draw_networkx_edges(dependent_tree, pos, edge_color='gray',
                                   arrowsize=20, width=1.0,
                                   connectionstyle='arc3,rad=0.1')

            nx.draw_networkx_labels(dependent_tree, pos, font_weight='bold',
                                    font_color='black', font_size=10)

            plt.show()

        return dependent_nodes


class Problem():
    def __init__(self, M: spaces.Space, D: spaces.Space, P: spaces.Space,
                 G: mappings.Mapping, T: mappings.Mapping,
                 norm_bound: float = None, data: np.ndarray = None) -> None:
        """
        Class representing the SOLA problem.

        Parameters:
        - M (Space): Model space.
        - D (Space): Data space.
        - P (Space): Property space.
        - G (Mapping): Model-data mapping.
        - T (Mapping): Model-property mapping.
        - norm_bound (float): Model norm bound.
        - data (np.ndarray, optional): Input data. Defaults to None.
        """

        self.M = M  # model space
        self.D = D  # data space
        self.P = P  # property space
        self.G = G  # model-data mapping
        self.T = T  # model-property mapping
        self.data = data  # data
        self.norm_bound = norm_bound  # model norm bound

        self.G_adjoint = G.adjoint
        self.T_adjoint = T.adjoint

        self.Lambda = None
        self.Lambda_inv = None
        self.Gamma = None
        self.sdata = None
        self.least_norm = None
        self.least_norm_solution = None
        self.X = None
        self.npf = None
        self.chi_diag = None
        self.epsilon = None
        self.least_norm_solution = None
        self.least_norm_property = None
        self.solution = None
        self.A = None
        self.H_diag = None
        self.relative_errors = None
        self.relative_errors2 = None

        self.fixed_items = ['M', 'D', 'P', 'G', 'T', 'd', 'norm_bound']
        self.dependencies = DependencyTree()

    def change_M(self, new_M: spaces.Space, new_G: mappings.Mapping,
                 new_T: mappings.Mapping, new_norm_bound: float):
        """
        Change the model space, data mapping, property mappin, and norm bound.

        Parameters:
        - new_M (Space): New model space.
        - new_G (Mapping): New data mapping.
        - new_T (Mapping): New property mapping.
        - new_norm_bound (float): New model norm bound.
        """
        # This deals mostly with the case when I have a DirectSum space and I
        # want to remove or add some spaces to it
        self.M = new_M
        self.G = new_G
        self.T = new_T
        self.norm_bound = new_norm_bound
        self.G_adjoint = new_G.adjoint
        self.T_adjoint = new_T.adjoint

        dependent_nodes = self.dependencies.find_reachable_nodes(['M', 'G', 'T']) - set(self.fixed_items) # noqa
        for alias in dependent_nodes:
            setattr(self, alias, None)

    def change_D(self, new_D: spaces.Space, new_G: mappings.Mapping,
                 new_data: np.ndarray):
        """
        Change the data space, data mapping, and input data.

        Parameters:
        - new_D (Space): New data space.
        - new_G (Mapping): New data mapping.
        - new_data (np.ndarray): New input data.
        """
        # Here I consider the case when the new data space is just R^N+n in
        # which case the data vector must be updated and we must also include
        # the new sensitivity kernels involved
        self.D = new_D
        self.G = new_G
        self.data = new_data

        dependent_nodes = self.dependencies.find_reachable_nodes(['D', 'G', 'd']) - set(self.fixed_items) # noqa
        for alias in dependent_nodes:
            setattr(self, alias, None)

    def change_P(self, new_P: spaces.Space, new_T: mappings.Mapping):
        """Change the property space and property mapping

        Args:
            new_P (Space): New property space
            new_T (Mapping): New property mapping
        """
        # Here I deal with the case when I want to introduce or remove a target
        # kernel
        self.P = new_P
        self.T = new_T

        dependent_nodes = self.dependencies.find_reachable_nodes(['P', 'T']) - set(self.fixed_items) # noqa
        for alias in dependent_nodes:
            setattr(self, alias, None)

    def change_T(self, new_T: mappings.Mapping, new_P: spaces.Space = None):
        """Change the property mapping and property space (optional)

        Args:
            new_T (Mapping): New property mapping
            new_P (Space, optional): New property space. Defaults to None.
        """
        # This is specifically for the case when I want to change a target
        # kernel, but I don't change the number of kernels
        if new_P is None:
            self.T = new_T

            dependent_nodes = self.dependencies.find_reachable_nodes(['T']) - set(self.fixed_items) # noqa
            for alias in dependent_nodes:
                setattr(self, alias, None)
        else:
            self.T = new_T
            self.P = new_P

            dependent_nodes = self.dependencies.find_reachable_nodes(['T', 'P']) - set(self.fixed_items) # noqa
            for alias in dependent_nodes:
                setattr(self, alias, None)

    def change_G(self, new_G: mappings.Mapping, new_D: spaces.Space = None):
        """Change data mapping and data space (optional)

        Args:
            new_G (Mapping): New data mapping
            new_D (Space, optional): New data space. Defaults to None.
        """
        if new_G is None:
            self.T = new_D

            dependent_nodes = self.dependencies.find_reachable_nodes(['T']) - set(self.fixed_items) # noqa
            for alias in dependent_nodes:
                setattr(self, alias, None)
        else:
            self.T = new_D
            self.P = new_G

            dependent_nodes = self.dependencies.find_reachable_nodes(['T', 'G']) - set(self.fixed_items) # noqa
            for alias in dependent_nodes:
                setattr(self, alias, None)

    def change_d(self, new_data: np.ndarray):
        """Change data

        Args:
            new_data (np.ndarray): new data
        """
        # This is specifically for the case when the data space remains
        # unchanged
        self.data = new_data

        dependent_nodes = self.dependencies.find_reachable_nodes(['d']) - set(self.fixed_items) # noqa
        for alias in dependent_nodes:
            setattr(self, alias, None)

    def change_bound(self, new_bound: float):
        """Change the Norm Bound

        Args:
            new_bound (float): New norm bound
        """
        self.norm_bound = new_bound

        dependent_nodes = self.dependencies.find_reachable_nodes(['norm_bound']) - set(self.fixed_items) # noqa
        for alias in dependent_nodes:
            setattr(self, alias, None)

    def _compute_Lambda(self):
        self.Lambda = self.G._compute_GramMatrix()

    def _compute_Lambda_inv(self):
        if self.Lambda is None:
            self._compute_Lambda()
        self.Lambda_inv = self.Lambda.invert()

    def _compute_sdata(self):
        if self.data is None:
            raise TypeError('The current problem does not have any data.'
                            'Please add data')
        if self.Lambda_inv is None:
            self._compute_Lambda_inv()
        self.sdata = self.Lambda_inv.map(self.data)

    def _compute_least_norm(self):
        if self.sdata is None:
            self._compute_sdata()
        self.least_norm = np.sqrt(self.D.inner_product(self.data, self.sdata))

    def _compute_norm_prefactor(self):
        if self.least_norm is None:
            self._compute_least_norm()
        if self.norm_bound is None:
            raise ValueError('Norm bouns is None. '
                             'Add a norm bound using change_bound() method.')
        self.npf = np.sqrt(self.norm_bound**2 - self.least_norm**2)

    def _compute_least_norm_solution(self):
        if self.sdata is None:
            self._compute_sdata()
        self.least_norm_solution = self.G_adjoint.map(self.sdata)

    def _compute_Gamma(self):
        self.Gamma = self.T * self.G_adjoint

    def _compute_X(self):
        if self.Gamma is None:
            self._compute_Gamma()
        if self.Lambda_inv is None:
            self._compute_Lambda_inv()
        self.X = self.Gamma * self.Lambda_inv

    def _compute_chi_diag(self):
        self.chi_diag = self.T._compute_GramMatrix_diag()

    def _compute_H_diag(self):
        if self.chi_diag is None:
            self._compute_chi_diag()
        if self.X is None:
            self._compute_X()
        if self.Gamma is None:
            self._compute_Gamma()
        self.H_diag = self.chi_diag - \
            np.sum(self.X.matrix * self.Gamma.matrix,
                   axis=1).reshape(self.chi_diag.shape)

    def _compute_epsilon(self):
        if self.npf is None:
            self._compute_norm_prefactor()
        if self.H_diag is None:
            self._compute_H_diag()
        try:
            self.epsilon = self.npf * np.sqrt(self.H_diag)
        except RuntimeWarning:
            print('Error: The norm bound is too small. ')

    def _compute_least_norm_model_solution(self):
        if self.sdata is None:
            self._compute_sdata()
        self.least_norm_model_solution = self.G_adjoint.map(self.sdata)

    def _compute_least_norm_property(self):
        if self.data is None:
            raise TypeError('The current problem does not have any data.'
                            ' Please add data')
        if self.X is None:
            self._compute_X()
        self.least_norm_property = self.X.map(self.data)

    def solve(self):
        if self.least_norm_property is None:
            self._compute_least_norm_property()
        if self.epsilon is None:
            self._compute_epsilon()
        self.solution = {'upper bound': self.least_norm_property + self.epsilon, # noqa
                         'lower bound': self.least_norm_property - self.epsilon} # noqa

    def _compute_resolving_kernels(self):
        if self.X is None:
            self._compute_X()
        self.A = self.X * self.G

    def _compute_relative_errors(self):
        if self.epsilon is None:
            self._compute_epsilon()
        if self.least_norm_property is None:
            self._compute_least_norm_property()
        self.relative_errors = 100 * self.epsilon / self.least_norm_property

    def _compute_relative_errors2(self):
        if self.epsilon is None:
            self._compute_epsilon()
        if self.least_norm_property is None:
            self._compute_least_norm_property()
        property_range = (np.max(self.least_norm_property) - # noqa
                          np.min(self.least_norm_property))
        self.relative_errors2 = 100 * self.epsilon / property_range

    def _compute_resoltion_misfit(self):
        resolution_misfit = np.zeros(self.P.dimension)
        if self.H_diag is None:
            self._compute_H_diag()
        for index, target in enumerate(self.T.kernels):
            resolution_misfit[index] = np.sqrt(self.H_diag[index]) / self.M.norm(target) # noqa

        return resolution_misfit

    def plot_solution(self, enquiry_points):
        # Will plot the property bounds, the least norm property, the resolving
        # kernels and the target kernels with a slider used to explore them. It
        # assumes that the problem is 1D and that the property vector contains
        # some property of the true model evaluated at only one position

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.5, subplot_titles=('A', 'B'))
        no_of_traces = 0  # by default

        for step in range(self.P.dimension):
            # Plot Least norm property
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color='#FFA500', width=4),
                    name='Property',
                    x=enquiry_points,
                    y=self.least_norm_property
                ),
                row=1, col=1
            )
            if step == 0:
                no_of_traces += 1
            # Resolving kernels
            for map in self.A.mappings:
                fig.add_trace(
                    go.Scatter(
                        visible=False,
                        line=dict(color='#DF0000', width=4),
                        name='Resolving Kernel: ' + str(round_to_sf(enquiry_points[step], 2)), # noqa
                        x=map.kernels[step].domain.mesh,
                        y=map.kernels[step].evaluate(map.kernels[step].domain.mesh) # noqa
                    ),
                    row=2, col=1
                )
                if step == 0:
                    no_of_traces += 1

        fig.update_xaxes(title_text='Domain', row=2, col=1)

        fig.data[0].visible = True

        steps = []

        for i in range(self.P.dimension):
            step = dict(
                method='update',
                args=[{'visible': [False] * (no_of_traces * self.P.dimension + 1)}, # noqa
                      {'title': 'Slider'}],
            )
            for j in range(no_of_traces):
                step['args'][0]['visible'][no_of_traces * i + j] = True

            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={'prefix': 'Frequency'},
            pad={'t': 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            title='Subplots with slider'
        )

        fig.show()

    def _plot_on_enquirypts_x_widths(self, target_parameter_1: np.ndarray,
                                     target_parameter_2: np.ndarray,
                                     quantity: np.ndarray,
                                     uninterpretable_region: np.ndarray,
                                     ticks: list,
                                     xticks, yticks,
                                     xlabel, ylabel, title, plot_colors,
                                     physical_parameters_symbols: list,
                                     cmap: str = None, norm=None,
                                     colorbar_format=None,
                                     fill_betweenx_calls=None, args_list=None):
        """
        Plots a quantity evaluated on a grid defined by two target parameters. This method also overlays information
        about target and resolving kernels at each grid point.

        This method is designed to provide a comprehensive visualization of how a certain quantity varies across a
        two-dimensional parameter space, with additional insights into the resolution capabilities of the analysis
        through the display of target and resolving kernels.

        Parameters:
        - target_parameter_1 (np.ndarray): The first target parameter, defining the x-axis of the grid.
        - target_parameter_2 (np.ndarray): The second target parameter, defining the y-axis of the grid.
        - quantity (np.ndarray): The quantity to be plotted over the parameter grid.
        - uninterpretable_region (np.ndarray): A boolean array indicating grid points that are not interpretable or valid.
        - ticks (list): Tick values for the colorbar.
        - xticks: Tick labels for the x-axis.
        - yticks: Tick labels for the y-axis.
        - xlabel: Label for the x-axis.
        - ylabel: Label for the y-axis.
        - title: Title of the plot.
        - plot_colors (list): Colors to be used for plotting the target and resolving kernels.
        - physical_parameters_symbols (list): Symbols representing physical parameters, used in the legend.
        - cmap (str, optional): Colormap for the quantity plot.
        - norm (optional): A matplotlib.colors.Normalize instance to scale luminance data.
        - colorbar_format (optional): Format specification for the colorbar labels.
        - fill_betweenx_calls (optional): List of functions to fill between for highlighting specific regions.
        - args_list (optional): List of arguments for each fill_betweenx call.

        Returns:
        - An interactive plot displaying the specified quantity over the parameter grid, with additional features
        such as colorbar, exclusion zones, and interactive kernel analysis on click.
        """

        N_target_parameter_1 = len(target_parameter_1)
        N_target_parameter_2 = len(target_parameter_2)

        # Plotting
        matplotlib.rcParams['hatch.linewidth'] = 5.0
        fig = plt.figure(figsize=(9, 8))
        # Set excluded areas to NaN to make them transparent
        quantity[uninterpretable_region] = np.nan
        # Plot the quantity
        plt.imshow(quantity, norm=norm, cmap=cmap)
        # Add colorbar
        plt.colorbar(ticks=ticks,
                     format=colorbar_format,
                     shrink=0.7)

        # Overlay exclusion zones with diagonal stripes in gray
        for i in range(N_target_parameter_2):
            for j in range(N_target_parameter_1):
                if uninterpretable_region[i, j]:
                    plt.fill_betweenx([i - 0.5, i + 0.5], j - 0.5, j + 0.5,
                                      color='gray', edgecolor='gray',
                                      hatch='/', alpha=0.3)

        # Adjust other plot settings
        plt.yticks(np.arange(0, len(target_parameter_1),
                             int(len(target_parameter_1) / 10) + 1),
                   yticks, rotation=40, fontsize=12)
        plt.xticks(np.arange(0, len(target_parameter_2),
                             int(len(target_parameter_2) / 10) + 1),
                   xticks, rotation=20, fontsize=16)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.title(title, fontsize=26)
        plt.grid(False)

        # Set the colors of the pixel boxes (if you select more pixels, the
        # colors will cycle)
        highlight_colors = ['#ee9617', '#f2ef0c', '#ee1717',
                            'black', '#f20cd6']
        # Store highlighted pixel coordinates globally
        highlighted_pixels = {}
        # Store rectanle objects
        highlighted_rects = {}
        # Store target kernel figures
        figures = {}

        def snap_to_pixel(x, step):
            return int(round(x / step))

        def highlight_pixel(fig_id, i, j):
            if fig_id in highlighted_pixels:
                highlighted_rects[fig_id].remove()
            rect = plt.Rectangle((i - .5, j - .5), 1, 1, linewidth=2,
                                 edgecolor=highlight_colors[fig_id % len(highlight_colors)], # noqa
                                 facecolor='None')
            fig.gca().add_patch(rect)
            highlighted_rects[fig_id] = rect
            highlighted_pixels[fig_id] = (i, j)
            fig.canvas.draw()

        def onclose(event):
            fig_ids = list(figures.keys())  # Create a copy of dictionary keys
            for fig_id in fig_ids:
                fig = figures[fig_id]
                if event.canvas == fig.canvas:
                    del figures[fig_id]
                    if fig_id in highlighted_rects:
                        highlighted_rects[fig_id].remove()
                        del highlighted_rects[fig_id]
                        del highlighted_pixels[fig_id]

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                i = snap_to_pixel(event.xdata, 1)
                j = snap_to_pixel(event.ydata, 1)
                figure = plt.figure(figsize=(10, 8))
                figure.canvas.mpl_connect('close_event', onclose)
                figures[figure.number] = figure
                sns.set_theme(style='white')
                sns.set_palette('YlGnBu')
                plt.title('Target vs Resolving kernels', fontsize=25)
                all_y_values = []
                for index, (target_mapping, resolving_mapping) in enumerate(zip(self.T.mappings, self.A.mappings)): # noqa
                    resolving_kernel = resolving_mapping.kernels[j + N_target_parameter_1 * i] # noqa
                    resolving_kernel_y_values = resolving_kernel.evaluate(resolving_kernel.domain.mesh) # noqa
                    plt.plot(resolving_kernel.domain.mesh,
                             resolving_kernel_y_values,
                             label='Resolving: ' + physical_parameters_symbols[index], # noqa
                             linewidth=2, color=plot_colors[index])
                    all_y_values.extend(resolving_kernel_y_values)
                    target_kernel = target_mapping.kernels[j + N_target_parameter_1 * i] # noqa
                    target_kernel_y_values = target_kernel.evaluate(target_kernel.domain.mesh) # noqa
                    plt.plot(target_kernel.domain.mesh, target_kernel_y_values,
                             label='Target: ' + physical_parameters_symbols[index], # noqa
                             linewidth=2, color=plot_colors[index], linestyle='dashed') # noqa
                    all_y_values.extend(target_kernel_y_values)
                y_min = min(all_y_values) * 1.2
                y_max = max(all_y_values) * 1.2

                y_min = min([-(y_max - y_min) * 0.1, y_min])
                if fill_betweenx_calls is not None:
                    for fill_betweenx_call, args in zip(fill_betweenx_calls, args_list): # noqa
                        fill_betweenx_call(*args)
                plt.xlim([target_kernel.domain.bounds[0][0],
                          target_kernel.domain.bounds[0][1]])
                plt.ylim([y_min, y_max])
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel('Enquiry Points', fontsize=20)
                plt.ylabel('Kernel Value', fontsize=20)
                plt.legend(fontsize=15)
                plt.tight_layout()
                plt.show()
                highlight_pixel(figure.number, i, j)

        # Plot the first figure
        plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        fig.tight_layout()
        plt.show()

    def a(self):
        pass

    def resolution_analysis(self, enquiry_points: np.ndarray, widths:np.ndarray,
                            error_type: str, domain: domains.Domain,
                            physical_parameters_symbols: dict,
                            ticks: list=None, norm_type: str='log', plot_range: list=None):
        """
        Performs a resolution analysis based on the provided enquiry points and widths, and visualizes the results.

        This method is designed to assess and visualize the resolution quality of a geophysical inversion process
        by analyzing the resolving kernels computed for a set of enquiry points and widths. It supports different
        types of error analysis including absolute errors, relative errors, and resolution misfit.

        Parameters:
        - enquiry_points (np.ndarray): Array of enquiry points at which the resolution is evaluated.
        - widths (np.ndarray): Array of widths defining the scale of analysis around each enquiry point.
        - error_type (str): Type of error to compute and visualize. Supported values are 'absolute', 'relative',
        and 'resolution_misfit'.
        - domain (domains.Domain): The domain over which the analysis is performed, providing context such as
        bounds and discretization.
        - physical_parameters_symbols (dict): Dictionary mapping physical parameters to their symbols for
        visualization purposes.
        - ticks (list, optional): Tick values for the colorbar. If None, default ticks are used.
        - norm_type (str, optional): Normalization type for the error visualization. Supported values are 'log'
        and others as per matplotlib norms.
        - plot_range (list, optional): The range of values to plot. If None, the range is determined based on
        the errors.

        Returns:
        - A plot visualizing the resolution analysis results, including error distributions across the enquiry
        points and widths, with options for interactive exploration of the resolving kernels.
        """
        N_enquiry_points = len(enquiry_points)
        N_widths = len(widths)

        # Ensure necessary computations are done
        if self.A is None:
            self._compute_resolving_kernels()

        if error_type == 'absolute':
            # Compute absolute errors
            if self.epsilon is None:
                self._compute_epsilon()
            errors = self.epsilon
        elif error_type == 'relative':
            # Compute modified relative errors
            if self.relative_errors2 is None:
                self._compute_relative_errors2()
            errors = self.relative_errors2
        elif error_type == 'resolution_misfit':
            # Compute resolution misfit
            errors = self._compute_resoltion_misfit()
        else:
            raise ValueError('Error type must be absolute, '
                             'relative, relative2 or resolution_misfit')

        # Compute exclusion zone
        domain_min, domain_max = domain.bounds[0]
        combinations = list(product(enquiry_points, widths))
        exclusion_zones = np.array([((center + spread / 2) > domain_max) or # noqa
                                    ((center - spread / 2) < domain_min) for
                                    center, spread in combinations])
        exclusion_map = (exclusion_zones.reshape(N_enquiry_points, N_widths)).T

        # Compute errors map
        errors_map = (errors.reshape(N_enquiry_points, N_widths)).T


        # Set figure parameters
        if error_type == 'absolute':
            title = 'Absolute Error Bounds'
            max_error = np.max(np.abs(errors))
            if norm_type == 'log':
                if plot_range is None:
                    plot_range = [max_error*1e-3, max_error]
                norm = LogNorm(vmin=plot_range[0], vmax=plot_range[1])
                if ticks is None:
                    ticks = np.logspace(plot_range[0], plot_range[1], 5)
            elif norm_type == 'linear':
                if plot_range is None:
                    plot_range = [0, max_error]
                norm = Normalize(vmin=plot_range[0], vmax=plot_range[1])
                if ticks is None:
                    ticks = np.linspace(plot_range[0], plot_range[1], 5)
        elif error_type == 'relative':
            title = 'Relative Error Bounds'
            if plot_range is None:
                plot_range = [1, 1000]
            norm = LogNorm(vmin=plot_range[0], vmax=plot_range[1])
            if ticks is None:
                ticks = [1, 10, 100, 1000]
        elif error_type == 'resolution_misfit':
            title = 'Resolution Misfit'
            if plot_range is None:
                plot_range = [0.001, 1]
            norm = LogNorm(vmin=plot_range[0], vmax=plot_range[1])
            if ticks is None:
                ticks = [0.01, 0.1, 1]

        xticks = ['{:.2}'.format(point) for point in
                    enquiry_points[::int(len(enquiry_points) / 10) + 1]]
        yticks = ['{:.0%}'.format(spread / (domain_max - domain_min)) for
                    spread in widths[::int(len(widths) / 10) + 1]]
        colors = sns.color_palette('YlGnBu', n_colors=100)

        # Plot the errors map (this gives an interactive plot)
        self._plot_on_enquirypts_x_widths(target_parameter_1=enquiry_points,
                                          target_parameter_2=widths,
                                          quantity=errors_map,
                                          uninterpretable_region=exclusion_map,
                                          ticks=ticks,
                                          xticks=xticks,
                                          yticks=yticks,
                                          xlabel='Enquiry Points',
                                          ylabel='Width',
                                          title=title,
                                          plot_colors=['#5ee22d', colors[99], '#fccd1a'], # noqa
                                          cmap='Blues_r',
                                          norm=norm,
                                          physical_parameters_symbols=list(physical_parameters_symbols.values()), # noqa
                                          colorbar_format=LogFormatter(10, labelOnlyBase=False)) # noqa

    def plot_necessary_norm_bounds(self, relative_error: float,
                                   domain: domains.Domain,
                                   enquiry_points, widths,
                                   physical_parameters_symbols):
        """Plots the necessary norm bound necesary to achieve the
        desired relative error as a multiple of the least norm.

        Args:
            relative_error (float): desired relative error
        """
        if self.least_norm_property is None:
            self._compute_least_norm_property()
        if self.least_norm is None:
            self._compute_least_norm()
        if self.H_diag is None:
            self._compute_H_diag()
        if self.A is None:
            self._compute_resolving_kernels()

        # Compute exclusion zone
        domain_min, domain_max = domain.bounds[0]
        N_enquiry_points = len(enquiry_points)
        N_widths = len(widths)
        combinations = list(product(enquiry_points, widths))
        exclusion_zones = np.array([((center + spread / 2) > domain_max) or # noqa
                                    ((center - spread / 2) < domain_min) for
                                    center, spread in combinations])
        exclusion_map = (exclusion_zones.reshape(N_enquiry_points, N_widths)).T

        alpha = np.sqrt((relative_error * np.ptp(self.least_norm_property) / # noqa
                         self.least_norm)**2 / self.H_diag + 1)
        alpha = (alpha.reshape(N_enquiry_points, N_widths)).T

        xticks = ['{:.2}'.format(point) for point in
                  enquiry_points[::int(len(enquiry_points) / 10) + 1]]
        yticks = ['{:.0%}'.format(spread / (domain_max - domain_min)) for
                  spread in widths[::int(len(widths) / 10) + 1]]
        colors = sns.color_palette('YlGnBu', n_colors=100)

        self._plot_on_enquirypts_x_widths(target_parameter_1=enquiry_points,
                                          target_parameter_2=widths,
                                          quantity=alpha,
                                          uninterpretable_region=exclusion_map,
                                          ticks=[1, 10, 100, 1e3],
                                          colorbar_label='Alpha',
                                          xticks=xticks,
                                          yticks=yticks,
                                          xlabel='Enquiry Points',
                                          ylabel='Widths', title='Alpha',
                                          plot_colors=['#5ee22d', colors[99], '#fccd1a'], # noqa
                                          cmap='Blues_r',
                                          norm=LogNorm(vmin=1, vmax=1e3),
                                          colorbar_format=None,
                                          physical_parameters_symbols=physical_parameters_symbols) # noqa