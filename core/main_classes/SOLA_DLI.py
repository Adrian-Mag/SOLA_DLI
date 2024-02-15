from core.main_classes.spaces import PCb, DirectSumSpace, RN
from core.main_classes.mappings import *
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from core.aux.other import round_to_sf
import plotly.graph_objects as go
from itertools import product
from matplotlib.colors import LogNorm  # Import LogNorm from colors module
import seaborn as sns
from matplotlib.ticker import LogFormatter 


class DependencyTree:
    """
    A class representing a dependency tree.

    Attributes:
    - item_aliases (dict): A dictionary mapping item aliases to their original names.
    - aliases_item (dict): A dictionary mapping original item names to their aliases.
    - G (DiGraph): The directed graph representing dependencies between items.
    - start_node (str): The starting node for operations like finding reachable or dependent nodes.

    Methods:
    - plot_dependency_tree(): Plot the entire dependency tree and optionally save the plot.
    - find_reachable_nodes(start_node, plot_reachable_graph=False): Find and return all nodes reachable from the given start node.
      Optionally plot and save the reachable graph.
    - find_dependent_nodes(start_node, plot_dependent_tree=False): Find and return all nodes dependent on the given start node.
      Optionally plot and save the dependent tree.
    """

    def __init__(self):
        items = ['$\mathcal{M}$', '$\mathcal{D}$', '$\mathcal{P}$', 'G', 'T', 'd',
                 '$\Lambda$', '$\Lambda^{-1}$', '$\Gamma$', '$\Lambda^{-1}d$',
                 '|$\widetilde{m}$|', '$\widetilde{m}$', 'M', '$\mathcal{H}_{ii}$',
                 '$\chi_{ii}$', 'X', 'npf', '$\epsilon_i$', '$\widetilde{p}$', 'sol',
                 'A', '$\epsilon_r$', '$\epsilon_{r2}$']

        aliases = ['M', 'D', 'P', 'G', 'T', 'd',
                   'Lambda', 'Lambda_inv', 'Gamma', 'sdata', 'least_norm',
                   'least_norm_solution', 'norm_bound', 'H_diag', 'chi_diag', 'X', 'npf', 'epsilon',
                   'least_norm_property', 'solution', 'A', 'relative_errors', 'relative_errors2']
    
        self.item_aliases = dict(zip(items, aliases))
        self.aliases_item = dict(zip(aliases, items))

        dependencies = [('G', 'Lambda'), ('Lambda', 'Lambda_inv'),
                        ('Lambda_inv', 'X'), ('T', 'Gamma'), ('G', 'Gamma'),
                        ('Gamma', 'X'), ('d', 'sdata'), ('Lambda_inv', 'sdata'),
                        ('sdata', 'least_norm'), ('sdata', 'least_norm_solution'),
                        ('M', 'G'), ('T', 'chi_diag'), ('chi_diag', 'H_diag'),
                        ('M', 'T'), ('least_norm', 'npf'), ('norm_bound', 'npf'),
                        ('D', 'G'), ('npf', 'epsilon'), ('X', 'H_diag'),
                        ('D', 'd'), ('P', 'T'), ('G', 'D'),
                        ('T', 'P'), ('Gamma', 'H_diag'), ('X', 'least_norm_property'),
                        ('d', 'least_norm_property'), ('H_diag', 'epsilon'), ('epsilon', 'solution'),
                        ('least_norm_property', 'solution'), ('X', 'A'), ('G', 'A'), 
                        ('least_norm_property', 'relative_errors'), ('epsilon', 'relative_errors'),
                        ('least_norm_property', 'relative_errors2'), ('epsilon', 'relative_errors2')]

        
        self.G = nx.DiGraph()
        self.G.add_nodes_from(items)
        self.G.add_edges_from([(self.aliases_item[src], self.aliases_item[end]) for src, end in dependencies])
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
            elif node in self._alias_to_item(['least_norm_property', 'least_norm_solution',
                                            'solution', 'A', 'epsilon', 'relative_errors',
                                            'relative_errors2']):
                node_colors.append('green')
            else:
                node_colors.append('skyblue')

        # Draw nodes with different colors
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(self.G, pos, node_size=700, node_color=node_colors, edgecolors='black', linewidths=1, alpha=0.8)

        # Draw edges and labels
        nx.draw_networkx_edges(self.G, pos, edge_color='gray', arrowsize=20, connectionstyle='arc3,rad=0.1', width=1.0)
        nx.draw_networkx_labels(self.G, pos, font_weight='bold', font_color='black', font_size=10)

        plt.show()

    def find_reachable_nodes(self, start_nodes, plot_reachable_graph=False):
        """
        Find and return all nodes reachable from the given start nodes.
        Optionally plot and save the reachable graph.

        Parameters:
        - start_nodes (list): The list of starting nodes.
        - plot_reachable_graph (bool): Whether to plot and save the reachable graph. Default is False.

        Returns:
        - set: A set of reachable nodes.
        """
        self.start_node = [self.aliases_item[start_node] for start_node in start_nodes]
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
                    reachable_node_colors.append('gold')  # Color for the starting node
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
            nx.draw_networkx_nodes(reachable_graph, pos, node_size=700, node_color=reachable_node_colors, edgecolors='black',
                                linewidths=1, alpha=0.8)

            # Draw edges and labels
            nx.draw_networkx_edges(reachable_graph, pos, edge_color='gray', arrowsize=20, connectionstyle='arc3,rad=0.1',
                                width=1.0)
            nx.draw_networkx_labels(reachable_graph, pos, font_weight='bold', font_color='black', font_size=10)

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
        - plot_dependent_tree (bool): Whether to plot and save the dependent tree. Default is False.

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
                    dependent_node_colors.append('gold')  # Color for the starting node
                elif node in self._alias_to_item(['norm_bound', 'M', 'D',
                                                'G', 'T', 'P', 'd']):
                    dependent_node_colors.append('red')
                elif node in self._alias_to_item(['P', 'M','relative_errors',
                                                'solution', 'A', 'epsilon',
                                                'relative_errors2']):
                    dependent_node_colors.append('green')
                else:
                    dependent_node_colors.append('skyblue')

            # Draw nodes with different colors
            plt.figure(figsize=(12, 8))
            nx.draw_networkx_nodes(dependent_tree, pos, node_size=700, node_color=dependent_node_colors, edgecolors='black',
                                linewidths=1, alpha=0.8)

            # Draw edges and labels
            nx.draw_networkx_edges(dependent_tree, pos, edge_color='gray', arrowsize=20, connectionstyle='arc3,rad=0.1',
                                width=1.0)
            nx.draw_networkx_labels(dependent_tree, pos, font_weight='bold', font_color='black', font_size=10)

            plt.show()

        return dependent_nodes


class Problem():
    def __init__(self, M: Space, D: Space, P: Space, G: Mapping,
                 T: Mapping, norm_bound: float, data: np.ndarray=None) -> None:
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
        self.M = M # model space
        self.D = D # data space
        self.P = P # property space
        self.G = G # model-data mapping
        self.T = T # model-property mapping
        self.data = data # data
        self.norm_bound = norm_bound # model norm bound

        self.G_adjoint = G.adjoint()
        self.T_adjoint = T.adjoint()

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

        self.fixed_items = ['M', 'D', 'P', 'G', 'T', 'd','norm_bound']
        self.dependencies = DependencyTree()

    def change_M(self, new_M: Space, new_G: Mapping, new_T: Mapping, new_norm_bound: float):
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
        self.G_adjoint = new_G.adjoint()
        self.T_adjoint = new_T.adjoint()

        dependent_nodes = self.dependencies.find_reachable_nodes(['M', 'G', 'T']) - set(self.fixed_items)
        for alias in dependent_nodes:
            setattr(self, alias, None)

    def change_D(self, new_D: Space, new_G: Mapping, new_data: np.ndarray):
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

        dependent_nodes = self.dependencies.find_reachable_nodes(['D', 'G', 'd']) - set(self.fixed_items)
        for alias in dependent_nodes:
            setattr(self, alias, None)
        
    def change_P(self, new_P: Space, new_T: Mapping):
        """Change the property space and property mapping

        Args:
            new_P (Space): New property space
            new_T (Mapping): New property mapping
        """        
        # Here I deal with the case when I want to introduce or remove a target
        # kernel
        self.P = new_P
        self.T = new_T

        dependent_nodes = self.dependencies.find_reachable_nodes(['P', 'T']) - set(self.fixed_items)
        for alias in dependent_nodes:
            setattr(self, alias, None)

    def change_T(self, new_T: Mapping, new_P: Space=None):
        """Change the property mapping and property space (optional)

        Args:
            new_T (Mapping): New property mapping
            new_P (Space, optional): New property space. Defaults to None.
        """        
        # This is specifically for the case when I want to change a target
        # kernel, but I don't change the number of kernels
        if new_P is None:
            self.T = new_T

            dependent_nodes = self.dependencies.find_reachable_nodes(['T']) - set(self.fixed_items)
            for alias in dependent_nodes:
                setattr(self, alias, None)
        else:
            self.T = new_T
            self.P = new_P

            dependent_nodes = self.dependencies.find_reachable_nodes(['T', 'P']) - set(self.fixed_items)
            for alias in dependent_nodes:
                setattr(self, alias, None)

    def change_G(self, new_G: Mapping, new_D: Space=None):
        """Change data mapping and data space (optional)

        Args:
            new_G (Mapping): New data mapping
            new_D (Space, optional): New data space. Defaults to None.
        """        
        if new_G is None:
            self.T = new_D

            dependent_nodes = self.dependencies.find_reachable_nodes(['T']) - set(self.fixed_items)
            for alias in dependent_nodes:
                setattr(self, alias, None)
        else:
            self.T = new_D
            self.P = new_G

            dependent_nodes = self.dependencies.find_reachable_nodes(['T', 'G']) - set(self.fixed_items)
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

        dependent_nodes = self.dependencies.find_reachable_nodes(['d']) - set(self.fixed_items)
        for alias in dependent_nodes:
            setattr(self, alias, None)

    def change_bound(self, new_bound: float):
        """Change the Norm Bound

        Args:
            new_bound (float): New norm bound
        """        
        self.norm_bound = new_bound

        dependent_nodes = self.dependencies.find_reachable_nodes(['norm_bound']) - set(self.fixed_items)
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
            raise TypeError('The current problem does not have any data. Please add data')   
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
        
        self.npf = np.sqrt(self.norm_bound**2 - self.least_norm**2)

    def _compute_least_norm_solution(self):
        if self.sdata is None:
            self._compute_sdata()
        self.least_norm_solution = self.G_adjoint.map(self.sdata)

    def _compute_Gamma(self):
        self.Gamma = self.T*self.G_adjoint

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
        self.H_diag = self.chi_diag - np.sum(self.X.matrix*self.Gamma.matrix, axis=1).reshape(self.chi_diag.shape)

    def _compute_epsilon(self):
        if self.npf is None:
            self._compute_norm_prefactor()
        if self.H_diag is None:
            self._compute_H_diag()
        try:
            self.epsilon = self.npf * np.sqrt(self.H_diag)
        except RuntimeWarning as rw:
            print('A')
    def _compute_least_norm_model_solution(self):
        if self.sdata is None:
            self._compute_sdata()
        self.least_norm_model_solution = self.G_adjoint.map(self.sdata)
    
    def _compute_least_norm_property(self):
        if self.data is None:
            raise TypeError('The current problem does not have any data. Please add data')   
        if self.X is None:
            self._compute_X()
        self.least_norm_property = self.X.map(self.data)

    def solve(self):
        if self.least_norm_property is None:
            self._compute_least_norm_property()
        if self.epsilon is None:
            self._compute_epsilon()
        self.solution = {'upper bound': self.least_norm_property + self.epsilon,
                         'lower bound': self.least_norm_property - self.epsilon}
    
    def _compute_resolving_kernels(self):
        if self.X is None:
            self._compute_X()
        self.A = self.X*self.G

    def _compute_relative_errors(self):
        if self.epsilon is None:
            self._compute_epsilon()
        if self.least_norm_property is None:
            self._compute_least_norm_property()
        self.relative_errors = 100 * self.epsilon / self.least_norm_property
    
    def _compute_relative_errros2(self):
        if self.epsilon is None:
            self._compute_epsilon()
        if self.least_norm_property is None:
            self._compute_least_norm_property()
        self.relative_errors2 = 100 * self.epsilon / (np.max(self.least_norm_property) - np.min(self.least_norm_property))

    def plot_solution(self, enquiry_points):
        # Will plot the property bounds, the least norm property, the resolving
        # kernels and the target kernels with a slider used to explore them. It
        # assumes that the problem is 1D and that the property vector contains
        # some property of the true model evaluated at only one position

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.5, subplot_titles=('A','B'))
        no_of_traces = 0 # by default

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
                        name='Resolving Kernel: ' + str(round_to_sf(enquiry_points[step], 2)),
                        x=map.kernels[step].domain.mesh,
                        y=map.kernels[step].evaluate(map.kernels[step].domain.mesh)
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
                args=[{'visible': [False] * (no_of_traces * self.P.dimension + 1)},
                      {'title': 'Slider'}],
            )
            for j in range(no_of_traces):
                step['args'][0]['visible'][no_of_traces*i + j] = True

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

    def plot_multi_widths_errors(self, domain: Domain, 
                                 enquiry_points, spreads, type):
        
        # Make sure we have everyhting computed well
        if self.A is None:
            self._compute_resolving_kernels()
        if type == 'absolute':
            if self.epsilon is None:
                self._compute_epsilon()
            errors = self.epsilon
        elif type == 'relative':
            if self.relative_errors is None:
                self._compute_relative_errors()
            errors = self.relative_errors
        elif type == 'relative2':
            if self.relative_errors2 is None:
                self._compute_relative_errros2()
            errors = self.relative_errors2
        else:
            raise ValueError('Error type must be absolute, relative, or relative2')
    
        # Compute exclusion zone
        N_enquiry_points = len(enquiry_points)
        N_spreads = len(spreads)
        combinations = list(product(enquiry_points, spreads))
        domain_min = domain.bounds[0][0]
        domain_max = domain.bounds[0][1]
        exclusion_zones = np.array([(((center + spread) > domain_max) or ((center - spread) < domain_min)) for center, spread in combinations])

        plt.figure('errrors', figsize=(12, 10))
        # Plotting initial image with adjustments
        bound_map = (errors.reshape(N_enquiry_points, N_spreads)).T
        exclusion_map = (exclusion_zones.reshape(N_enquiry_points, N_spreads)).T
        bound_map[exclusion_map] = 0

        plt.imshow(bound_map, norm=LogNorm(vmin=1, vmax=100))
        sns.set_theme()
        plt.set_cmap('Blues_r')
        plt.rc('font', size=14)  # Set font size

        # Setting colorbar and ticks
        formatter = LogFormatter(10, labelOnlyBase=False)
        ticks = [1, 5, 10, 100, 500, 1000]
        cb = plt.colorbar(ticks=ticks, format=formatter)
        cb.set_label('Relative Error bound as %', fontsize=14)

        # Setting y and x ticks with labels and rotation
        n_spreads = int(len(spreads) / 10) + 1
        n_locations = int(len(enquiry_points) / 10) + 1
        normalized_spreads = 100 * spreads / (domain_max - domain_min)
        plt.yticks(np.arange(0, len(spreads), n_spreads), [f'{spread:.1e}%' for i, spread in enumerate(normalized_spreads) if i % n_spreads == 0], fontsize=12)
        plt.xticks(np.arange(0, len(enquiry_points), n_locations), [f'{point:.1e}' for i, point in enumerate(enquiry_points) if i % n_locations == 0], rotation=20, fontsize=12)
        plt.xlabel('Radius [km]', fontsize=14)
        plt.title('Relative Error Bounds', fontsize=16)
        name = 'test'#f"{target_type}_{kernel_type}_Norm:{true_norm}_K:{N}_S:{N_spreads}.{f'{min_spread*100:.2f}'}_E:{N_enquiry_points}.pdf"

        # Creating the second figure
        plt.figure(figsize=(10, 8))
        img = plt.imshow(bound_map, norm=LogNorm(vmin=1, vmax=100))
        plt.colorbar().set_label('Relative Error bound as %', fontsize=14)
        plt.title('Relative Error Bounds', fontsize=16)
        plt.yticks(np.arange(0, len(spreads), n_spreads), [f'{spread:.1e}%' for spread in normalized_spreads[::n_spreads]], fontsize=12)
        plt.xticks(np.arange(0, len(enquiry_points), n_locations), [f'{point:.1e}' for point in enquiry_points[::n_locations]], rotation=20, fontsize=12)
        plt.xlabel('Radius [km]', fontsize=14)
        plt.grid(False)

        # Saving the plot
        plt.savefig(name, format='pdf')

        # Defining the rectangle for highlighting
        highlight_rect = plt.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(highlight_rect)

        # Function to round to the nearest pixel
        def snap_to_pixel(x, step):
            return int(round(x / step))

        # Function to highlight a pixel
        def highlight_pixel(i, j):
            highlight_rect.set_xy((i - 0.5, j - 0.5))

        # Function for mouse click event
        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                i = snap_to_pixel(event.xdata, 1)
                j = snap_to_pixel(event.ydata, 1)

                plt.figure(10)
                sns.set_theme(style='white')
                sns.set_palette('YlGnBu')
                blues = plt.get_cmap('YlGnBu')
                plt.title('Target vs Averaging kernels', fontsize=16)
                for target_mapping, resolving_mapping in zip(self.T.mappings, self.A.mappings):
                    plt.plot(domain.mesh, target_mapping.kernels[j + N_spreads * i].evaluate(domain.mesh)[1],
                              label='Target', linewidth=3, color='red')
                    plt.plot(domain.mesh, resolving_mapping.kernels[j + N_spreads * i].evaluate(domain.mesh)[1],
                              label='Resolving', linewidth=2, color=blues(0.6))
                plt.xlabel('Radius [km]', fontsize=15)
                plt.show()

                highlight_pixel(i, j)
                plt.draw()

        plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        plt.show()