from core.main_classes.spaces import PCb, DirectSumSpace, RN
from core.main_classes.mappings import *
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

class DependencyTree:
    """
    A class representing a dependency tree.

    Attributes:
    - item_aliases (dict): A dictionary mapping item aliases to their original names.
    - aliases_item (dict): A dictionary mapping original item names to their aliases.
    - G (DiGraph): The directed graph representing dependencies between items.
    - start_node (str): The starting node for operations like finding reachable or dependent nodes.

    Methods:
    - plot_dependency_tree(save_filename='tree.png'): Plot the entire dependency tree and save the plot.
    - find_reachable_nodes(start_node): Find and return all nodes reachable from the given start node.
    - find_dependent_nodes(start_node, plot_dependent_tree=False, save_filename='dependent_tree.png'):
      Find and return all nodes dependent on the given start node. Optionally plot and save the dependent tree.
    """

    def __init__(self):
        items = ['$\mathcal{M}$', '$\mathcal{D}$', '$\mathcal{P}$', 'G', 'T', 'd',
                 '$\Lambda$', '$\Lambda^{-1}$', '$\Gamma$', '$\Lambda^{-1}d$',
                 '|$\widetilde{m}$|', '$\widetilde{m}$', 'M', '$\mathcal{H}_{ii}$',
                 '$\chi_{ii}$', 'X', 'npf', '$\epsilon_i$', '$\widetilde{p}$', 'sol',
                 'A']

        aliases = ['model_space', 'data_space', 'property_space', 'G', 'T', 'd',
                   'Lambda', 'Lambda_inv', 'Gamma', 'sdata', 'least_norm',
                   'least_norm_solution', 'M', 'H', 'chi', 'X', 'npf', 'epsilon',
                   'least_norm_property', 'solution', 'resolving_kernel']
        self.item_aliases = dict(zip(aliases, items))
        self.aliases_item = dict(zip(items, aliases))

        dependencies = [('G', '$\Lambda$'), ('$\Lambda$', '$\Lambda^{-1}$'),
                        ('$\Lambda^{-1}$', 'X'), ('T', '$\Gamma$'), ('G', '$\Gamma$'),
                        ('$\Gamma$', 'X'), ('d', '$\Lambda^{-1}d$'), ('$\Lambda^{-1}$', '$\Lambda^{-1}d$'),
                        ('$\Lambda^{-1}d$', '|$\widetilde{m}$|'), ('$\Lambda^{-1}d$', '$\widetilde{m}$'),
                        ('$\mathcal{M}$', 'G'), ('T', '$\chi_{ii}$'), ('$\chi_{ii}$', '$\mathcal{H}_{ii}$'),
                        ('$\mathcal{M}$', 'T'), ('|$\widetilde{m}$|', 'npf'), ('M', 'npf'),
                        ('$\mathcal{D}$', 'G'), ('npf', '$\epsilon_i$'), ('X', '$\mathcal{H}_{ii}$'),
                        ('$\mathcal{D}$', 'd'), ('$\mathcal{P}$', 'T'), ('G', '$\mathcal{D}$'),
                        ('T', '$\mathcal{P}$'), ('$\Gamma$', '$\mathcal{H}_{ii}$'), ('X', '$\widetilde{p}$'),
                        ('d', '$\widetilde{p}$'), ('$\mathcal{H}_{ii}$', '$\epsilon_i$'), ('$\epsilon_i$', 'sol'),
                        ('$\widetilde{p}$', 'sol'), ('X', 'A'), ('G', 'A')]
        self.G = nx.DiGraph()
        self.G.add_nodes_from(items)
        self.G.add_edges_from(dependencies)
        self.start_node = None

    def plot_dependency_tree(self, save_filename='tree.png'):
        """
        Plot the entire dependency tree and optionally save the plot.

        Parameters:
        - save_filename (str): The filename to save the plot. Default is 'tree.png'.
        """
        pos = graphviz_layout(self.G, prog='dot')

        # Set node colors and edge colors
        node_colors = []
        edge_colors = []
        for node in self.G.nodes:
            if node == self.start_node:
                node_colors.append('gold')  # Color for the starting node
            elif node in ['M', '$\mathcal{M}$', '$\mathcal{D}$',
                          'G', 'T', '$\mathcal{P}$', 'd']:
                node_colors.append('red')
            elif node in ['$\widetilde{p}$', '$\widetilde{m}$',
                          'sol', 'A', '$\epsilon_i$']:
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

    def find_reachable_nodes(self, start_node):
        """
        Find and return all nodes reachable from the given start node.

        Parameters:
        - start_node (str): The starting node.

        Returns:
        - set: A set of reachable nodes.
        """
        self.start_node = self.item_aliases[start_node]
        reachable_nodes = set()

        def dfs(node):
            reachable_nodes.add(node)
            for neighbor in self.G.neighbors(node):
                if neighbor not in reachable_nodes:
                    dfs(neighbor)

        dfs(self.item_aliases[start_node])
        return reachable_nodes

    def find_dependent_nodes(self, start_node, plot_dependent_tree=False, save_filename='dependent_tree.png'):
        """
        Find and return all nodes dependent on the given start node.
        Optionally plot and save the dependent tree.

        Parameters:
        - start_node (str): The starting node.
        - plot_dependent_tree (bool): Whether to plot and save the dependent tree. Default is False.
        - save_filename (str): The filename to save the dependent tree plot. Default is 'dependent_tree.png'.

        Returns:
        - set: A set of dependent nodes.
        """
        self.start_node = self.item_aliases[start_node]
        dependent_nodes = set()

        def dfs(node):
            dependent_nodes.add(node)
            for predecessor in self.G.predecessors(node):
                if predecessor not in dependent_nodes:
                    dfs(predecessor)

        dfs(self.item_aliases[start_node])

        if plot_dependent_tree:
            dependent_tree = self.G.subgraph(dependent_nodes)
            pos = graphviz_layout(dependent_tree, prog='dot')

            # Set node colors and edge colors
            dependent_node_colors = []
            dependent_edge_colors = []
            for node in dependent_tree.nodes:
                if node == self.start_node:
                    dependent_node_colors.append('gold')  # Color for the starting node
                elif node in ['M', '$\mathcal{M}$', '$\mathcal{D}$',
                              'G', 'T', '$\mathcal{P}$', 'd']:
                    dependent_node_colors.append('red')
                elif node in ['$\widetilde{p}$', '$\widetilde{m}$',
                              'sol', 'A', '$\epsilon_i$']:
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
        self.M = M
        self.D = D
        self.P = P
        self.G = G
        self.T = T
        self.data = data
        self.norm_bound = norm_bound

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
        self.least_norm_model_solution = None
        self.least_norm_property_solution = None
        self.solution = None

        aliases = ['model_space', 'data_space', 'property_space', 'G', 'T', 'd',
                   'Lambda', 'Lambda_inv', 'Gamma', 'sdata', 'least_norm',
                   'least_norm_solution', 'M', 'H', 'chi', 'X', 'npf', 'epsilon',
                   'least_norm_property', 'solution', 'resolving_kernel']

    def change_M(self, new_M: Space, new_G: Mapping, new_T: Mapping):
        self.M = new_M
        self.G = new_G
        self.T = new_T
        self.G_adjoint = new_G.adjoint()
        self.T_adjoint = new_T.adjoint()

        
    
    def _compute_Lambda(self):
        self.Lambda = self.G._compute_GramMatrix()

    def _compute_Lambda_inv(self):
        if self.Lambda is not None:
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
        self.least_norm = self.D.inner_product(self.data, self.sdata)

    def _compute_norm_prefactor(self):
        if self.least_norm is None:
            self._compute_least_norm()
        self.npf = np.sqrt(self.M**2 - self.least_norm**2)

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
        self.H_diag = self.chi_diag - np.sum(self.X.matrix*self.Gamma.matrix.T, axis=1)

    def _compute_epsilon(self):
        if self.npf is None:
            self._compute_norm_prefactor()
        if self.H_diag is None:
            self._compute_H_diag()
        self.epsilon = self.npf * np.sqrt(self.H_diag)
    
    def _compute_least_norm_model_solution(self):
        if self.sdata is None:
            self._compute_sdata()
        self.least_norm_model_solution = self.G_adjoint.map(self.sdata)
    
    def _compute_least_norm_property_solution(self):
        if self.data is None:
            raise TypeError('The current problem does not have any data. Please add data')   
        if self.X is None:
            self._compute_X()
        self.least_norm_property_solution = self.X.map(self.data)

    def solve(self):
        if self.least_norm_property_solution is None:
            self._compute_least_norm_property_solution()
        if self.epsilon is None:
            self._compute_epsilon()
        self.solution = {'upper bound': self.least_norm_property_solution + self.epsilon,
                         'lower bound': self.least_norm_property_solution - self.epsilon}