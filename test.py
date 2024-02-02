import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

class DependencyTree:
    def __init__(self):
        items = ['$\mathcal{M}$', '$\mathcal{D}$', '$\mathcal{P}$', 'G', 'T', 'd',
             '$\Lambda$', '$\Lambda^{-1}$', '$\Gamma$', '$\Lambda^{-1}d$',
             '|$\widetilde{m}$|', '$\widetilde{m}$', 'M', '$\mathcal{H}_{ii}$',
             '$\chi_{ii}$', 'X', 'npf', '$\epsilon_i$', '$\widetilde{p}$', 'sol',
             'A']

        self.aliases = ['model_space', 'data_space', 'property_space', 'G', 'T', 'd', 
                'Lambda', 'Lambda_inv', 'Gamma', 'sdata', 'least_norm',
                'least_norm_solution', 'M', 'H', 'chi', 'X', 'npf', 'epsilon',
                'least_norm_property', 'solution', 'resolving_kernel']
        self.item_aliases = dict(zip(self.aliases, items))
        self.aliases_item = dict(zip(items, self.aliases))

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

    def print_node_names(self):
        print(self.aliases)

    def plot_dependency_tree(self, save_filename='tree.png'):
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

def main():
    # Specify items and dependencies
    

    # Create the DependencyTree instance
    dependency_tree = DependencyTree()

    # Plot the dependency tree
    dependency_tree.plot_dependency_tree()

    # Specify the starting node
    start_node = 'solution'  # Choose your starting node

    # Find reachable nodes
    reachable_nodes = dependency_tree.find_reachable_nodes(start_node)
    print(f"Nodes reachable from {start_node}: {reachable_nodes}")

    # Find dependent nodes and optionally plot the dependent tree
    dependent_nodes = dependency_tree.find_dependent_nodes(start_node, plot_dependent_tree=True)
    print(f"Nodes on which {start_node} depends: {dependent_nodes}")

if __name__ == "__main__":
    main()
