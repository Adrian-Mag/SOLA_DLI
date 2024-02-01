import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

def create_dependency_tree():
    G = nx.DiGraph()

    # Add nodes representing items
    items = ['$\mathcal{M}$', '$\mathcal{D}$', '$\mathcal{P}$', 'G', 'T', 'd',
             '$\Lambda$', '$\Lambda^{-1}$', '$\Gamma$', '$\Lambda^{-1}d$', 
             '|$\widetilde{m}$|', '$\widetilde{m}$', 'M', '$\mathcal{H}_{ii}$',
             '$\chi_{ii}$', 'X']
    G.add_nodes_from(items)

    # Add edges representing dependencies
    dependencies = [('G', '$\Lambda$'), ('$\Lambda$', '$\Lambda^{-1}$'), 
                    ('$\Lambda^{-1}$', 'X'), ('T', '$\Gamma$'), ('G', '$\Gamma$'), 
                    ('$\Gamma$', 'X'), ('d', '$\Lambda^{-1}d$'), ('$\Lambda^{-1}$', '$\Lambda^{-1}d$'),
                    ('$\Lambda^{-1}d$', '|$\widetilde{m}$|'), ('$\Lambda^{-1}d$', '$\widetilde{m}$'),
                    ('$\mathcal{M}$', 'G'), ('T', '$\chi_{ii}$'), ('$\chi_{ii}$', '$\mathcal{H}_{ii}$'),
                    ('$\mathcal{M}$', 'T'), 
                    ('$\mathcal{D}$', 'G'),
                    ('$\mathcal{D}$', 'd'), ('$\mathcal{P}$', 'T'), ('G', '$\mathcal{D}$'), 
                    ('T', '$\mathcal{P}$')]
    G.add_edges_from(dependencies)

    return G

def plot_dependency_tree(G):
    pos = graphviz_layout(G, prog='dot')

    # Set node colors
    node_colors = ['red' if node in ['M', '$\mathcal{M}$', '$\mathcal{D}$', 
                                     'G', 'T', '$\mathcal{P}$', 'd'] else 'skyblue' for node in G.nodes]

    # Draw nodes with different colors
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, edgecolors='black', linewidths=1, alpha=0.8)

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=20, connectionstyle='arc3,rad=0.1', width=1.0)
    nx.draw_networkx_labels(G, pos, font_weight='bold', font_color='black', font_size=10)

    plt.show()

def main():
    # Create the dependency tree graph
    dependency_tree = create_dependency_tree()

    # Plot the dependency tree
    plot_dependency_tree(dependency_tree)

if __name__ == "__main__":
    main()
