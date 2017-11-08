# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:59:39 2017

@author: jn107154
"""

import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(graph, labels=None, edge_text_pos=0.5):

    # extract nodes from graph
    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

    # create networkx graph
    G=nx.Graph()

    # add nodes
    for node in nodes:
        G.add_node(node)

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # draw graph
    pos = nx.shell_layout(G)
    nx.draw(G, pos)
    graph_pos=nx.shell_layout(G)
    nx.draw_networkx_labels(G, graph_pos)
    if labels is None:
        labels = range(len(graph))
    
    
    edge_labels = dict(zip(graph, labels))
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
                                 label_pos=edge_text_pos)

    # show graph
    plt.show()


"""
# draw example
graph = [(20, 21),(21, 22),(22, 23), (23, 24),(24, 25), (25, 20)]
draw_graph(graph)


graph2 = [('a', 'b'), ('b', 'c'), ('a', 'c'), ('d', 'a')]
labs = list("7248")
draw_graph(graph2, labels = labs)
"""
